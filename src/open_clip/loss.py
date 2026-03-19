import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss
    


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        
        clip_loss = torch.tensor(0)
        
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):

    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
            num_captions=1,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}
        self.num_captions = num_captions

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels

        if self.num_captions > 1:
            labels = labels.unsqueeze(-1)
            labels = labels.repeat(1, 1, self.num_captions)
            labels = labels.reshape(num_logits, -1)

        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / text_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):

        per_gpu_loss = []
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)
        per_gpu_loss.append(loss.data.item())

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        curr_loss = self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                        loss = curr_loss + loss
                        per_gpu_loss.append(curr_loss.item())

                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    curr_loss = self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    loss = curr_loss + loss
                    per_gpu_loss.append(curr_loss.item())
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    curr_loss = self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    loss = curr_loss + loss
                    per_gpu_loss.append(curr_loss.item())
                    text_features_to_right = text_features_from_left

        # return {"contrastive_loss": loss, "per_gpu_loss": per_gpu_loss} if output_dict else loss
        return {"contrastive_loss": loss} if output_dict else loss


class FilteredSigLipLoss(nn.Module):
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
            pseudo_txt_txt_th=0.90,
            num_captions=1
    ):
        super().__init__()

        self.pseudo_txt_txt_th = pseudo_txt_txt_th

        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir
        self.num_captions = num_captions

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}


    def _duplicate_labels(self, labels):
        num_logits = labels.shape[0]
        labels = labels.unsqueeze(-1)
        labels = labels.repeat(1, 1, self.num_captions)
        labels = labels.reshape(num_logits, -1)
        return labels
    
    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels

        if self.num_captions > 1:
            labels = self._duplicate_labels(labels)
        return labels
    
    def get_pseudo_ground_truth(self, 
            teacher_image_features,
            teacher_image_features_recv,
            teacher_text_features,
            teacher_text_features_recv,
        ) -> torch.Tensor:

        num_images = teacher_image_features.shape[0]
        num_txt = teacher_text_features.shape[0]

        # Txt x Txt based peuso labels
        sim_scores_txt_txt = teacher_text_features @ teacher_text_features_recv.T
        labels_txt_txt = (sim_scores_txt_txt > self.pseudo_txt_txt_th)
        if self.num_captions > 1:
            # Ntxt x Ntxt -> Nimg x Ntxt
            labels_txt_txt = labels_txt_txt.reshape(num_images, self.num_captions, -1).sum(1)

        labels_txt_txt = 2. * labels_txt_txt - 1.

        return labels_txt_txt


    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self,
              image_features,
              text_features,
              logit_scale,
              logit_bias,
              teacher_image_features,
              teacher_image_features_recv,
              teacher_text_features,
              teacher_text_features_recv,
              use_pseudo_labels=False,
              negative_only=False
        ):
        
        if use_pseudo_labels:
            labels = self.get_pseudo_ground_truth(
                teacher_image_features=teacher_image_features,
                teacher_image_features_recv=teacher_image_features_recv,
                teacher_text_features=teacher_text_features,
                teacher_text_features_recv=teacher_text_features_recv,
            )
        else:
            labels = self.get_ground_truth(
                image_features.device,
                image_features.dtype,
                image_features.shape[0],
                negative_only=negative_only,
            )
        
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        loss = -F.logsigmoid(labels * logits).sum() / text_features.shape[0]

        return loss

    def forward(self, 
        image_features, 
        text_features, 
        logit_scale, 
        logit_bias, 
        dist_image_features,
        dist_text_features,
        dist_logit_scale,
        dist_logit_bias,
        output_dict=False
    ):
        
        per_gpu_loss = []
        teacher_image_features = dist_image_features
        teacher_text_features = dist_text_features

        loss = self._loss(
                image_features=image_features,
                text_features=text_features,
                logit_scale=logit_scale,
                logit_bias=logit_bias, 
                teacher_image_features=None,
                teacher_image_features_recv=None,
                teacher_text_features=None,
                teacher_text_features_recv=None,
                use_pseudo_labels=False,
        )
        
        per_gpu_loss.append(loss.item())

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                teacher_text_features_to_right = teacher_text_features_to_left = teacher_text_features
                teacher_image_features_to_right = teacher_image_features_to_left = teacher_image_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)

                for i in range(num_bidir):

                    all_text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    all_teacher_text_features_recv = neighbour_exchange_bidir(
                        left_rank,
                        right_rank,
                        teacher_text_features_to_left,
                        teacher_text_features_to_right,
                    )
                    
                    all_teacher_image_features_recv = neighbour_exchange_bidir(
                        left_rank,
                        right_rank,
                        teacher_image_features_to_left,
                        teacher_image_features_to_right,
                    )
                    

                    for text_features_recv, teacher_text_feats_recv, teacher_image_feats_recv in zip(all_text_features_recv, 
                                                                                                     all_teacher_text_features_recv, 
                                                                                                     all_teacher_image_features_recv):
                        curr_loss = self._loss(
                            image_features=image_features,
                            text_features=text_features_recv,
                            logit_scale=logit_scale,
                            logit_bias=logit_bias,
                            teacher_image_features=teacher_image_features,
                            teacher_image_features_recv=teacher_image_feats_recv,
                            teacher_text_features=teacher_text_features,
                            teacher_text_features_recv=teacher_text_feats_recv,
                            use_pseudo_labels=True,
                        )
                        loss = loss + curr_loss
                        per_gpu_loss.append(curr_loss.item())

                    text_features_to_left, text_features_to_right = all_text_features_recv
                    teacher_text_features_to_left, teacher_text_features_to_right = all_teacher_text_features_recv
                    teacher_image_features_to_left, teacher_image_features_to_right = all_teacher_image_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)
                    
                    teacher_text_features_recv = neighbour_exchange(
                        left_rank, right_rank, teacher_text_features_to_right)
                    
                    teacher_image_features_recv = neighbour_exchange(
                        left_rank, right_rank, teacher_image_features_to_right)

                    curr_loss = self._loss(
                        image_features=image_features,
                        text_features=text_features_recv,
                        logit_scale=logit_scale,
                        logit_bias=logit_bias,
                        teacher_image_features=teacher_image_features,
                        teacher_image_features_recv=teacher_image_features_recv,
                        teacher_text_features=teacher_text_features,
                        teacher_text_features_recv=teacher_text_features_recv,
                        use_pseudo_labels=True,
                    )
                    loss = loss + curr_loss
                    per_gpu_loss.append(curr_loss.item())

            else:
                text_features_to_right = text_features
                teacher_text_features_to_right = teacher_text_features
                teacher_image_features_to_right = teacher_image_features

                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)
                    
                    teacher_text_features_from_left = neighbour_exchange(
                        left_rank, right_rank, teacher_text_features_to_right)
                    
                    teacher_image_features_from_left = neighbour_exchange(
                        left_rank, right_rank, teacher_image_features_to_right)

                    curr_loss = self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        teacher_text_features,
                        teacher_text_features_from_left,
                        teacher_image_features,
                        teacher_image_features_from_left,
                        use_pseudo_labels=True,
                    )

                    loss = loss + curr_loss
                    per_gpu_loss.append(curr_loss.item())

                    text_features_to_right = text_features_from_left
                    teacher_text_features_to_right = teacher_text_features_from_left
                    teacher_image_features_to_right = teacher_image_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss
    

####################################################################################################
#####        cross-modal losses ####################################################################

def func_attention(query, context, smooth, eps=1e-8, raw_feature_norm="clipped_l2norm"):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)


    # Get attention
    # --> (batch_query, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = F.softmax(attn, dim=2)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = F.normalize(attn, dim=2, p=2, eps=eps)
    elif raw_feature_norm == "clipped_l2norm":
        attn = F.leaky_relu(attn, 0.1)
        attn = F.normalize(attn, dim=2, p=2, eps=eps)
    elif raw_feature_norm == "l1norm":
        attn = F.normalize(attn, dim=2, p=1, eps=eps)
    elif raw_feature_norm == "clipped_l1norm":
        attn = F.leaky_relu(attn, 0.1)
        attn = F.normalize(attn, dim=2, p=1, eps=eps)
    elif raw_feature_norm == "clipped":
        attn = F.leaky_relu(attn, 0.1)
    elif raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax()(attn*smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


######################################################################
def all_gather_tensor(data_tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    gathered_tensors = [torch.zeros_like(data_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, data_tensor)
    gathered_tensors[rank] = data_tensor
    all_tensors = torch.cat(gathered_tensors, dim=0)
    return all_tensors


def all_gather_variable_size_tensor(data_tensor: torch.Tensor, rank: int, world_size: int, gather_indices=False, return_slices=False) -> torch.Tensor:
    data_tensor = data_tensor.contiguous()

    device = data_tensor.device
    # gather tensor batch-size across processes
    size_tens = torch.tensor([data_tensor.shape[0]], dtype=torch.int64, device=device)
    size_tens = all_gather_tensor(size_tens, rank=rank, world_size=world_size).cpu()

    # pad & collect tensors across processes
    max_size = size_tens.max()

    padded = torch.empty(max_size, *data_tensor.shape[1:],
                         dtype=data_tensor.dtype,
                         device=device)
    padded[:data_tensor.shape[0]] = data_tensor

    ag = all_gather_tensor(padded, rank=rank, world_size=world_size)

    # collect tensors of original sizes
    slices = []
    for i, sz in enumerate(size_tens):
        start_idx = i * max_size
        end_idx = start_idx + sz.item()

        if end_idx >= start_idx:
            slices.append(ag[start_idx:end_idx])
    if gather_indices:
        # special case
        assert len(ag.shape) == 1, f"the aggregated tensor has shape {len(ag.shape)} dimensions, while onle 1-dim is valid."
        for i in range(1, len(slices)):
            prev_slice = slices[i-1]
            prev_max_i = prev_slice.max() + 1
            # increase the indices of current slice by prev_max_i
            slices[i] += prev_max_i

    if return_slices:
        return slices

    ret = torch.cat(slices, dim=0)
    # done, return
    return ret


def gather_nounphrases(nounphrases_features,
                       nounphrases_indices,
                       gather_with_grad=False,
                       rank=0,
                       world_size=1):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'

    if gather_with_grad:
        all_nounphrases_features = [torch.distributed.nn.all_gather(nounphrases_features)]
    else:
        all_nounphrases_features = all_gather_variable_size_tensor(nounphrases_features, rank=rank, world_size=world_size)
    
    all_nounphrases_indices = all_gather_variable_size_tensor(nounphrases_indices, rank=rank, world_size=world_size, gather_indices=True)
    return all_nounphrases_features, all_nounphrases_indices

    
####################################################################################################

class C2LIP_Loss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            #####################
            # Nounphrase loss
            npc_loss=True,
            npc_loss_scale=1.0,
            ##########################
            xac_loss=True,
            xac_loss_scale=0.01,
            ##########################
            loss_feature_norm=False
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

        # Nounphrase loss
        self.npc_loss = npc_loss
        self.npc_loss_scale = npc_loss_scale
        self.xac_loss = xac_loss
        self.xac_loss_scale = xac_loss_scale
        
        self.loss_feature_norm = loss_feature_norm

        
    def forward(self, 
                image_features, 
                text_features, 
                logit_scale, 
                logit_bias, 
                output_dict=False,
                # NP inputs
                nounphrases_features=None,
                nounphrases_indices=None,
                # for XAC
                image_tokens=None, ):
        """
        cross-modal ranking loss doesn't support local_loss and use_horovod 

        Different Losses:
            - hard negative: standard clip contrastive loss, assuming hard-negatives as extra negative for computing logits_per_image, logits_per_text is the same as clip
            - imc_loss: standard clip contrastive loss + contrastive loss on text embeddings (between ground truth caption embedding and hard-negative caption embedding)
            - cmr_loss: standard clip contrastive loss + rank loss between gt pair and hg pair
        """
        device = image_features.device

        total_loss = 0.
        
        ## standard SigLIP contrastive loss
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            
            gt_all_text_features = all_text_features
            
            if self.local_loss:
                # siglip 
                logits_per_image = torch.matmul(image_features, all_text_features.t()) * logit_scale +logit_bias
                #logits_per_text = torch.matmul(text_features, all_image_features.t()) * logit_scale +logit_bias
            else:
                if self.loss_feature_norm:
                    normalized_gt_all_text_features = F.normalize(gt_all_text_features, dim=-1)
                    normalized_all_image_features = F.normalize(all_image_features, dim=-1)
                    logits_per_image = torch.matmul(normalized_all_image_features, normalized_gt_all_text_features.t()) * logit_scale +logit_bias
                else:
                    normalized_gt_all_text_features = gt_all_text_features
                    normalized_all_image_features = all_image_features
                    logits_per_image = torch.matmul(all_image_features, gt_all_text_features.t()) * logit_scale +logit_bias
        else:
            # not updating very long time
            gt_text_features = text_features

            if self.loss_feature_norm:
                normalized_image_features = F.normalize(image_features, dim=-1)
                normalized_gt_text_features = F.normalize(gt_text_features, dim=-1)
            else:
                normalized_image_features = image_features
                normalized_gt_text_features = gt_text_features
            
            logits_per_image = torch.matmul(normalized_image_features, normalized_gt_text_features.t()) * logit_scale + logit_bias
        
        labels = torch.zeros_like(logits_per_image) - 1 # -1 (negative) everywhere except the diagonal
        labels.fill_diagonal_(1.) # diagonal elements are 1 - positive
        contrastive_loss = -F.logsigmoid(labels * logits_per_image).sum() / logits_per_image.shape[1] #-torch.mean(F.logsigmoid(labels * logits_per_image))
            
        loss_outputs = {"contrastive_loss": contrastive_loss}
        total_loss += contrastive_loss
        
        if self.npc_loss or self.xac_loss:
            if self.world_size > 1:
                # gather features
                if self.xac_loss:
                    assert image_tokens is not None
                    image_tokens = all_gather_tensor(image_tokens, rank=self.rank, world_size=self.world_size)
                nounphrases_features = all_gather_variable_size_tensor(nounphrases_features, rank=self.rank, world_size=self.world_size)
                nounphrases_indices = all_gather_variable_size_tensor(nounphrases_indices, rank=self.rank, world_size=self.world_size, gather_indices=True)
            
            if self.npc_loss:
                if self.loss_feature_norm:
                    normalized_nounphrases_features = F.normalize(nounphrases_features, dim=-1)
                    logits_per_image = torch.matmul(image_features, normalized_nounphrases_features.t()) * logit_scale
                else:
                    logits_per_image = torch.matmul(image_features, nounphrases_features.t()) * logit_scale
                if logit_bias is not None:
                    logits_per_image += logit_bias
                labels = - torch.ones_like(logits_per_image)
                for col, idx in enumerate(nounphrases_indices):
                    labels[idx][col] = 1 # set positive if nounphrase appears in image
                npc_loss = -F.logsigmoid(labels * logits_per_image).sum() / logits_per_image.shape[1]
                loss_outputs["npc_loss"] = npc_loss
                total_loss += npc_loss * self.npc_loss_scale
            
            if self.xac_loss:
                M = image_tokens.shape[0] # num of images
                N = nounphrases_features.shape[0] # num of nounphrases
                expanded_np_features = nounphrases_features.unsqueeze(1) # (N, 1, feat_dim)
                
                logits = []
                
                for i in range(N):
                    # (M, 1, feat_dim)
                    query = expanded_np_features[i:i+1].expand(M, -1, -1)
                    # (M, 1, feat_dim)
                    weighted_features_, _ = func_attention(query=query, context=image_tokens, raw_feature_norm="no_norm", smooth=1.)
                    # weighted_features_[i,:,:] -> pooled features of an image
                    weighted_features_ = F.normalize(weighted_features_, dim=-1, eps=1e-8)
                    weighted_features_ = weighted_features_.squeeze(1) # (batch, feat_dim)
                    if self.loss_feature_norm:
                        query_features = F.normalize(nounphrases_features[i:i+1], dim=-1) #(1, feat_dim)
                    else:
                        query_features = nounphrases_features[i:i+1] #(1, feat_dim)
                    query_i_logits = (weighted_features_ @ query_features.T) #(M, 1) similarity of all images -> 1 caption
                    logits.append(query_i_logits)

                logits = torch.cat(logits, dim=1)

                if logit_scale is not None:
                    logits *= logit_scale
                if logit_bias is not None:
                    logits += logit_bias
                
                labels = - torch.ones_like(logits)
                for col, idx in enumerate(nounphrases_indices):
                    labels[idx][col] = 1 # set positive if nounphrase appears in image
                xac_loss = -F.logsigmoid(labels * logits).sum() / logits.shape[1]
                loss_outputs["xac_loss"] = xac_loss
                total_loss += xac_loss * self.xac_loss_scale

        #return total_loss, thresholds, cmr_loss, imc_loss
        if output_dict:
            loss_outputs["loss"] = total_loss
            return loss_outputs
        
        forward_outputs = [total_loss, contrastive_loss]
        if self.npc_loss:
            forward_outputs.append(npc_loss)
        if self.xac_loss:
            forward_outputs.append(xac_loss)
        return tuple(forward_outputs)