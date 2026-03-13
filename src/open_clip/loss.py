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
    
###############################################################################################################
def gather_features_da(
        image_features,
        text_features,
        valid_caption_mask,
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
        if valid_caption_mask is not None:
            all_valid_caption_mask = hvd.allgather(valid_caption_mask)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
            if valid_caption_mask is not None:
                all_valid_caption_mask=torch.cat(torch.distributed.nn.all_gather(valid_caption_mask), dim=0)
            else:
                all_valid_caption_mask = None
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if valid_caption_mask is not None:
                gathered_valid_caption_mask = [torch.zeros_like(valid_caption_mask) for _ in range(world_size)]
                dist.all_gather(gathered_valid_caption_mask, valid_caption_mask)
            else:
                gathered_valid_caption_mask = None
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                if valid_caption_mask is not None:
                    gathered_valid_caption_mask[rank] = valid_caption_mask
                
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)
            if valid_caption_mask is not None:
                all_valid_caption_mask = torch.cat(gathered_valid_caption_mask, dim=0)
            else:
                all_valid_caption_mask = None

    return all_image_features, all_text_features, all_valid_caption_mask


def get_cmr_loss(gt_logits_per_image:torch.Tensor, 
                 da_logits_per_image:torch.Tensor, 
                 valid_caption_mask, thresholds:torch.Tensor, 
                 threshold_type:str="mean") -> torch.Tensor:
    # calculating cmr loss
    gt_similarity = gt_logits_per_image.diag().reshape(-1,1).expand(da_logits_per_image.shape)
    # gt_similarity=gt_logits_per_image.gather(0,torch.arange(min(gt_logits_per_image.shape),device=gt_logits_per_image.device).reshape(1,-1)).reshape(min(gt_logits_per_image.shape),1).expand(da_logits_per_image.shape)
    cmr_loss = nn.functional.relu((thresholds+da_logits_per_image-gt_similarity))*valid_caption_mask

    # updating thresholds
    if threshold_type == 'mean':
        mask = da_logits_per_image!=0
        average_similarity_for_types = (da_logits_per_image*mask).sum(dim=0)/mask.sum(dim=0)
        thresholds = (gt_similarity.mean(0)-average_similarity_for_types).expand(gt_similarity.shape)
        thresholds = thresholds.detach()
    elif threshold_type == 'max':
        thresholds, max_indices = (gt_similarity*valid_caption_mask-da_logits_per_image).max(0)
        thresholds = thresholds.expand(gt_similarity.shape)/5
        thresholds = thresholds.detach()
    return cmr_loss.mean(),thresholds


def get_imc_loss(embedding_matrix:torch.Tensor):
    """
    gt_logits_per_image: standard clip similarity matrix, diag is true gt similarity value : shape [batch_size,5xbatch_size]
    embedding_matrix: extra similarity matrix served as denominator in clip loss
    """
    bs = embedding_matrix.shape[0]
    embedding_matrix = embedding_matrix.reshape(bs, bs, -1) # [batch_size, batch_size, 4]
    vecs = []
    for i in range(bs):
        vecs.append(embedding_matrix[i,i,:])
    embedding_matrix = torch.stack(vecs) # [batch_size, 4]
    labels = torch.zeros(embedding_matrix.shape[0], device=embedding_matrix.device, dtype=torch.long)
    imc_loss = F.cross_entropy(embedding_matrix, labels)
    return imc_loss

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


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    # w12 = torch.sum(x1 * x2, dim)
    # x11 = torch.norm(x1, 2, dim)
    # x22 = torch.norm(x2, 2, dim)
    # return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
    x11 = F.normalize(x1, 2, dim, eps=eps)
    x22 = F.normalize(x2, 2, dim, eps=eps)
    w12 = torch.sum(x11 * x22, dim)
    return w12



def xattn_score_t2i(images, captions, cap_lens, raw_feature_norm="clipped_l2norm", agg_func='LogSumExp', lambda_lse=6., lambda_softmax=9.):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        if n_word == 0:
            row_sim = torch.zeros((n_image, 1), dtype=images.dtype, device=images.device)
            similarities.append(row_sim)
            continue
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext, attn = func_attention(cap_i_expand, images, raw_feature_norm=raw_feature_norm, smooth=lambda_softmax)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        if agg_func == 'LogSumExp' or agg_func == "LogSumExp_norm":
            row_sim.mul_(lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/lambda_lse
            if agg_func == "LogSumExp_norm":
                # normalize by the number of words
                row_sim = row_sim - torch.log(n_word)
        elif agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    
    return similarities


def calculate_similarity_scores(image_tokens, text_tokens, text_mask, 
                                 raw_feature_norm="clipped_l2norm", 
                                 agg_func='LogSumExp', 
                                 lambda_lse=6., lambda_softmax=9.,
                                 use_hard_negative_text=False):
    if text_mask[0][0] == 0:
        text_mask = text_mask[:,1:]
    
    bsize = image_tokens.shape[0]
    if use_hard_negative_text:
        gt_text_tokens = text_tokens[:bsize]
        gt_text_mask = text_mask[:bsize]
        da_text_tokens = text_tokens[bsize:] #(batch*4, feat_dim)
        da_text_mask = text_mask[bsize:] # (batch*4, text_len)
    else:
        assert bsize == text_tokens.shape[0]
        gt_text_tokens = text_tokens
        gt_text_mask = text_mask

    # compute GT image-sentence score matrix
    gt_text_lengths = torch.sum(gt_text_mask, dim=1).to(text_tokens.device)    
    
    gt_scores = xattn_score_t2i(image_tokens, gt_text_tokens, gt_text_lengths, 
                             raw_feature_norm=raw_feature_norm, agg_func=agg_func,
                             lambda_lse=lambda_lse, lambda_softmax=lambda_softmax)

    # assert not torch.isnan(gt_scores).any()
    
    # compute DA image-sentence score matrix
    da_scores = []
    da_valid_mask = []
    if use_hard_negative_text:
        for i in range(bsize):
            cur_image_tokens = image_tokens[i:i+1] # (1, seq_len, feat_dim)
            cur_da_text_tokens = da_text_tokens[4*i:4*i+4] #(4, seq_len, feat_dim)
            cur_da_text_mask = da_text_mask[4*i:4*i+4] # (4, seq_len)

            text_lengths = torch.sum(cur_da_text_mask, dim=1).to(text_tokens.device)
            valid_mask = torch.where(text_lengths > 0, 1, 0)
        
            # (1, 4)
            scores = xattn_score_t2i(cur_image_tokens, cur_da_text_tokens, text_lengths, 
                                    raw_feature_norm=raw_feature_norm, agg_func=agg_func,
                                    lambda_lse=lambda_lse, lambda_softmax=lambda_softmax)
            
            # assert not torch.isnan(scores).any()

            da_scores.append(scores)
            da_valid_mask.append(valid_mask)
        
        da_scores = torch.cat(da_scores, dim=0) # (batch, 4)
        da_valid_mask = torch.stack(da_valid_mask)
    
    return (gt_scores, da_scores, da_valid_mask) if use_hard_negative_text else gt_scores


def cross_modal_token_contrastive_hinge_loss(image_tokens, text_tokens, text_mask, 
                                 raw_feature_norm="clipped_l2norm", 
                                 agg_func='LogSumExp', 
                                 lambda_lse=6., lambda_softmax=9.,
                                 margin=0.2,
                                 hardest_negative=False,
                                 use_hard_negative_text=False,
                                 valid_hard_negative_mask=None):
    
    similarity_scores = calculate_similarity_scores(image_tokens=image_tokens,
                                         text_tokens=text_tokens,
                                         text_mask=text_mask,
                                         raw_feature_norm=raw_feature_norm,
                                         agg_func=agg_func,
                                         lambda_lse=lambda_lse,
                                         lambda_softmax=lambda_softmax,
                                         use_hard_negative_text=use_hard_negative_text)
    # if use_hard_negative_text is True, similarity_scores = (batch, batch+4)
    # otherwise, square matrix (batch, batch)

    if use_hard_negative_text:
        gt_scores, da_scores, valid_negative_mask = similarity_scores
        similarity_scores = torch.cat((gt_scores, da_scores), dim=1)
        if valid_hard_negative_mask is not None:
            valid_negative_mask = valid_negative_mask * valid_hard_negative_mask
    
    diagonal = similarity_scores.diag().view(similarity_scores.size(0), 1)
    d1 = diagonal.expand_as(similarity_scores) # (num_images, num_images + 4) if using hard negative, or (num_images, num_images)
    d2 = diagonal.t().expand_as(similarity_scores[:,:similarity_scores.shape[0]]) # (num_images, num_images)

    # compare every diagonal score to scores in its column
    # caption retrieval
    cost_s = (margin + similarity_scores - d1).clamp(min=0) # (num_images, num_images + 4) if using hard negative, or (num_images, num_images)
    if use_hard_negative_text:
        cost_s[:,-4:] = cost_s[:,-4:] * valid_negative_mask
    # compare every diagonal score to scores in its row
    # image retrieval
    cost_im = (margin + similarity_scores[:,:similarity_scores.shape[0]] - d2).clamp(min=0) # (num_images, num_images)

    # clear diagonals
    # mask = torch.eye(similarity_scores.size(0)) > .5
    # mask = mask.to(device=cost_im.device)
    # cost_s = cost_s.masked_fill_(mask, 0)
    # cost_im = cost_im.masked_fill_(mask, 0)

    cost_s.fill_diagonal_(0.)
    cost_im.fill_diagonal_(0.)

    cost_s_numel = (cost_s.shape[0] - 1) * cost_s.shape[0] # ignore the diagonal
    cost_im_numel = (cost_im.shape[0] - 1) * cost_im.shape[0] # ignore the diagonal
    if use_hard_negative_text:
        cost_s_numel += valid_negative_mask.sum()

    # keep the maximum violating negative for each query
    if hardest_negative:
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]
    cost_s /= cost_s_numel # support fp16
    cost_im /= cost_im_numel # support fp16
    return cost_s.sum() + cost_im.sum()


def cross_modal_token_contrastive_sigmoid_loss(image_tokens, text_tokens, text_mask, 
                                 raw_feature_norm="clipped_l2norm", 
                                 agg_func='LogSumExp', 
                                 lambda_lse=6., lambda_softmax=9.,
                                 use_hard_negative_text=False,
                                 valid_hard_negative_mask=None):
    
    similarity_scores = calculate_similarity_scores(image_tokens=image_tokens,
                                         text_tokens=text_tokens,
                                         text_mask=text_mask,
                                         raw_feature_norm=raw_feature_norm,
                                         agg_func=agg_func,
                                         lambda_lse=lambda_lse,
                                         lambda_softmax=lambda_softmax,
                                         use_hard_negative_text=use_hard_negative_text)
    
    # if use_hard_negative_text is True, similarity_scores = (batch, batch+4)
    # otherwise, square matrix (batch, batch)
    
    if use_hard_negative_text:
        gt_scores, da_scores, valid_negative_mask = similarity_scores
        if valid_hard_negative_mask is not None:
            valid_negative_mask = valid_negative_mask * valid_hard_negative_mask

        assert gt_scores.shape[1] == gt_scores.shape[0] and da_scores.shape[1] == 4 and gt_scores.shape[0] == da_scores.shape[0]
        
        total_numel = gt_scores.shape[0]**2
        
        # clear all scores of invalid instances to 0
        da_scores = da_scores * valid_negative_mask
        total_numel += valid_negative_mask.sum()
        #assert not torch.isnan(similarity_scores).any()
        similarity_scores = torch.cat((gt_scores, da_scores), dim=1)
    else:
        #assert similarity_scores.shape[0] == similarity_scores.shape[1]
        total_numel = similarity_scores.numel()

    labels = torch.zeros_like(similarity_scores) - 1 # -1 (negative) everywhere except the diagonal
    labels.fill_diagonal_(1.) # diagonal elements are 1, i.e positive
    
    loss_mat = -F.logsigmoid(labels * similarity_scores)
    if use_hard_negative_text:
        loss_mat[:,-4:] = loss_mat[:,-4:] * valid_negative_mask
    loss_mat /= total_numel
    return loss_mat.sum()


# gather visual/text token sequences for cross-modal loss (SCAN)
def gather_token_sequences(image_tokens,
                           text_tokens,
                           text_token_mask,
                           use_hard_negative=False,
                           gather_with_grad=False,
                           rank=0,
                           world_size=1):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'

    if use_hard_negative:
        # have to concat GT and DA tensors separately
        gt_text_tokens = text_tokens[:image_tokens.shape[0]].contiguous()
        da_text_tokens = text_tokens[image_tokens.shape[0]:].contiguous()
        gt_text_token_mask = text_token_mask[:image_tokens.shape[0]].contiguous() if text_token_mask is not None else None
        da_text_token_mask = text_token_mask[image_tokens.shape[0]:].contiguous() if text_token_mask is not None else None
        
        if gather_with_grad:
            all_image_tokens = torch.cat(torch.distributed.nn.all_gather(image_tokens), dim=0)
            all_gt_text_tokens = torch.cat(torch.distributed.nn.all_gather(gt_text_tokens), dim=0)
            all_da_text_tokens = torch.cat(torch.distributed.nn.all_gather(da_text_tokens), dim=0)
            all_text_tokens = torch.cat((all_gt_text_tokens, all_da_text_tokens), dim=0)
            if text_token_mask is None:
                all_text_token_mask = None
            else:
                all_gt_text_token_mask = torch.cat(torch.distributed.nn.all_gather(gt_text_token_mask), dim=0)
                all_da_text_token_mask = torch.cat(torch.distributed.nn.all_gather(da_text_token_mask), dim=0)
                all_text_token_mask = torch.cat((all_gt_text_token_mask, all_da_text_token_mask), dim=0)
        else:    
            gathered_image_tokens = [torch.zeros_like(image_tokens) for _ in range(world_size)]
            gathered_gt_text_tokens = [torch.zeros_like(gt_text_tokens) for _ in range(world_size)]
            gathered_da_text_tokens = [torch.zeros_like(da_text_tokens) for _ in range(world_size)]
            gathered_gt_text_token_mask = [torch.zeros_like(gt_text_token_mask) for _ in range(world_size)] if text_token_mask is not None else None
            gathered_da_text_token_mask = [torch.zeros_like(da_text_token_mask) for _ in range(world_size)] if text_token_mask is not None else None

            dist.all_gather(gathered_image_tokens, image_tokens)
            dist.all_gather(gathered_gt_text_tokens, gt_text_tokens)
            dist.all_gather(gathered_da_text_tokens, da_text_tokens)
            gathered_image_tokens[rank] = image_tokens
            gathered_gt_text_tokens[rank] = gt_text_tokens
            gathered_da_text_tokens[rank] = da_text_tokens

            all_image_tokens = torch.cat(gathered_image_tokens, dim=0)
            all_text_tokens = torch.cat(gathered_gt_text_tokens+gathered_da_text_tokens, dim=0)

            if text_token_mask is not None:
                dist.all_gather(gathered_gt_text_token_mask, gt_text_token_mask)
                dist.all_gather(gathered_da_text_token_mask, da_text_token_mask)
                gathered_gt_text_token_mask[rank] = gt_text_token_mask
                gathered_da_text_token_mask[rank] = da_text_token_mask
                all_text_token_mask = torch.cat(gathered_gt_text_token_mask+gathered_da_text_token_mask, dim=0)
            else:
                all_text_token_mask = None
    else:
        if gather_with_grad:
            all_image_tokens = torch.cat(torch.distributed.nn.all_gather(image_tokens), dim=0)
            all_text_tokens = torch.cat(torch.distributed.nn.all_gather(text_tokens), dim=0)
            if text_token_mask is None:
                all_text_token_mask = None
            else:
                all_text_token_mask = torch.cat(torch.distributed.nn.all_gather(text_token_mask), dim=0)
        else:
            gathered_image_tokens = [torch.zeros_like(image_tokens) for _ in range(world_size)]
            gathered_text_tokens = [torch.zeros_like(text_tokens) for _ in range(world_size)]
            gathered_text_token_mask = [torch.zeros_like(text_token_mask) for _ in range(world_size)] if text_token_mask is not None else None

            dist.all_gather(gathered_image_tokens, image_tokens)
            dist.all_gather(gathered_text_tokens, text_tokens)
            gathered_image_tokens[rank] = image_tokens
            gathered_text_tokens[rank] = text_tokens

            all_image_tokens = torch.cat(gathered_image_tokens, dim=0)
            all_text_tokens = torch.cat(gathered_text_tokens, dim=0)

            if gathered_text_token_mask is not None:
                dist.all_gather(gathered_text_token_mask, text_token_mask)
                gathered_text_token_mask[rank] = text_token_mask
                all_text_token_mask = torch.cat(gathered_text_token_mask, dim=0)
            else:
                all_text_token_mask = None
        
    return all_image_tokens, all_text_tokens, all_text_token_mask


def compute_flair_loss(image_tokens, # (batch, seq_len, feature_dim)
                       text_features, # (batch, feature_dim)
                       raw_feature_norm="no_norm",
                       logits_scale=None,
                       logits_bias=None,
                       lambda_softmax=1.,
                       loss_feature_norm=False):
    N = image_tokens.shape[0]
    assert N == text_features.shape[0]
    # expanded_text_features[i,j,:]: feature of caption[j], repeated to correspond to image[i]
    expanded_text_features = text_features.unsqueeze(1) # (batch, 1, feature_dim)

    logits = []

    for i in range(N): # for each text caption
        # (batch, 1, feat_dim)
        query = expanded_text_features[i:i+1].expand(N, -1, -1)
        # (batch, 1, feat_dim)
        weighted_features_, _ = func_attention(query=query, context=image_tokens, raw_feature_norm=raw_feature_norm, smooth=lambda_softmax)
        # weighted_features_[i,:,:] -> pooled features of an image
        weighted_features_ = F.normalize(weighted_features_, dim=-1, eps=1e-8)
        weighted_features_ = weighted_features_.squeeze(1) # (batch, feat_dim)
        if loss_feature_norm:
            query_features = F.normalize(text_features[i:i+1], dim=-1) #(1, feat_dim)
        else:
            query_features = text_features[i:i+1] #(1, feat_dim)
        query_i_logits = (weighted_features_ @ query_features.T) #(batch, 1) similarity of all images -> 1 caption
        logits.append(query_i_logits)
    
    logits = torch.cat(logits, dim=1)

    if logits_scale is not None:
        logits *= logits_scale
    if logits_bias is not None:
        logits += logits_bias
    
    labels = torch.zeros_like(logits) - 1 # -1 (negative) everywhere except the diagonal
    labels.fill_diagonal_(1.) # diagonal elements are 1, i.e positive
    
    output_loss = -F.logsigmoid(labels * logits).sum() / logits.shape[1]
    return output_loss
    

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
######################################################################

# intramodal np loss #################################################
def per_sample_intramodal_loss(slice: torch.Tensor, logit_scale, logit_bias=None):
    try:
        if slice.shape[0] == 1:
            return 0
    except:
        return 0
    
    scores = torch.matmul(slice, slice.t()) * logit_scale # (num_phrase, num_phrases)
    if logit_bias is not None:
        scores += logit_bias
    # get sigmoid
    scores = - F.logsigmoid(-scores) # only consider negatives
    # get upper triangle (above main diagonal) only
    scores = torch.triu(scores, diagonal=1)
    # calculate score
    num_item = scores.shape[0]*(scores.shape[0]-1) / 2 # number of valid items
    final_score = scores.sum()
    return {"total": final_score, "count": num_item}


def intramodal_contrastive_loss(
        features: torch.Tensor,   # (num_phrases, feature_dim)
        indices: torch.Tensor,    # (num_phrases)
        logit_scale,
        logit_bias=None,
        num_samples=None,
) -> torch.Tensor: # torch.Tensor, shape = (1,)
    # 1. group features per sample
    if num_samples is None:
        # count samples
        max_index = -1
        for item in indices:
            if item > max_index:
                max_index = item
        num_samples = max_index + 1
    samples_slices = []
    indices_range = torch.arange(0, indices.shape[0], dtype=torch.int64, device=indices.device)
    for i in range(num_samples):
        mask = indices == i
        slice_indices = indices_range.masked_select(mask)
        slice = features[slice_indices]
        samples_slices.append(slice)

    # 2. Calculate scores per sample & gather
    total = 0
    cnt = 0
    for slice in samples_slices:
        if slice.shape[0] == 1:
            continue
        ret = per_sample_intramodal_loss(slice, logit_scale, logit_bias)
        total += ret["total"]
        cnt += ret["count"]
    
    # 3. return final loss
    final_score = total / cnt if cnt > 0 else 0
    return final_score


# nounphrases loss ###################################################
def nounphrase_loss(image_features, # (batch, feature_dim)
                    image_tokens, # (batch, seq_len, feature_dim)
                    nounphrases_features, # (num_phrases, feature_dim)
                    nounphrases_indices, # (num_phrases)
                    nounphrases_token_features, # (num_nounphrases, seq_len, feature_dim)
                    nounphrases_token_mask, # (num_nounphrases, seq_len)
                    hn_nounphrases_features, # (num_hard_negative_phrases, feature_dim)
                    hn_nounphrases_indices, # (num_hard_negative_phrases)
                    logit_scale, 
                    logit_bias=None,
                    np_instance_loss=True, 
                    np_instance_loss_scale=1.,
                    np_token_loss=True,
                    np_token_loss_scale=0.1,
                    np_token_token_loss=False,
                    np_token_token_loss_scale=1.,
                    # for np_token_loss ######
                    raw_feature_norm="clipped_l2norm", 
                    agg_func='LogSumExp', #'Mean', #'LogSumExp', # use mean in this case because we need to include all nounphrases in the final score estimate
                    token_loss_func="sigmoid", # "hinge" or "sigmoid",
                    hinge_margin=0.2, 
                    lambda_lse=6., 
                    lambda_softmax=9.,
                    ##########################
                    np_intramodal_loss=True,
                    np_intramodal_loss_scale=0.005,
                    ##########################
                    np_flair_loss=True,
                    np_flair_loss_scale=1.,
                    ##########################
                    np_hard_negative_loss=True,
                    np_hard_negative_loss_scale=1.,
                    ##########################
                    np_hard_negative_flair_loss=True,
                    np_hard_negative_flair_loss_scale=1.,
                    ##########################
                    loss_feature_norm=False,
                    ##########################
                    rank=0,
                    world_size=1):
    if image_features is None and np_instance_loss:
        raise ValueError("image features must be provided and np_instance_loss is enabled")
    if image_tokens is None and np_token_loss:
        raise ValueError("image tokens features must be provided and np_token_loss is enabled")
    
    if loss_feature_norm:
        image_features = F.normalize(image_features, dim=-1)
    
    # gather all data if using distributed
    if world_size > 1:
        image_features = all_gather_tensor(image_features, rank=rank, world_size=world_size)
        if image_tokens is not None:
            image_tokens = all_gather_tensor(image_tokens, rank=rank, world_size=world_size)
        nounphrases_features = all_gather_variable_size_tensor(nounphrases_features, rank=rank, world_size=world_size)
        nounphrases_indices = all_gather_variable_size_tensor(nounphrases_indices, rank=rank, world_size=world_size, gather_indices=True)
        if nounphrases_token_features is not None:
            nounphrases_token_features = all_gather_variable_size_tensor(nounphrases_token_features, rank=rank, world_size=world_size)
            nounphrases_token_mask = all_gather_variable_size_tensor(nounphrases_token_mask, rank=rank, world_size=world_size)
        if hn_nounphrases_features is not None:
            hn_nounphrases_features = all_gather_variable_size_tensor(hn_nounphrases_features, rank=rank, world_size=world_size)
            hn_nounphrases_indices = all_gather_variable_size_tensor(hn_nounphrases_indices, rank=rank, world_size=world_size, gather_indices=True)
    
    instance_loss, token_loss, intramodal_loss, token_token_loss = 0., 0., 0., 0.

    outputs = {}
    
    if np_instance_loss:
        if loss_feature_norm:
            normalized_nounphrases_features = F.normalize(nounphrases_features, dim=-1)
            logits_per_image = torch.matmul(image_features, normalized_nounphrases_features.t()) * logit_scale
        else:
            logits_per_image = torch.matmul(image_features, nounphrases_features.t()) * logit_scale
        if logit_bias is not None:
            logits_per_image += logit_bias
        labels = - torch.ones_like(logits_per_image)
        for col, idx in enumerate(nounphrases_indices):
            labels[idx][col] = 1 # set positive if nounphrase appears in image
        # instance_loss = -F.logsigmoid(labels * logits_per_image).sum() / logits_per_image.shape[0]
        instance_loss = -F.logsigmoid(labels * logits_per_image).sum() / logits_per_image.shape[1]
        # instance_loss = -F.logsigmoid(labels * logits_per_image).sum() / torch.numel(logits_per_image)
        outputs["np_instance_loss"] = instance_loss * np_instance_loss_scale

    
    if np_flair_loss:
        M = image_tokens.shape[0] # num of images
        N = nounphrases_features.shape[0] # num of nounphrases
        expanded_np_features = nounphrases_features.unsqueeze(1) # (N, 1, feat_dim)
        
        logits = []
        
        for i in range(N):
            # (M, 1, feat_dim)
            query = expanded_np_features[i:i+1].expand(M, -1, -1)
            # (M, 1, feat_dim)
            weighted_features_, _ = func_attention(query=query, context=image_tokens, raw_feature_norm="no_norm", smooth=1.)#raw_feature_norm="raw_feature_norm", smooth=lambda_softmax)
            # weighted_features_[i,:,:] -> pooled features of an image
            weighted_features_ = F.normalize(weighted_features_, dim=-1, eps=1e-8)
            weighted_features_ = weighted_features_.squeeze(1) # (batch, feat_dim)
            if loss_feature_norm:
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
        flair_loss = -F.logsigmoid(labels * logits).sum() / logits.shape[1]
        outputs["np_flair_loss"] = flair_loss * np_flair_loss_scale


    if np_token_loss:
        # reshape nounphrases_features (num_phrases, feature_dim) -> (batch, num_phrases_per_sample, feature_dim)
        N = image_tokens.shape[0] # num of samples
        max_length = 0
        for i in range(N):
            matches = torch.where(nounphrases_indices == i, 1, 0)
            cnt = matches.sum()
            max_length = max(max_length, cnt.item())
        """
        np_tensor = torch.zeros((N, max_length, image_tokens.shape[-1]), dtype=nounphrases_features.dtype, device=nounphrases_features.device)
        np_cnt = torch.zeros(N, dtype=torch.int64, device=nounphrases_features.device)
        # start copying
        start_row = 0
        for i in range(N):
            matches = torch.where(nounphrases_indices == i, 1, 0)
            cnt = matches.sum()
            np_cnt[i] = cnt
            end_row = start_row + cnt.item()
            # np_tensor[i,0:cnt,:] = nounphrases_features[start_row:end_row,:]
            np_tensor[i,0:cnt] += nounphrases_features[start_row:end_row]
            start_row = end_row
        """
        np_tensor = []
        np_cnt = torch.zeros(N, dtype=torch.int64, device=nounphrases_features.device)
        start_row = 0
        for i in range(N):
            matches = torch.where(nounphrases_indices == i, 1, 0)
            cnt = matches.sum()
            np_cnt[i] = cnt
            end_row = start_row + cnt.item()
            slice = nounphrases_features[start_row:end_row]
            # pad = torch.zeros((max_length-cnt, nounphrases_features.shape[-1]), device=nounphrases_features.device, dtype=nounphrases_features.dtype)
            # slice = torch.cat([slice, pad], dim=0)
            padded_slice = F.pad(slice, pad=(0, 0, 0, max_length-cnt), mode="constant", value=0)
            np_tensor.append(padded_slice)
            start_row = end_row
        np_tensor = torch.stack(np_tensor)

        # compute image-phrases score matrix
        scores = xattn_score_t2i(image_tokens, np_tensor, np_cnt, 
                                raw_feature_norm=raw_feature_norm, agg_func=agg_func,
                                lambda_lse=lambda_lse, lambda_softmax=lambda_softmax)
        assert scores.shape[0] == scores.shape[1]

        if token_loss_func == "sigmoid":
            labels = torch.zeros_like(scores) - 1 # -1 (negative) everywhere except the diagonal
            labels.fill_diagonal_(1.) # diagonal elements are 1, i.e positive
            # FOR DEBUGGING
            # token_loss = - F.logsigmoid(labels * scores)
            # token_loss = token_loss.sum()
            # token_loss = token_loss / torch.numel(scores)
            # token_loss = token_loss * np_token_loss_scale
            # END FOR DEBUGGING
            token_loss = torch.sum(-F.logsigmoid(labels * scores) / torch.numel(scores))
        elif token_loss_func == "hinge":
            diagonal = scores.diag().view(scores.size(0), 1)
            d1 = diagonal.expand_as(scores)
            d2 = diagonal.t().expand_as(scores)

            # compare every diagonal score to scores in its column
            # caption retrieval
            cost_s = (hinge_margin + scores - d1).clamp(min=0)
            # compare every diagonal score to scores in its row
            # image retrieval
            cost_im = (hinge_margin + scores - d2).clamp(min=0)

            # clear diagonals
            # mask = torch.eye(scores.size(0)) > .5
            # mask = mask.to(device=cost_im.device)
            
            # cost_s = cost_s.masked_fill_(mask, 0)
            # cost_im = cost_im.masked_fill_(mask, 0)
            cost_s.fill_diagonal_(0.)
            cost_im.fill_diagonal_(0.)

            numel = cost_im.numel() - cost_im.shape[0]

            # keep the maximum violating negative for each query
            token_loss = torch.sum(cost_s / numel) + torch.sum(cost_im / numel)
        else:
            raise RuntimeError(f"unknown token loss type: {token_loss_func}")
        
        outputs["np_token_loss"] = token_loss * np_token_loss_scale
        
    if np_intramodal_loss:
        if loss_feature_norm:
            normalized_nounphrases_features = F.normalize(nounphrases_features, dim=-1)
        else:
            normalized_nounphrases_features = nounphrases_features
        intramodal_loss = intramodal_contrastive_loss(features=normalized_nounphrases_features,
                                                      indices=nounphrases_indices,
                                                      logit_scale=logit_scale,
                                                      logit_bias=logit_bias,
                                                      num_samples=image_features.shape[0])
        intramodal_loss = intramodal_loss * np_intramodal_loss_scale
        outputs["np_intramodal_loss"] = intramodal_loss

    ## np_token_token_loss
    if np_token_token_loss:
        if nounphrases_token_mask[0][0] == 0:
            nounphrases_token_mask = nounphrases_token_mask[:,1:]
        
        text_lengths = torch.sum(nounphrases_token_mask, dim=1)

        # (num_images, num_phrases)
        similarity_scores = xattn_score_t2i(image_tokens, nounphrases_token_features, text_lengths,
                                raw_feature_norm=raw_feature_norm, agg_func=agg_func,
                                lambda_lse=lambda_lse, lambda_softmax=lambda_softmax)

        if token_loss_func == "sigmoid":
            labels = torch.zeros_like(similarity_scores) - 1 # (num_images, num_phrases)
            for col, idx in enumerate(nounphrases_indices):
                labels[idx][col] = 1 # set positive if nounphrase appears in image
            # instance_loss = -F.logsigmoid(labels * logits_per_image).sum() / logits_per_image.shape[0]
            token_token_loss = torch.sum(-F.logsigmoid(labels * similarity_scores) / torch.numel(similarity_scores))
        elif token_loss_func == "hinge":
            # expand (by duplicating rows) the scores matrix -> square
            score_mask = torch.zeros_like(similarity_scores, dtype=torch.bool)
            for col, idx in enumerate(nounphrases_indices):
                score_mask[idx][col] = True
            expanded_scores = torch.zeros((similarity_scores.shape[1], similarity_scores.shape[1]), device=image_features.device, dtype=image_features.dtype)
            expanded_mask = torch.zeros((similarity_scores.shape[1], similarity_scores.shape[1]), device=image_features.device, dtype=torch.bool)
            for col, idx in enumerate(nounphrases_indices):
                expanded_scores[col] += similarity_scores[idx]
                expanded_mask[col] += score_mask[idx]

            # expanded_scores & expanded_mask are both (num_phrases, num_phrases)

            diagonal = expanded_scores.diag().view(expanded_scores.size(0), 1)
            d1 = diagonal.expand_as(expanded_scores)
            d2 = diagonal.t().expand_as(expanded_scores)

            # compare every diagonal score to scores in its column
            # caption retrieval
            cost_s = (hinge_margin + expanded_scores - d1).clamp(min=0)
            # compare every diagonal score to scores in its row
            # image retrieval
            cost_im = (hinge_margin + expanded_scores - d2).clamp(min=0)

            # clear diagonals
            cost_s.masked_fill_(expanded_mask, 0.)
            cost_im.masked_fill_(expanded_mask, 0.)

            numel = cost_im.numel() - expanded_mask.sum()

            # keep the maximum violating negative for each query
            token_token_loss = torch.sum(cost_s / numel) + torch.sum(cost_im / numel)
        else:
            raise RuntimeError(f"unknown token loss type: {token_loss_func}")
        
        outputs["np_token_token_loss"] = token_token_loss * np_token_token_loss_scale

    if np_hard_negative_loss:
        N = image_features.shape[0] # num of samples
        total_hn_loss = 0
        valid_cnt = 0
        if loss_feature_norm:
            normalized_hn_nounphrases_features = F.normalize(hn_nounphrases_features, dim=-1)
        else:
            normalized_hn_nounphrases_features = hn_nounphrases_features
        for i in range(N):
            matched_mask = hn_nounphrases_indices == i
            indices = matched_mask.nonzero()
            if len(indices) == 0:
                continue # no hard negative for sample
            indices = indices.expand(-1, normalized_hn_nounphrases_features.shape[-1])
            sample_hn_features_i = normalized_hn_nounphrases_features.gather(dim=0, index=indices)
            logits_i = image_features[i:i+1] @ sample_hn_features_i.T * logit_scale # (1, num_hn_features)
            if logit_bias is not None:
                logits_i += logit_bias
            
            hn_loss = - F.logsigmoid(- logits_i).mean()
            
            total_hn_loss += hn_loss
            valid_cnt += 1
        total_hn_loss *= (N / valid_cnt) # scale to same range as regular siglip loss
        outputs["np_hard_negative_loss"] = total_hn_loss * np_hard_negative_loss_scale

    if np_hard_negative_flair_loss:
        N = image_features.shape[0] # num of samples
        total_hn_flair_loss = 0
        valid_cnt = 0
        for i in range(N):
            matched_mask = hn_nounphrases_indices == i
            indices = matched_mask.nonzero()
            if len(indices) == 0:
                continue # no hard negative for sample
            indices = indices.expand(-1, hn_nounphrases_features.shape[-1])
            sample_hn_features_i = hn_nounphrases_features.gather(dim=0, index=indices) # (num_hn_features, feat_dim)
            
            expanded_np_features = sample_hn_features_i.unsqueeze(1) # (K, 1, feat_dim)
            if loss_feature_norm:
                normalized_sample_hn_features_i = F.normalize(sample_hn_features_i, dim=-1)
            else:
                normalized_sample_hn_features_i = sample_hn_features_i

            K = sample_hn_features_i.shape[0]
        
            logits_i = []
            
            for j in range(K):
                # (1, 1, feat_dim)
                query = expanded_np_features[j:j+1]
                # (1, 1, feat_dim)
                weighted_features_, _ = func_attention(query=query, context=image_tokens[i:i+1], raw_feature_norm="no_norm", smooth=1.)#raw_feature_norm="raw_feature_norm", smooth=lambda_softmax)
                # weighted_features_[i,:,:] -> pooled features of an image
                weighted_features_ = F.normalize(weighted_features_, dim=-1, eps=1e-8)
                weighted_features_ = weighted_features_.squeeze() # (feat_dim)
                query_i_logits = (normalized_sample_hn_features_i[j] * weighted_features_).sum() # (1) single score
                logits_i.append(query_i_logits)

            logits_i = torch.stack(logits_i) # (K, 1)
            
            if logit_bias is not None:
                logits_i += logit_bias
            
            hn_flair_loss = - F.logsigmoid(- logits_i).mean()
            
            total_hn_flair_loss += hn_flair_loss
            valid_cnt += 1
        total_hn_flair_loss *= (N / valid_cnt) # scale to same range as regular siglip loss
        outputs["np_hard_negative_flair_loss"] = total_hn_flair_loss * np_hard_negative_flair_loss_scale
            
    
    ## take the total & return
    return outputs
        
    
####################################################################################################

# new CE_CLIP loss
class Clip_DA_Loss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            cmr_loss=False,
            imc_loss=False,
            hardnegative=False,
            imc_loss_weight=0.2,
            cmr_loss_weight=0.2,
            threshold_type='mean',
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
        self.cmr_loss=cmr_loss
        self.imc_loss=imc_loss
        self.imc_loss_weight=imc_loss_weight
        self.cmr_loss_weight=cmr_loss_weight
        self.threshold_type=threshold_type
        self.hardnegative=hardnegative
        
    def forward(self, image_features, text_features, valid_caption_mask, logit_scale, thresholds, output_dict=False):
        """
        cross-modal ranking loss doesn't support local_loss and use_horovod 

        Different Losses:
            - hard negative: standard clip contrastive loss, assuming hard-negatives as extra negative for computing logits_per_image, logits_per_text is the same as clip
            - imc_loss: standard clip contrastive loss + contrastive loss on text embeddings (between ground truth caption embedding and hard-negative caption embedding)
            - cmr_loss: standard clip contrastive loss + rank loss between gt pair and hg pair
        """
        device = image_features.device
        cmr_loss, imc_loss = 0.0, 0.0

        if self.world_size > 1:
            all_image_features, all_text_features, all_valid_caption_mask = gather_features_da(
                image_features, text_features, valid_caption_mask,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            
            caption_types = torch.tensor(([1]*image_features.shape[0]+[2]*image_features.shape[0]*4)*self.world_size)
            gt_all_text_features = all_text_features[caption_types==1] # batch_size * word_size
            da_all_text_features = all_text_features[caption_types==2] # 4 * batch_size * word_size
            gt_len,feature_size = all_image_features.shape[0],all_image_features.shape[-1]


            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                #extra hard negative loss
                if self.hardnegative:
                    all_text_features = torch.cat([gt_all_text_features, da_all_text_features])
                    logits_per_image = logit_scale * all_image_features @ all_text_features.T # batch_size * 5xbatch_size       
                else:
                    logits_per_image = logit_scale * all_image_features @ gt_all_text_features.T

                logits_per_text = logit_scale * gt_all_text_features @ all_image_features.T

                # cross-modal rank loss
                if self.cmr_loss:
                    da_logits_per_image = logit_scale * (da_all_text_features.reshape(gt_len,-1,feature_size)@ all_image_features.unsqueeze(-1)).squeeze() * all_valid_caption_mask
                    cmr_loss, thresholds = get_cmr_loss(logits_per_image, da_logits_per_image, all_valid_caption_mask, thresholds, self.threshold_type)
                
                # intra-modal contrastive loss
                if self.imc_loss:
                    text_embedding_matrix = logit_scale * gt_all_text_features @ da_all_text_features.T  #(all_batch_size,4*all_batch_size)
                    imc_loss = get_imc_loss(text_embedding_matrix)

                # TODO: Implement SCAN loss for distributed training

        else:
            gt_len, feature_size = image_features.shape[0],image_features.shape[-1]
            gt_text_features = text_features[:image_features.shape[0]]
            da_text_features = text_features[image_features.shape[0]:]
            all_text_features = torch.cat([gt_text_features,da_text_features])
            
            if self.hardnegative:
                logits_per_image = logit_scale * image_features @ all_text_features.T
            else:
                logits_per_image = logit_scale * image_features @ gt_text_features.T
            logits_per_text = logit_scale * gt_text_features @ image_features.T
        
            if self.cmr_loss:
                da_logits_per_image = logit_scale * (da_text_features.reshape(gt_len,-1,feature_size)@ image_features.unsqueeze(-1)).squeeze() * valid_caption_mask
                cmr_loss, thresholds = get_cmr_loss(logits_per_image, da_logits_per_image, valid_caption_mask, thresholds, self.threshold_type)
            
            if self.imc_loss:
                text_embedding_matrix = logit_scale * gt_text_features @ da_text_features.T #(batch_size,4*batch_size)
                imc_loss = get_imc_loss(text_embedding_matrix)         
        
        num_logits = logits_per_image.shape[0]
        
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        
        loss_outputs = {"contrastive_loss": total_loss}
        
        if self.cmr_loss:
            #total_loss += cmr_loss*self.cmr_loss_weight
            loss_outputs["cmr_loss"] = cmr_loss*self.cmr_loss_weight
        
        if self.imc_loss:
            #total_loss += imc_loss*self.imc_loss_weight
            loss_outputs["imc_loss"] = imc_loss*self.imc_loss_weight
            
        #return total_loss, thresholds, cmr_loss, imc_loss
        if output_dict:
            loss_outputs["thresholds"] = thresholds
            return loss_outputs
        
        return total_loss, loss_outputs["cmr_loss"] if self.cmr_loss else None, loss_outputs["imc_loss"] if self.imc_loss else None, thresholds


class SigLip_DA_Loss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            #####################
            # CE-CLIP loss
            cmr_loss=False,
            imc_loss=False,
            hardnegative=False,
            imc_loss_weight=0.2,
            cmr_loss_weight=0.2,
            threshold_type='mean',
            #####################
            # SCAN loss
            scan_loss=False,
            scan_loss_weight=0.5,
            scan_feature_norm="clipped_l2norm",
            scan_agg_func="LogSumExp",
            scan_lambda_lse=6.,
            scan_lambda_softmax=9.,
            scan_loss_type="sigmoid",
            scan_hinge_margin=0.2,
            scan_hard_negative=False,
            scan_ce_hard_negative=False,
            #####################
            # Nounphrase loss
            np_loss=False,
            np_instance_loss=True,
            np_token_loss=True,
            np_intramodal_loss=True,
            np_loss_weight=0.5,
            np_intramodal_loss_scale=0.005,
            np_token_loss_scale=0.1,
            np_token_token_loss=False,
            np_token_token_loss_scale=1.,
            #####################
            # FLAIR loss
            flair_loss=False,
            flair_loss_scale=1.,
            ##########################
            np_flair_loss=False,
            np_flair_loss_scale=1.,
            ##########################
            np_hard_negative_loss=False,
            np_hard_negative_loss_scale=1.,
            ##########################
            np_hard_negative_flair_loss=False,
            np_hard_negative_flair_loss_scale=1.,
            ##########################,
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

        # CLIP-DA losses
        self.cmr_loss=cmr_loss
        self.imc_loss=imc_loss
        self.imc_loss_weight=imc_loss_weight
        self.cmr_loss_weight=cmr_loss_weight
        self.threshold_type=threshold_type
        self.hardnegative=hardnegative

        # SCAN loss
        self.scan_loss = scan_loss
        self.scan_loss_weight = scan_loss_weight
        self.scan_feature_norm = scan_feature_norm
        self.scan_agg_func = scan_agg_func
        self.scan_lambda_lse = scan_lambda_lse
        self.scan_lambda_softmax = scan_lambda_softmax
        self.scan_loss_type = scan_loss_type
        self.scan_hinge_margin = scan_hinge_margin
        self.scan_hard_negative = scan_hard_negative
        self.scan_ce_hard_negative = scan_ce_hard_negative

        # Nounphrase loss
        self.np_loss = np_loss
        self.np_instance_loss = np_instance_loss
        self.np_token_loss = np_token_loss
        self.np_intramodal_loss = np_intramodal_loss
        self.np_loss_weight = np_loss_weight
        self.np_intramodal_loss_scale = np_intramodal_loss_scale
        self.np_token_loss_scale = np_token_loss_scale
        
        self.np_token_token_loss = np_token_token_loss
        self.np_token_token_loss_scale = np_token_token_loss_scale
        
        self.np_flair_loss = np_flair_loss
        self.np_flair_loss_scale = np_flair_loss_scale
        
        self.np_hard_negative_loss = np_hard_negative_loss
        self.np_hard_negative_loss_scale = np_hard_negative_loss_scale
        
        self.np_hard_negative_flair_loss = np_hard_negative_flair_loss
        self.np_hard_negative_flair_loss_scale = np_hard_negative_flair_loss_scale

        # FLAIR
        self.flair_loss = flair_loss
        self.flair_loss_scale = flair_loss_scale

        self.loss_feature_norm = loss_feature_norm

        
    def forward(self, 
                image_features, 
                text_features, 
                logit_scale, 
                logit_bias, 
                output_dict=False, 
                # CE arguments
                valid_caption_mask=None,
                thresholds=None, 
                # end CE arguments
                # SCAN inputs
                image_tokens=None, 
                text_tokens=None, 
                text_token_mask=None,
                # end SCAN
                # NP inputs
                nounphrases_features=None,
                nounphrases_indices=None,
                nounphrases_token_features=None,
                nounphrases_token_mask=None,
                hn_nounphrases_features=None,
                hn_nounphrases_indices=None,):
        """
        cross-modal ranking loss doesn't support local_loss and use_horovod 

        Different Losses:
            - hard negative: standard clip contrastive loss, assuming hard-negatives as extra negative for computing logits_per_image, logits_per_text is the same as clip
            - imc_loss: standard clip contrastive loss + contrastive loss on text embeddings (between ground truth caption embedding and hard-negative caption embedding)
            - cmr_loss: standard clip contrastive loss + rank loss between gt pair and hg pair
        """
        device = image_features.device
        cmr_loss, imc_loss, scan_loss, np_loss = 0.0, 0.0, 0.0, 0.0
        flair_loss = 0.

        USE_DA = True if image_features.shape[0] != text_features.shape[0] else False

        if self.world_size > 1:
            all_image_features, all_text_features, all_valid_caption_mask = gather_features_da(
                image_features, text_features, valid_caption_mask,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            
            if USE_DA:
                # use hard-negative captions
                caption_types = torch.tensor(([1]*image_features.shape[0]+[2]*image_features.shape[0]*4)*self.world_size)
                gt_all_text_features = all_text_features[caption_types==1] # batch_size * word_size
                da_all_text_features = all_text_features[caption_types==2] # 4 * batch_size * word_size
                gt_len, feature_size = all_image_features.shape[0], all_image_features.shape[-1]
            else:
                gt_all_text_features = all_text_features
                da_all_text_features = None
            
            gt_len, feature_size = all_image_features.shape[0], all_image_features.shape[-1]

            if self.local_loss:
                # siglip 
                logits_per_image = torch.matmul(image_features, all_text_features.t()) * logit_scale +logit_bias
                #logits_per_text = torch.matmul(text_features, all_image_features.t()) * logit_scale +logit_bias
            else:
                # siglip + hardnegative
                if self.hardnegative and USE_DA:
                    all_text_features = torch.cat([gt_all_text_features, da_all_text_features])
                    if self.loss_feature_norm:
                        normalized_gt_all_text_features = F.normalize(gt_all_text_features, dim=-1)
                        normalized_all_text_features = F.normalize(all_text_features, dim=-1)
                        normalized_all_image_features = F.normalize(all_image_features, dim=-1)
                        logits_per_image = torch.matmul(normalized_all_image_features, normalized_all_text_features.t()) * logit_scale +logit_bias
                    else:
                        normalized_gt_all_text_features = gt_all_text_features
                        normalized_all_image_features = all_image_features
                        logits_per_image = torch.matmul(all_image_features, all_text_features.t()) * logit_scale +logit_bias
                else:
                    if self.loss_feature_norm:
                        normalized_gt_all_text_features = F.normalize(gt_all_text_features, dim=-1)
                        normalized_all_image_features = F.normalize(all_image_features, dim=-1)
                        logits_per_image = torch.matmul(normalized_all_image_features, normalized_gt_all_text_features.t()) * logit_scale +logit_bias
                    else:
                        normalized_gt_all_text_features = gt_all_text_features
                        normalized_all_image_features = all_image_features
                        logits_per_image = torch.matmul(all_image_features, gt_all_text_features.t()) * logit_scale +logit_bias

                #logits_per_text = torch.matmul(gt_all_text_features, all_image_features.t()) * logit_scale + logit_bias

                if USE_DA:
                    if self.loss_feature_norm:
                        normalized_da_all_text_features = F.normalize(da_all_text_features, dim=-1)
                    else:
                        normalized_da_all_text_features = da_all_text_features

                # cross-modal rank loss
                if self.cmr_loss:
                    da_logits_per_image = logit_scale * (normalized_da_all_text_features.reshape(gt_len,-1,feature_size)@ normalized_all_image_features.unsqueeze(-1)).squeeze()
                    if logit_bias is not None:
                        da_logits_per_image += logit_bias
                    da_logits_per_image = da_logits_per_image * all_valid_caption_mask
                    cmr_loss, thresholds = get_cmr_loss(logits_per_image, da_logits_per_image, all_valid_caption_mask, thresholds, self.threshold_type)
                
                # intra-modal contrastive loss
                if self.imc_loss:
                    text_embedding_matrix = logit_scale * normalized_gt_all_text_features @ normalized_da_all_text_features.T  #(all_batch_size,4*all_batch_size)
                    if logit_bias is not None:
                        text_embedding_matrix += logit_bias
                    imc_loss = get_imc_loss(text_embedding_matrix)

            # ~~TODO: Implement SCAN loss for distributed training~~
            if self.scan_loss:
                assert text_token_mask is not None, "To use SCAN loss, the text_token_mask must be specified"
                # NOTE: hard-coded truncation of last token for SigLIP text encoder
                text_token_mask = text_token_mask[:,:-1].contiguous()
                if USE_DA and not self.hardnegative:
                    # keep only GT text tokens before distributed gathering
                    text_tokens = text_tokens[:image_tokens.shape[0]].contiguous()
                    text_token_mask = text_token_mask[:image_tokens.shape[0]].contiguous()
                ########################
                all_image_tokens, all_text_tokens, all_text_token_mask = gather_token_sequences(image_tokens=image_tokens,
                                                                                                text_tokens=text_tokens,
                                                                                                text_token_mask=text_token_mask,
                                                                                                use_hard_negative=USE_DA and self.hardnegative,
                                                                                                gather_with_grad=self.gather_with_grad,
                                                                                                rank=self.rank,
                                                                                                world_size=self.world_size)
            elif self.flair_loss:
                all_image_tokens = all_gather_tensor(data_tensor=image_tokens, 
                                                     rank=self.rank,
                                                     world_size=self.world_size)

            if self.scan_loss:
                if self.scan_loss_type == "hinge":
                    scan_loss = cross_modal_token_contrastive_hinge_loss(image_tokens=all_image_tokens,
                                                                   text_tokens=all_text_tokens,
                                                                   text_mask=all_text_token_mask,
                                                                   raw_feature_norm=self.scan_feature_norm,
                                                                   agg_func=self.scan_agg_func,
                                                                   lambda_lse=self.scan_lambda_lse,
                                                                   lambda_softmax=self.scan_lambda_softmax,
                                                                   margin=self.scan_hinge_margin,
                                                                   hardest_negative=self.scan_hard_negative,
                                                                   use_hard_negative_text=self.scan_ce_hard_negative,
                                                                   valid_hard_negative_mask=all_valid_caption_mask)
                elif self.scan_loss_type == "sigmoid":
                    scan_loss = cross_modal_token_contrastive_sigmoid_loss(image_tokens=all_image_tokens,
                                                                   text_tokens=all_text_tokens,
                                                                   text_mask=all_text_token_mask,
                                                                   raw_feature_norm=self.scan_feature_norm,
                                                                   agg_func=self.scan_agg_func,
                                                                   lambda_lse=self.scan_lambda_lse,
                                                                   lambda_softmax=self.scan_lambda_softmax,
                                                                   use_hard_negative_text=self.scan_ce_hard_negative,
                                                                   valid_hard_negative_mask=all_valid_caption_mask)
                    
            if self.flair_loss:
                flair_loss = compute_flair_loss(image_tokens=all_image_tokens,
                                                text_features=gt_all_text_features,
                                                logits_scale=logit_scale,
                                                logits_bias=logit_bias,
                                                # raw_feature_norm=self.scan_feature_norm,
                                                # lambda_softmax=self.scan_lambda_softmax
                                                loss_feature_norm=self.loss_feature_norm
                                                )
                
        else:
            # not updating very long time
            gt_len, feature_size = image_features.shape[0],image_features.shape[-1]
            if USE_DA:
                gt_text_features = text_features[:image_features.shape[0]]
                da_text_features = text_features[image_features.shape[0]:]
                all_text_features = torch.cat([gt_text_features,da_text_features])
            else:
                gt_text_features = text_features

            if self.loss_feature_norm:
                normalized_image_features = F.normalize(image_features, dim=-1)
                normalized_gt_text_features = F.normalize(gt_text_features, dim=-1)
                if USE_DA:
                    normalized_da_text_features = F.normalize(da_text_features, dim=-1)
                    normalized_all_text_features = torch.cat([normalized_gt_text_features,normalized_da_text_features])
            else:
                normalized_image_features = image_features
                normalized_gt_text_features = gt_text_features
                if USE_DA:
                    normalized_da_text_features = da_text_features
                    normalized_all_text_features = all_text_features
            
            
            if self.hardnegative and USE_DA:
                logits_per_image = torch.matmul(normalized_image_features, normalized_all_text_features.t()) * logit_scale + logit_bias
            else:
                logits_per_image = torch.matmul(normalized_image_features, normalized_gt_text_features.t()) * logit_scale + logit_bias
            #logits_per_text = torch.matmul(gt_text_features, image_features.t()) * logit_scale + logit_bias
            
            if self.cmr_loss:
                da_logits_per_image = logit_scale * (normalized_da_text_features.reshape(gt_len,-1,feature_size)@ normalized_da_text_features.unsqueeze(-1)).squeeze()
                if logit_bias is not None:
                    da_logits_per_image += logit_bias
                da_logits_per_image = da_logits_per_image * valid_caption_mask
                cmr_loss, thresholds = get_cmr_loss(logits_per_image, da_logits_per_image, valid_caption_mask, thresholds, self.threshold_type)
            
            if self.imc_loss:
                text_embedding_matrix = logit_scale * normalized_gt_text_features @ normalized_da_text_features.T #(batch_size,4*batch_size)
                if logit_bias is not None:
                    text_embedding_matrix += logit_bias
                imc_loss = get_imc_loss(text_embedding_matrix)

            if self.scan_loss:
                assert text_token_mask is not None, "To use SCAN loss, the text_token_mask must be specified"
                # NOTE: hard-coded truncation of last token for SigLIP text encoder
                text_token_mask = text_token_mask[:,:-1].contiguous()
                ########################
                if self.scan_loss_type == "hinge":
                    scan_loss = cross_modal_token_contrastive_hinge_loss(image_tokens=image_tokens,
                                                                   text_tokens=text_tokens,
                                                                   text_mask=text_token_mask,
                                                                   raw_feature_norm=self.scan_feature_norm,
                                                                   agg_func=self.scan_agg_func,
                                                                   lambda_lse=self.scan_lambda_lse,
                                                                   lambda_softmax=self.scan_lambda_softmax,
                                                                   margin=self.scan_hinge_margin,
                                                                   hardest_negative=self.scan_hard_negative,
                                                                   use_hard_negative_text=self.scan_ce_hard_negative,
                                                                   valid_hard_negative_mask=valid_caption_mask)
                elif self.scan_loss_type == "sigmoid":
                    scan_loss = cross_modal_token_contrastive_sigmoid_loss(image_tokens=image_tokens,
                                                                   text_tokens=text_tokens,
                                                                   text_mask=text_token_mask,
                                                                   raw_feature_norm=self.scan_feature_norm,
                                                                   agg_func=self.scan_agg_func,
                                                                   lambda_lse=self.scan_lambda_lse,
                                                                   lambda_softmax=self.scan_lambda_softmax,
                                                                   use_hard_negative_text=self.scan_ce_hard_negative,
                                                                   valid_hard_negative_mask=valid_caption_mask)
                else:
                    raise RuntimeError(f"unknown SCAN loss type: {self.scan_loss_type}")

            if self.flair_loss:
                flair_loss = compute_flair_loss(image_tokens=image_tokens,
                                                text_features=gt_text_features,
                                                logits_scale=logit_scale,
                                                logits_bias=logit_bias,
                                                # raw_feature_norm=self.scan_feature_norm,
                                                # lambda_softmax=self.scan_lambda_softmax
                                                loss_feature_norm=self.loss_feature_norm
                                                )
        
        labels = torch.zeros_like(logits_per_image) - 1 # -1 (negative) everywhere except the diagonal
        labels.fill_diagonal_(1.) # diagonal elements are 1 - positive
        total_loss = -F.logsigmoid(labels * logits_per_image).sum() / logits_per_image.shape[1] #-torch.mean(F.logsigmoid(labels * logits_per_image))
            
        loss_outputs = {"contrastive_loss": total_loss}
        
        if self.cmr_loss:
            #total_loss += cmr_loss*self.cmr_loss_weight
            loss_outputs["cmr_loss"] = cmr_loss*self.cmr_loss_weight
        
        if self.imc_loss:
            #total_loss += imc_loss*self.imc_loss_weight
            loss_outputs["imc_loss"] = imc_loss*self.imc_loss_weight

        if self.scan_loss:
            loss_outputs["scan_loss"] = scan_loss*self.scan_loss_weight

        if self.flair_loss:
            loss_outputs["flair_loss"] = flair_loss*self.flair_loss_scale

        if self.np_loss:
            # nounphrase_loss would perform distributed gathering on its own, so do not pass all_xxx_features
            np_losses = nounphrase_loss(image_features=image_features,
                                        image_tokens=image_tokens,
                                        nounphrases_features=nounphrases_features,
                                        nounphrases_indices=nounphrases_indices,
                                        nounphrases_token_features=nounphrases_token_features,
                                        nounphrases_token_mask=nounphrases_token_mask,
                                        hn_nounphrases_features=hn_nounphrases_features,
                                        hn_nounphrases_indices=hn_nounphrases_indices,
                                        logit_scale=logit_scale,
                                        logit_bias=logit_bias,
                                        np_instance_loss=self.np_instance_loss,
                                        np_instance_loss_scale=self.np_loss_weight,
                                        np_token_loss=self.np_token_loss,
                                        np_intramodal_loss=self.np_intramodal_loss,
                                        np_intramodal_loss_scale=self.np_intramodal_loss_scale,
                                        np_token_loss_scale=self.np_token_loss_scale,
                                        np_token_token_loss=self.np_token_token_loss,
                                        np_token_token_loss_scale=self.np_token_token_loss_scale,
                                        np_flair_loss=self.np_flair_loss,
                                        np_flair_loss_scale=self.np_flair_loss_scale,
                                        np_hard_negative_loss=self.np_hard_negative_loss,
                                        np_hard_negative_loss_scale=self.np_hard_negative_loss_scale,
                                        np_hard_negative_flair_loss=self.np_hard_negative_flair_loss,
                                        np_hard_negative_flair_loss_scale=self.np_hard_negative_flair_loss_scale,
                                        token_loss_func=self.scan_loss_type,
                                        hinge_margin=self.scan_hinge_margin,
                                        rank=self.rank,
                                        world_size=self.world_size,
                                        agg_func=self.scan_agg_func,
                                        loss_feature_norm=self.loss_feature_norm)
            
            loss_outputs.update(np_losses)
            np_loss = sum(np_losses.values())
            loss_outputs["np_loss"] = np_loss
            
            
        #return total_loss, thresholds, cmr_loss, imc_loss
        if output_dict:
            loss_outputs["thresholds"] = thresholds # for CLIP-CE
            return loss_outputs
        
        return total_loss, loss_outputs["cmr_loss"] if self.cmr_loss else None, loss_outputs["imc_loss"] if self.imc_loss else None, loss_outputs["scan_loss"] if self.scan_loss else None, np_loss if self.np_loss else None, thresholds