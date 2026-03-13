import torch
from .loss import func_attention


class SiglipFlairWrapper(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, feature_norm="clipped_l2norm", lambda_softmax=9.):
        super().__init__()
        self.model = base_model
        setattr(self.model.visual, "output_tokens", True)
        setattr(self.model.text, "output_tokens", True)
        setattr(self.model, "output_tokens", True)
        self.feature_norm = feature_norm
        self.lambda_softmax = lambda_softmax
        self.is_true_flair = False
        
    @torch.no_grad()
    def encode_text(self, text, normalize=True):
        return self.model.encode_text(text, normalize=normalize, output_tokens=False)
    

    @torch.no_grad()
    def encode_image(self, image, normalize=True):
        _, image_tokens = self.model.encode_image(image, normalize=normalize) #(N_im, seq_len, feat_dim)
        return image_tokens

    @torch.no_grad()
    def text_conditioned(self, image_token_features, text_features, normalize=True):
        N_im = image_token_features.shape[0]
        N_txt = text_features.shape[0]
        # assert N == text_features.shape[0]
        
        all_image_features = []
        text_features = text_features.unsqueeze(1) #(text_batch, 1, feat_dim)
        
        for i in range(N_txt):
            query_features = text_features[i:i+1].expand(N_im, -1, -1)
            # (N_im, 1, feat_dim)
            image_features, _ = func_attention(query=query_features, context=image_token_features, raw_feature_norm=self.feature_norm, smooth=self.lambda_softmax)
            all_image_features.append(image_features)
        
        all_image_features = torch.cat(all_image_features, dim=1) # (N_im, N_txt, feature_dim)
        if normalize:
            all_image_features = torch.nn.functional.normalize(all_image_features, dim=-1)
        return all_image_features
    

    
    @torch.no_grad()
    def encode_image_end2end(self, image, text_features, normalize=True):
        # batch size of image can be different from batch size of text features
        # image: (N_im, 3, H, W)
        # text_features: (N_txt, feat_dim)

        # get image token features
        _, image_tokens = self.model.encode_image(image, normalize=normalize) #(N_im, seq_len, feat_dim)
        N_im = image_tokens.shape[0]
        N_txt = text_features.shape[0]
        # assert N == text_features.shape[0]
        
        all_image_features = []
        text_features = text_features.unsqueeze(1) #(text_batch, 1, feat_dim)
        
        for i in range(N_txt):
            query_features = text_features[i:i+1].expand(N_im, -1, -1)
            # (N_im, 1, feat_dim)
            image_features, _ = func_attention(query=query_features, context=image_tokens, raw_feature_norm=self.feature_norm, smooth=self.lambda_softmax)
            all_image_features.append(image_features)
        
        all_image_features = torch.cat(all_image_features, dim=1) # (N_im, N_txt, feature_dim)
        if normalize:
            all_image_features = torch.nn.functional.normalize(all_image_features, dim=-1)
        return all_image_features
    
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError