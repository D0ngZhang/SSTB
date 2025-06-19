import torch.nn as nn
import torch.nn.functional as F
import torch
from swinunetr import *


def create_patches_and_mask(input_tensor, patch_size=16, mask_ratio=0.75):
    # input_tensor shape: [b, 1, 256, 256]
    b, c, h, w = input_tensor.shape

    # Check if the input size is divisible by patch size
    assert h % patch_size == 0 and w % patch_size == 0, "Input dimensions must be divisible by patch size"

    # Create patches
    patches = input_tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(b, -1, c, patch_size, patch_size)

    # Number of patches
    num_patches = patches.size(1)

    # Create mask
    mask = torch.rand(b, num_patches).to(torch.device("cuda")) > mask_ratio

    # Apply mask
    masked_patches = patches * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    restored_tensor = masked_patches.view(b, h // patch_size, w // patch_size, c, patch_size, patch_size)
    restored_tensor = restored_tensor.permute(0, 3, 1, 4, 2, 5).contiguous()
    restored_tensor = restored_tensor.view(b, c, h, w)

    mask = mask.view(b, h // patch_size, w // patch_size)
    mask = mask.unsqueeze(1)  # Add channel dimension
    mask = F.interpolate(mask.float(), size=(h, w), mode='nearest')

    return restored_tensor, mask

class SSLModel(nn.Module):
    def __init__(self):
        super(SSLModel, self).__init__()

        self.semantic_head = SemanticHead()
        self.intensity_head = SwinUNETR(img_size=[256, 256], in_channels=1, out_channels=1, spatial_dims=2)

    def forward(self, x):
        projections, saliency_map = self.semantic_head(x)
        saliency_map = F.interpolate(saliency_map.detach(), size=(256, 256), mode='nearest')

        random_masked_x, masks = create_patches_and_mask(x)
        saliency_masked_x = saliency_map * x
        reminder = torch.maximum(masks, saliency_map)
        reminder_x = reminder * x
        restored_x = self.intensity_head(reminder_x)
        return projections, restored_x, random_masked_x, saliency_masked_x, reminder_x

# semantic contrastive learning branch
class SemanticHead(nn.Module):
    def __init__(self, embed_dim=384):
        super(SemanticHead, self).__init__()
        self.swin = SwinUNETR(img_size=[256, 256], in_channels=1, out_channels=1, spatial_dims=2)
        self.normalize = self.swin.normalize
        self.proj1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 128), nn.LeakyReLU(negative_slope=0.1),
                                   nn.LayerNorm(128), nn.Linear(128, 64), nn.LeakyReLU(negative_slope=0.1),
                                   nn.LayerNorm(64), nn.Linear(64, 32), nn.LeakyReLU(negative_slope=0.1),
                                   nn.LayerNorm(32), nn.Linear(32, 1))

        self.proj2 = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 32), nn.LeakyReLU(negative_slope=0.1),
                                   nn.LayerNorm(32), nn.Linear(32, 16), nn.LeakyReLU(negative_slope=0.1),
                                   nn.LayerNorm(16), nn.Linear(16, 16))

    def forward(self, x):
        hidden_states_out = self.swin.swinViT(x, self.normalize)

        features = hidden_states_out[4]
        # print("features", features.shape)

        proj = self.proj1(features.permute(0, 2, 3, 1))
        # print("proj", proj.shape)

        b, h, w, c = proj.shape

        proj = proj.view(b, h*w)

        proj = self.proj2(proj)

        importance_scores = torch.norm(features.detach(), dim=1, keepdim=True)
        # print("importance_scores", importance_scores.shape)
        saliency_map = importance_scores.view(-1, 1, 8, 8)  # saliency_map shape: [b, 1, num_patch_height, num_patch_width]
        binary_saliency_map = torch.zeros_like(saliency_map).to(saliency_map.device)

        for i in range(saliency_map.size(0)):
            single_saliency_map = saliency_map[i]

            single_saliency_map = (single_saliency_map - single_saliency_map.min()) / (
                        single_saliency_map.max() - single_saliency_map.min())

            flat_map = single_saliency_map.flatten()
            # print(flat_map.shape)
            topk_values, topk_indices = torch.topk(flat_map, 16)

            binary_flat_map = torch.zeros_like(flat_map).to(saliency_map.device)
            binary_flat_map[topk_indices] = 1.0
            binary_saliency_map[i] = binary_flat_map.reshape(single_saliency_map.shape)

        return proj, binary_saliency_map



class InfoNCE(nn.Module):

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)

def transpose(x):
    return x.transpose(-2, -1)

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

def info_nce(query, positive_key,
             negative_keys=None, temperature=0.1,
             reduction='mean', negative_mode='unpaired'):

    if query.dim() != 2:
        raise ValueError('query must be 2D tensor')
    if positive_key.dim() != 2:
        raise ValueError('positive_key must be 2D tensor')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError('negative_keys must be 2D tensor for negative_mode=unpaired')
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError('negative_keys must be 3D tensor for negative_mode=paired')

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)

    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)  # (N, 1)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)  # (N, M)
        elif negative_mode == 'paired':
            query = query.unsqueeze(1)  # (N, 1, D)
            negative_logits = query @ transpose(negative_keys) # (N, 1, M)
            negative_logits = negative_logits.squeeze(1)  # (N, M)

        logits = torch.cat([positive_logit, negative_logits], dim=1)  # (N, 1+M)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)  # (N,)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)  # (N, N)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)