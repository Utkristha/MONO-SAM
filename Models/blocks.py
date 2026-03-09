import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


# ── Depth Anything V2 Encoder Feature Extractor ───────────────────────────────
def get_depth_encoder_features(depth_model, raw_img_np):
    features = {}
    input_size = {}

    def hook(module, input, output):
        features['out'] = output

    handle = depth_model.pretrained.blocks[-1].register_forward_hook(hook)

    with torch.no_grad():
        image, (h, w) = depth_model.image2tensor(raw_img_np, input_size=518)
        input_size['h'] = image.shape[2] // 14  # patch size is 14
        input_size['w'] = image.shape[3] // 14
        depth_model.forward(image.to(DEVICE))

    handle.remove()

    enc = features['out'][:, 1:, :]  # remove CLS token
    B, N, C = enc.shape

    pH = input_size['h']
    pW = input_size['w']

    spatial_map = enc.reshape(B, pH, pW, C).permute(0, 3, 1, 2)  # [1, 384, pH, pW]
    return spatial_map


# ── Depth Projection: 384 → 256, any size → 64x64 ────────────────────────────
class DepthAnythingProjection(nn.Module):
    def __init__(self, in_channels=384, out_channels=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.proj(x)
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        return x  # [B, 256, 64, 64]


# ── Positional Embeddings ─────────────────────────────────────────────────────
class PositionalEmbeddings(nn.Module):
    def __init__(self, num_tokens=4096, embedding_dim=256):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, embedding_dim) * 0.02)

    def forward(self, x):
        return x + self.pos_embedding


# ── Transformer Block ─────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, nhead=8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        return self.transformer(x)


# ── Depth SAM Fusion ──────────────────────────────────────────────────────────
class DepthSAMFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_encoder        = PositionalEmbeddings()
        self.fusion_transformer = TransformerBlock()
        self.output_conv        = nn.Conv2d(256, 256, kernel_size=1)

    def forward(self, img_feature, depth_feature):
        B, C, H, W = img_feature.shape
        img_tokens   = img_feature.flatten(2).permute(0, 2, 1)
        depth_tokens = depth_feature.flatten(2).permute(0, 2, 1)
        img_tokens   = self.pos_encoder(img_tokens)
        depth_tokens = self.pos_encoder(depth_tokens)
        fused_tokens = self.fusion_transformer(img_tokens + depth_tokens)
        fused_grid   = fused_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        return self.output_conv(fused_grid)
# ── Depth Decoder ────────────────────────────────────────────────────────────────
class DepthDecoder(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        # takes fused + skip concatenated → 512 channels
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels * 2, 128, kernel_size=3, padding=1),  # 512 → 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, fused, skip, target_size=(480, 640)):
        x = torch.cat([fused, skip], dim=1)                              # [B, 512, 64, 64]
        x = self.decoder(x)                                              # [B, 1, 64, 64]
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x                                                         # [B, 1, 480, 640]
    
# ── Point prompt ────────────────────────────────────────────────────────────────
def get_point_prompt_from_mask(gt_mask_np, mask_size=256, sam_size=1024):
    ys, xs = np.where(gt_mask_np > 0)
    if len(xs) == 0:
        cx, cy = sam_size // 2, sam_size // 2
    else:
        scale = sam_size / mask_size          # 1024/256 = 4.0
        cx = int(xs.mean() * scale)           # scale to SAM space
        cy = int(ys.mean() * scale)

    point_coords = torch.tensor([[[cx, cy]]], dtype=torch.float32)
    point_labels = torch.tensor([[1]], dtype=torch.int)
    return point_coords, point_labels


# ── Full Model ────────────────────────────────────────────────────────────────
class MonocularSAMModel(nn.Module):
    def __init__(self, sam_model, depth_anything_model, depth_proj, fusion_layer, depth_decoder):
        super().__init__()
        self.sam           = sam_model
        self.depth_model   = depth_anything_model
        self.depth_proj    = depth_proj
        self.fusion_layer  = fusion_layer
        self.depth_decoder = depth_decoder          # ← new

        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False
        for param in self.depth_model.pretrained.parameters():
            param.requires_grad = False

    def forward(self, sam_feat, depth_feat, point_coords=None, point_labels=None):
            depth_embeddings = self.depth_proj(depth_feat)
            fused_embeddings = self.fusion_layer(sam_feat, depth_embeddings)

            B = fused_embeddings.shape[0]
            image_pe = self.sam.prompt_encoder.get_dense_pe()  # [1, C, H, W]

            all_masks = []
            all_ious  = []

            for b in range(B):
                # Slice per-sample features
                feat_b = fused_embeddings[b].unsqueeze(0)  # [1, C, H, W]

                # Build per-sample point prompt
                if point_coords is not None:
                    points_b = (
                        point_coords[b].unsqueeze(0),   # [1, 1, 2]
                        point_labels[b].unsqueeze(0),   # [1, 1]
                    )
                else:
                    points_b = None

                sparse_emb, dense_emb = self.sam.prompt_encoder(
                    points=points_b,
                    boxes=None,
                    masks=None,
                )
                # sparse_emb: [1, num_tokens, C]  dense_emb: [1, C, H, W]

                low_res_mask, iou_pred = self.sam.mask_decoder(
                    image_embeddings         = feat_b,
                    image_pe                 = image_pe,
                    sparse_prompt_embeddings = sparse_emb,
                    dense_prompt_embeddings  = dense_emb,
                    multimask_output         = False,
                )
                all_masks.append(low_res_mask)   # [1, 1, H, W]
                all_ious.append(iou_pred)         # [1, 1]

            low_res_masks   = torch.cat(all_masks, dim=0)   # [B, 1, H, W]
            iou_predictions = torch.cat(all_ious,  dim=0)   # [B, 1]

            pred_depth = self.depth_decoder(fused_embeddings, depth_embeddings)
            return low_res_masks, iou_predictions, pred_depth