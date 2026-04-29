import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared loss
# ---------------------------------------------------------------------------

class WingLoss(nn.Module):
    """
    Wing loss for keypoint regression (Feng et al., 2018).

    Heavily penalises small-to-medium errors relative to MSE, which matters
    for pose estimation where sub-pixel accuracy on each joint is the goal.
    Falls back to a linear term for large errors to stay bounded.

    Args:
        w: width of the non-linear region (default 10.0)
        eps: curvature of the log region (default 2.0)
    """
    def __init__(self, w: float = 10.0, eps: float = 2.0) -> None:
        super().__init__()
        self.w = w
        self.eps = eps
        # C makes the function continuous at the boundary |x| = w
        self.C = w - w * torch.log(torch.tensor(1.0 + w / eps))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = (pred - target).abs()
        # Non-linear region for small errors; linear region for large errors
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1.0 + diff / self.eps),
            diff - self.C.to(diff.device),
        )
        return loss.mean()


# ---------------------------------------------------------------------------
# SimpleRadarPoseCNN  (improved)
# ---------------------------------------------------------------------------
# Changes from previous version:
#   - Backbone widened: 2 -> 32 -> 64 -> 128 -> 256 -> 256 (5 blocks)
#   - SpatialDropout2d replaces Dropout in backbone: drops entire feature maps,
#     better regularisation for convolutional layers than element-wise dropout
#   - MaxPool replaced by stride-2 conv in the last two blocks so the network
#     can learn its own downsampling rather than discarding via max
#   - Global Average Pooling replaces flatten+huge Linear, retaining spatial
#     signal and cutting the param-wasting transition layer
#   - GELU activation: smoother gradient flow than ReLU for regression tasks
#   - Per-joint regression heads: each of the 17 joints gets its own small MLP
#     so joints can specialise (wrist behaves differently from hip)
# ---------------------------------------------------------------------------

class _ConvBnGelu(nn.Module):
    """Conv2d -> BN -> GELU building block."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleRadarPoseCNN(nn.Module):
    """
    Improved CNN for radar-to-pose regression.

    Input:  (B, 2, 256, 128)
    Output: (B, num_keypoints, 2)
    """

    def __init__(self, num_keypoints: int = 17, dropout: float = 0.2) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints

        # Wider backbone — 5 stages, stride-2 on stages 3-5 to learn downsampling
        self.backbone = nn.Sequential(
            _ConvBnGelu(2,   32,  stride=1),   # (32, 256, 128)
            nn.MaxPool2d(2),                    # (32, 128,  64)

            _ConvBnGelu(32,  64,  stride=1),   # (64, 128,  64)
            nn.MaxPool2d(2),                    # (64,  64,  32)

            _ConvBnGelu(64,  128, stride=1),   # (128,  64,  32)
            # SpatialDropout drops whole channels — better for conv than element-wise
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(2),                    # (128,  32,  16)

            # Learned stride-2 downsampling from here: lets network decide what to keep
            _ConvBnGelu(128, 256, stride=2),   # (256,  16,   8)
            nn.Dropout2d(p=dropout),

            _ConvBnGelu(256, 256, stride=2),   # (256,   8,   4)
        )

        # Global Average Pooling: (B, 256, 8, 4) -> (B, 256)
        # Retains spatial signal without a huge flatten linear layer
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Per-joint heads: each joint gets its own MLP from the 256-d feature vector
        self.joint_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 2),
            )
            for _ in range(num_keypoints)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)          # (B, 256, 8, 4)
        feat = self.gap(feat).squeeze(-1).squeeze(-1)  # (B, 256)

        # Stack per-joint predictions
        coords = torch.stack(
            [head(feat) for head in self.joint_heads], dim=1
        )  # (B, num_keypoints, 2)
        return coords


# ---------------------------------------------------------------------------
# Shared ResBlock (used by RadarPoseResTransformer backbone)
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """Pre-activation ResBlock: BN -> GELU -> Conv -> BN -> GELU -> Conv + shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        # Pre-activation (He et al. 2016 v2): normalise before activation,
        # which improves gradient flow in deeper networks
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.gelu(self.bn1(x)))
        out = self.conv2(F.gelu(self.bn2(out)))
        return out + self.shortcut(x)


# ---------------------------------------------------------------------------
# RadarPoseResTransformer  (improved)
# ---------------------------------------------------------------------------
# Changes from previous version:
#   - Wider backbone: 2->32->64->128->256, giving richer spatial features
#   - Spatial token pooling replaces the single giant Linear(16384, 17*128):
#     a small conv produces one spatial map per joint, then adaptive avg-pool
#     extracts a d_model vector per joint — each joint learns where to look
#   - Learnable positional embeddings on transformer tokens: attention now
#     knows which token corresponds to which joint
#   - Deeper transformer: 4 layers, d_model=256, 8 heads (was 2/128/4)
#   - Wider per-joint heads: 256->128->64->2 (was 128->64->2)
#   - Pre-LN transformer layers (norm_first=True): more stable training
# ---------------------------------------------------------------------------

class RadarPoseResTransformer(nn.Module):
    """
    Improved ResBlock CNN backbone + Transformer head for radar-to-pose regression.

    Input:  (B, 2, 256, 128)
    Output: (B, num_keypoints, 2)
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        d_model: int = 256,
        nhead: int = 8,
        num_transformer_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.d_model = d_model

        # Wider backbone: more channels for richer spatial representations
        self.backbone = nn.Sequential(
            _ResBlock(2,   32,  stride=2),   # (32,  128, 64)
            _ResBlock(32,  64,  stride=2),   # (64,   64, 32)
            _ResBlock(64,  128, stride=2),   # (128,  32, 16)
            _ResBlock(128, 256, stride=2),   # (256,  16,  8)
        )

        # Spatial token pooling: one conv per joint extracts a d_model-sized
        # descriptor by attending to its own spatial region of the feature map.
        # Far more efficient and expressive than a single flat Linear projection.
        self.token_conv = nn.Conv2d(256, num_keypoints * d_model, kernel_size=1)
        self.token_pool = nn.AdaptiveAvgPool2d(1)  # collapses H,W -> (B, K*d, 1, 1)

        # Learnable positional embeddings: tells attention which token is which joint
        self.pos_embed = nn.Parameter(torch.randn(1, num_keypoints, d_model) * 0.02)

        # Deeper transformer with pre-LN (norm_first) for more stable training
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: normalise inputs before attention/FFN
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
            enable_nested_tensor=False,
        )

        # Wider per-joint heads: more capacity to decode each joint's token
        self.joint_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 128),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 2),
            )
            for _ in range(num_keypoints)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        feat = self.backbone(x)                        # (B, 256, 16, 8)

        # Per-joint spatial pooling: conv expands channels, pool collapses spatial
        tokens = self.token_conv(feat)                 # (B, K*d_model, 16, 8)
        tokens = self.token_pool(tokens)               # (B, K*d_model, 1, 1)
        tokens = tokens.view(B, self.num_keypoints, self.d_model)  # (B, K, d)

        # Add positional embeddings so attention knows joint identity
        tokens = tokens + self.pos_embed

        tokens = self.transformer(tokens)              # (B, K, d_model)

        coords = torch.stack(
            [head(tokens[:, i, :]) for i, head in enumerate(self.joint_heads)],
            dim=1,
        )  # (B, K, 2)
        return coords


# ---------------------------------------------------------------------------
# RadarPoseHeatmapUNet
# ---------------------------------------------------------------------------
# Architecture motivation:
#   Both existing models regress (x, y) directly from a global feature vector.
#   The dominant paradigm in pose estimation (HRNet, ViTPose, SimpleBaseline)
#   instead predicts per-joint *spatial heatmaps* and extracts coordinates via
#   soft-argmax. This works better for keypoint accuracy because:
#     - The supervision signal is dense (every pixel contributes a gradient)
#     - The network learns an explicit spatial belief, not a memorised XY pair
#     - Soft-argmax is differentiable and provides sub-pixel localisation
#
# Design choices:
#   - U-Net topology: encoder downsamples 4x, decoder upsamples back with skip
#     connections from each encoder stage. Skip connections feed high-resolution
#     spatial detail back to the decoder, critical for sub-pixel accuracy.
#   - Heatmap head: 1x1 conv produces (B, K, H, W). Heatmap resolution is 1/2
#     of input (128 x 64) — fine enough for accurate soft-argmax.
#   - Soft-argmax with softmax over spatial dims: differentiable, no temperature
#     hyperparameter to tune. Output coords are in [0, 1] matching bbox-norm targets.
# ---------------------------------------------------------------------------

def _soft_argmax_2d(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Differentiable spatial soft-argmax.

    Args:
        heatmaps: (B, K, H, W) raw logits.
    Returns:
        coords: (B, K, 2) with x, y in [0, 1].
    """
    B, K, H, W = heatmaps.shape
    flat = heatmaps.view(B, K, -1)
    probs = F.softmax(flat, dim=-1).view(B, K, H, W)

    # Coordinate grids in [0, 1]
    xs = torch.linspace(0, 1, W, device=heatmaps.device, dtype=heatmaps.dtype).view(1, 1, 1, W)
    ys = torch.linspace(0, 1, H, device=heatmaps.device, dtype=heatmaps.dtype).view(1, 1, H, 1)

    x = (probs * xs).sum(dim=(2, 3))
    y = (probs * ys).sum(dim=(2, 3))
    return torch.stack([x, y], dim=-1)


class _UpBlock(nn.Module):
    """Decoder block: upsample, concat skip, fuse with conv-bn-gelu."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class RadarPoseHeatmapUNet(nn.Module):
    """
    U-Net heatmap regressor with soft-argmax for radar-to-pose estimation.

    Input:  (B, 2, 256, 128)
    Output: (B, num_keypoints, 2)   coords in [0, 1]

    Pipeline:
        Encoder (5 stages, downsampling 4x)
        Bottleneck
        Decoder (3 upsample blocks with skip connections)
        Heatmap head: 1x1 conv -> (B, K, 128, 64)
        Soft-argmax -> (B, K, 2)
    """

    def __init__(self, num_keypoints: int = 17, base_ch: int = 32) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints

        c1 = base_ch        # 32
        c2 = base_ch * 2    # 64
        c3 = base_ch * 4    # 128
        c4 = base_ch * 8    # 256
        c5 = base_ch * 8    # 256 (bottleneck stays 256)

        # Encoder — each stage: ResBlock that downsamples by 2 (except first)
        self.enc1 = _ResBlock(2, c1, stride=1)        # (32, 256, 128)  skip
        self.enc2 = _ResBlock(c1, c2, stride=2)       # (64, 128, 64)   skip
        self.enc3 = _ResBlock(c2, c3, stride=2)       # (128, 64, 32)   skip
        self.enc4 = _ResBlock(c3, c4, stride=2)       # (256, 32, 16)   skip
        self.bottleneck = _ResBlock(c4, c5, stride=2) # (256, 16, 8)

        # Decoder — symmetric upsampling with skip connections
        self.dec4 = _UpBlock(c5, c4, c4)              # -> (256, 32, 16)
        self.dec3 = _UpBlock(c4, c3, c3)              # -> (128, 64, 32)
        self.dec2 = _UpBlock(c3, c2, c2)              # -> (64, 128, 64)

        # Heatmap head: 1x1 conv producing K spatial heatmaps
        # Stops at 128x64 — high enough for accurate soft-argmax
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.Conv2d(c2, num_keypoints, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder with skip captures
        s1 = self.enc1(x)            # (32, 256, 128)
        s2 = self.enc2(s1)           # (64, 128, 64)
        s3 = self.enc3(s2)           # (128, 64, 32)
        s4 = self.enc4(s3)           # (256, 32, 16)
        b  = self.bottleneck(s4)     # (256, 16, 8)

        # Decoder with skip connections
        d4 = self.dec4(b,  s4)       # (256, 32, 16)
        d3 = self.dec3(d4, s3)       # (128, 64, 32)
        d2 = self.dec2(d3, s2)       # (64, 128, 64)

        heatmaps = self.heatmap_head(d2)   # (B, K, 128, 64)
        coords = _soft_argmax_2d(heatmaps) # (B, K, 2)
        return coords

    def forward_with_heatmaps(self, x: torch.Tensor):
        """Helper for training with auxiliary heatmap loss. Returns (coords, heatmaps)."""
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b  = self.bottleneck(s4)
        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        heatmaps = self.heatmap_head(d2)
        coords = _soft_argmax_2d(heatmaps)
        return coords, heatmaps


# ---------------------------------------------------------------------------
# RadarPoseDETR
# ---------------------------------------------------------------------------
# Architecture motivation:
#   The existing RadarPoseResTransformer pools spatial info into per-joint
#   tokens *before* attention runs. That decision is made by a convolution,
#   not by attention itself. This DETR-style variant flips that: we keep the
#   spatial feature map intact and use 17 learnable joint queries that
#   cross-attend to it. Each joint query learns *which spatial regions to
#   attend to*, end-to-end with the loss. This is the architecture used by
#   TokenPose, POET, and PETR for pose estimation.
#
# Design choices:
#   - CNN encoder: same backbone as RadarPoseResTransformer (proven works)
#   - Flatten spatial dims into a sequence of 128 spatial tokens (16*8)
#   - Add 2D positional encoding to the sequence so attention can localise
#   - 17 learnable joint queries (one per keypoint)
#   - Transformer decoder with 4 layers: each layer does self-attention over
#     joint queries (joint relationships) AND cross-attention to encoder tokens
#     (spatial localisation). Both happen, both supervised end-to-end.
#   - Per-joint regression heads
# ---------------------------------------------------------------------------

class RadarPoseDETR(nn.Module):
    """
    DETR-style radar pose estimator: CNN encoder + transformer decoder with
    learnable joint queries that cross-attend to spatial features.

    Input:  (B, 2, 256, 128)
    Output: (B, num_keypoints, 2)
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.d_model = d_model

        # CNN encoder — same proven backbone shape as RadarPoseResTransformer
        self.backbone = nn.Sequential(
            _ResBlock(2,   32,  stride=2),   # (32, 128, 64)
            _ResBlock(32,  64,  stride=2),   # (64, 64, 32)
            _ResBlock(64,  128, stride=2),   # (128, 32, 16)
            _ResBlock(128, d_model, stride=2),  # (d_model, 16, 8)
        )

        # Compute encoder output spatial size (16 * 8 = 128 spatial tokens)
        with torch.no_grad():
            dummy = torch.zeros(1, 2, 256, 128)
            feat = self.backbone(dummy)
            _, _, H, W = feat.shape
        self.feat_h, self.feat_w = H, W
        num_spatial = H * W  # 128

        # Learnable 2D positional encoding for the spatial tokens
        # One vector per spatial location, telling attention "where" each token came from
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, num_spatial, d_model) * 0.02)

        # Learnable joint queries: K vectors that will cross-attend to spatial features
        # Initialised small so the network learns useful queries from scratch
        self.joint_queries = nn.Parameter(torch.randn(1, num_keypoints, d_model) * 0.02)

        # Transformer decoder — 4 layers, pre-LN for stable training
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Per-joint regression heads
        self.joint_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 128),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 2),
            )
            for _ in range(num_keypoints)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # CNN encoder -> (B, d_model, H, W)
        feat = self.backbone(x)

        # Flatten spatial dims into a sequence: (B, H*W, d_model)
        memory = feat.flatten(2).transpose(1, 2)
        memory = memory + self.spatial_pos_embed   # add 2D positional info

        # Expand joint queries to batch size
        queries = self.joint_queries.expand(B, -1, -1)  # (B, K, d_model)

        # Decoder: queries cross-attend to spatial features
        # self-attention on queries lets joints reason about each other,
        # cross-attention lets each query look at any spatial location
        decoded = self.decoder(tgt=queries, memory=memory)  # (B, K, d_model)

        coords = torch.stack(
            [head(decoded[:, i, :]) for i, head in enumerate(self.joint_heads)],
            dim=1,
        )  # (B, K, 2)
        return coords
