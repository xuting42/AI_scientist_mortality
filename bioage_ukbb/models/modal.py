"""
MODAL (Multi-Organ Deep Aging Learner) implementation.

Cross-modal self-supervised learning combining OCT imaging and blood biomarkers
with contrastive learning and organ-specific aging subscores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import math
from einops import rearrange, repeat


class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings for Vision Transformer."""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image (batch, channels, height, width)
        
        Returns:
            Patch embeddings (batch, n_patches, embed_dim)
        """
        x = self.projection(x)  # (B, E, H/P, W/P)
        x = rearrange(x, 'b e h w -> b (h w) e')
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for Vision Transformer."""
    
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.projection_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
        
        Returns:
            Attention output (batch, seq_len, embed_dim)
        """
        B, N, E = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attention_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, E)
        
        # Final projection
        x = self.projection(x)
        x = self.projection_dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer block for Vision Transformer."""
    
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for OCT image processing."""
    
    def __init__(self, config: Any):
        super().__init__()
        
        self.config = config
        img_size = config.oct_image_size[0]
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            img_size=img_size,
            patch_size=config.vit_patch_size,
            embed_dim=config.vit_embed_dim
        )
        
        # Positional embedding
        n_patches = self.patch_embedding.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.vit_embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, n_patches + 1, config.vit_embed_dim)
        )
        self.dropout = nn.Dropout(config.vit_dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                config.vit_embed_dim,
                config.vit_num_heads,
                config.vit_mlp_ratio,
                config.vit_dropout
            )
            for _ in range(config.vit_depth)
        ])
        
        self.norm = nn.LayerNorm(config.vit_embed_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (batch, channels, height, width)
        
        Returns:
            Image features (batch, embed_dim)
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embedding(x)
        
        # Add CLS token
        cls_tokens = repeat(self.cls_token, '1 1 e -> b 1 e', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Return CLS token as image representation
        return x[:, 0]


class BiomarkerMLP(nn.Module):
    """MLP for processing blood biomarker features."""
    
    def __init__(self, config: Any):
        super().__init__()
        
        input_dim = len(config.biomarker_features)
        hidden_dims = config.mlp_hidden_dims
        dropout = config.mlp_dropout
        activation = config.mlp_activation
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output_dim = prev_dim
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'leaky_relu': nn.LeakyReLU(0.1)
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Biomarker features (batch, n_features)
        
        Returns:
            Processed features (batch, output_dim)
        """
        return self.mlp(x)


class ContrastiveLearning(nn.Module):
    """Contrastive learning module for cross-modal alignment."""
    
    def __init__(
        self,
        image_dim: int,
        biomarker_dim: int,
        projection_dim: int = 128,
        temperature: float = 0.07
    ):
        super().__init__()
        
        self.temperature = temperature
        
        # Projection heads
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        self.biomarker_projection = nn.Sequential(
            nn.Linear(biomarker_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def forward(
        self,
        image_features: torch.Tensor,
        biomarker_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss between image and biomarker features.
        
        Args:
            image_features: Image features (batch, image_dim)
            biomarker_features: Biomarker features (batch, biomarker_dim)
        
        Returns:
            Contrastive loss
        """
        # Project features
        image_proj = F.normalize(self.image_projection(image_features), dim=-1)
        biomarker_proj = F.normalize(self.biomarker_projection(biomarker_features), dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(image_proj, biomarker_proj.t()) / self.temperature
        
        # Contrastive loss (InfoNCE)
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)
        
        loss_i2b = F.cross_entropy(logits, labels)
        loss_b2i = F.cross_entropy(logits.t(), labels)
        
        return (loss_i2b + loss_b2i) / 2


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion for multi-modal features."""
    
    def __init__(
        self,
        image_dim: int,
        biomarker_dim: int,
        fusion_dim: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Project to same dimension
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.biomarker_proj = nn.Linear(biomarker_dim, fusion_dim)
        
        # Cross-attention layers
        self.image_to_bio_attention = nn.MultiheadAttention(
            fusion_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.bio_to_image_attention = nn.MultiheadAttention(
            fusion_dim, n_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim)
        )
        
        self.output_dim = fusion_dim
    
    def forward(
        self,
        image_features: torch.Tensor,
        biomarker_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse multi-modal features using cross-attention.
        
        Args:
            image_features: Image features (batch, image_dim)
            biomarker_features: Biomarker features (batch, biomarker_dim)
        
        Returns:
            Fused features (batch, fusion_dim)
        """
        # Project features
        image_feat = self.image_proj(image_features).unsqueeze(1)
        bio_feat = self.biomarker_proj(biomarker_features).unsqueeze(1)
        
        # Cross-attention
        attended_bio, _ = self.image_to_bio_attention(
            bio_feat, image_feat, image_feat
        )
        attended_image, _ = self.bio_to_image_attention(
            image_feat, bio_feat, bio_feat
        )
        
        # Residual connections
        attended_bio = self.norm1(attended_bio + bio_feat)
        attended_image = self.norm2(attended_image + image_feat)
        
        # Concatenate and feed-forward
        fused = torch.cat([
            attended_image.squeeze(1),
            attended_bio.squeeze(1)
        ], dim=-1)
        
        fused = self.ffn(fused)
        
        return fused


class OrganSpecificHeads(nn.Module):
    """Organ-specific prediction heads for subscores."""
    
    def __init__(
        self,
        input_dim: int,
        organ_systems: List[str],
        hidden_dim: int = 64
    ):
        super().__init__()
        
        self.organ_systems = organ_systems
        
        # Create a prediction head for each organ system
        self.heads = nn.ModuleDict({
            organ: nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1)
            )
            for organ in organ_systems
        })
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute organ-specific aging scores.
        
        Args:
            x: Input features (batch, input_dim)
        
        Returns:
            Dictionary of organ-specific scores
        """
        scores = {}
        for organ, head in self.heads.items():
            scores[organ] = head(x)
        
        return scores


class MODAL(nn.Module):
    """
    Multi-Organ Deep Aging Learner.
    
    Combines OCT imaging and blood biomarkers through cross-modal learning
    with contrastive alignment and organ-specific aging assessment.
    """
    
    def __init__(self, config: Any):
        """
        Initialize MODAL model.
        
        Args:
            config: MODAL configuration object
        """
        super().__init__()
        self.config = config
        
        # Vision Transformer for OCT images
        if config.use_oct_imaging:
            self.vision_transformer = VisionTransformer(config)
            image_dim = config.vit_embed_dim
        else:
            self.vision_transformer = None
            image_dim = 0
        
        # MLP for biomarkers
        if config.use_blood_biomarkers:
            self.biomarker_mlp = BiomarkerMLP(config)
            biomarker_dim = self.biomarker_mlp.output_dim
        else:
            self.biomarker_mlp = None
            biomarker_dim = 0
        
        # Contrastive learning
        if config.use_contrastive_learning and image_dim > 0 and biomarker_dim > 0:
            self.contrastive = ContrastiveLearning(
                image_dim,
                biomarker_dim,
                config.contrastive_projection_dim,
                config.contrastive_temperature
            )
        else:
            self.contrastive = None
        
        # Multi-modal fusion
        if config.fusion_method == "cross_attention" and image_dim > 0 and biomarker_dim > 0:
            self.fusion = CrossAttentionFusion(
                image_dim,
                biomarker_dim,
                config.fusion_dim,
                config.fusion_heads,
                config.fusion_dropout
            )
            fusion_output_dim = self.fusion.output_dim
        elif config.fusion_method == "concat":
            self.fusion = None
            fusion_output_dim = image_dim + biomarker_dim
        else:
            self.fusion = None
            fusion_output_dim = max(image_dim, biomarker_dim)
        
        # Final prediction network
        self.prediction_network = nn.Sequential(
            nn.Linear(fusion_output_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Main age prediction head
        self.age_head = nn.Linear(64, 1)
        
        # Organ-specific heads
        if config.compute_subscores:
            self.organ_heads = OrganSpecificHeads(
                64,
                config.organ_systems
            )
        else:
            self.organ_heads = None
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def extract_features(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extract features from each modality.
        
        Args:
            inputs: Dictionary containing 'oct_image' and/or 'biomarkers'
        
        Returns:
            Tuple of (image_features, biomarker_features)
        """
        image_features = None
        biomarker_features = None
        
        # Extract OCT features
        if self.vision_transformer is not None and 'oct_image' in inputs:
            image_features = self.vision_transformer(inputs['oct_image'])
        
        # Extract biomarker features
        if self.biomarker_mlp is not None and 'biomarkers' in inputs:
            biomarker_features = self.biomarker_mlp(inputs['biomarkers'])
        
        return image_features, biomarker_features
    
    def fuse_features(
        self,
        image_features: Optional[torch.Tensor],
        biomarker_features: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Fuse multi-modal features.
        
        Args:
            image_features: OCT image features
            biomarker_features: Blood biomarker features
        
        Returns:
            Fused features
        """
        if image_features is None:
            return biomarker_features
        
        if biomarker_features is None:
            return image_features
        
        if self.fusion is not None:
            # Use cross-attention fusion
            fused = self.fusion(image_features, biomarker_features)
        elif self.config.fusion_method == "concat":
            # Simple concatenation
            fused = torch.cat([image_features, biomarker_features], dim=-1)
        elif self.config.fusion_method == "add":
            # Element-wise addition (requires same dimension)
            fused = image_features + biomarker_features
        else:
            # Default to concatenation
            fused = torch.cat([image_features, biomarker_features], dim=-1)
        
        return fused
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        compute_contrastive_loss: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MODAL.
        
        Args:
            inputs: Dictionary containing:
                - 'oct_image': OCT images (batch, 3, H, W)
                - 'biomarkers': Blood biomarkers (batch, n_features)
            compute_contrastive_loss: Whether to compute contrastive loss
        
        Returns:
            Dictionary containing predictions and optional outputs
        """
        # Extract features from each modality
        image_features, biomarker_features = self.extract_features(inputs)
        
        # Fuse features
        fused_features = self.fuse_features(image_features, biomarker_features)
        
        # Pass through prediction network
        features = self.prediction_network(fused_features)
        
        # Main age prediction
        age_prediction = self.age_head(features)
        
        outputs = {
            'prediction': age_prediction,
            'features': features
        }
        
        # Compute organ-specific scores
        if self.organ_heads is not None:
            organ_scores = self.organ_heads(features)
            outputs['organ_scores'] = organ_scores
            
            # Compute combined score
            combined_score = torch.stack(
                list(organ_scores.values()), dim=-1
            ).mean(dim=-1, keepdim=True)
            outputs['combined_organ_score'] = combined_score
        
        # Compute contrastive loss if requested
        if compute_contrastive_loss and self.contrastive is not None:
            if image_features is not None and biomarker_features is not None:
                contrastive_loss = self.contrastive(
                    image_features, biomarker_features
                )
                outputs['contrastive_loss'] = contrastive_loss
        
        # Add modality-specific features
        if image_features is not None:
            outputs['image_features'] = image_features
        if biomarker_features is not None:
            outputs['biomarker_features'] = biomarker_features
        
        return outputs
    
    def get_organ_interpretation(
        self,
        organ_scores: Dict[str, torch.Tensor]
    ) -> Dict[str, str]:
        """
        Interpret organ-specific aging scores.
        
        Args:
            organ_scores: Dictionary of organ scores
        
        Returns:
            Dictionary of interpretations
        """
        interpretations = {}
        
        for organ, score in organ_scores.items():
            score_val = score.mean().item()
            
            if score_val > 5:
                status = "accelerated aging"
            elif score_val < -5:
                status = "decelerated aging"
            else:
                status = "normal aging"
            
            interpretations[organ] = f"{organ}: {status} (score: {score_val:.2f})"
        
        return interpretations