"""
HENAW (Hierarchical Elastic Net with Adaptive Weighting) Model Implementation
PyTorch implementation for biological age prediction using UK Biobank data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class HENAWOutput:
    """Output container for HENAW model predictions"""
    biological_age: torch.Tensor
    mortality_risk: Optional[torch.Tensor] = None
    morbidity_risks: Optional[Dict[str, torch.Tensor]] = None
    system_embeddings: Optional[torch.Tensor] = None
    biomarker_weights: Optional[torch.Tensor] = None
    feature_importance: Optional[torch.Tensor] = None


class AdaptiveWeightingModule(nn.Module):
    """
    Adaptive weighting mechanism based on coefficient of variation
    w_i(t) = exp(-λ·CV_i(t)) / Σⱼ exp(-λ·CV_j(t))
    """
    
    def __init__(self, n_biomarkers: int, lambda_param: float = 1.0, temperature: float = 0.5):
        super().__init__()
        self.n_biomarkers = n_biomarkers
        self.lambda_param = lambda_param
        self.temperature = temperature
        
        # Learnable parameters for age-dependent weighting
        self.age_weight_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, n_biomarkers),
            nn.Softmax(dim=-1)
        )
        
        # Running statistics for coefficient of variation
        self.register_buffer('running_mean', torch.zeros(n_biomarkers))
        self.register_buffer('running_var', torch.ones(n_biomarkers))
        self.register_buffer('running_count', torch.tensor(0.0))
        
    def update_statistics(self, x: torch.Tensor) -> None:
        """Update running statistics for coefficient of variation calculation"""
        with torch.no_grad():
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0)
            
            # Exponential moving average
            alpha = 0.1
            self.running_mean = (1 - alpha) * self.running_mean + alpha * batch_mean
            self.running_var = (1 - alpha) * self.running_var + alpha * batch_var
            self.running_count += 1
    
    def forward(self, x: torch.Tensor, age: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute adaptive weights for biomarkers
        
        Args:
            x: Biomarker values [batch_size, n_biomarkers]
            age: Age values [batch_size, 1]
        
        Returns:
            weights: Adaptive weights [batch_size, n_biomarkers]
        """
        # Update statistics during training
        if self.training:
            self.update_statistics(x)
        
        # Compute coefficient of variation
        cv = torch.sqrt(self.running_var) / (torch.abs(self.running_mean) + 1e-8)
        
        # Base weights from CV
        base_weights = F.softmax(-self.lambda_param * cv / self.temperature, dim=-1)
        
        # Age-dependent adjustment if provided
        if age is not None:
            age_norm = (age - 50.0) / 20.0  # Normalize age around 50
            age_weights = self.age_weight_net(age_norm)
            weights = base_weights.unsqueeze(0) * age_weights
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            weights = base_weights.unsqueeze(0).expand(x.size(0), -1)
        
        return weights


class BiologicalSystemEncoder(nn.Module):
    """
    Hierarchical encoder for biological systems
    Groups biomarkers into systems and creates system-level representations
    """
    
    def __init__(self, 
                 n_biomarkers: int,
                 system_groups: Dict[str, List[int]],
                 embedding_dim: int = 16,
                 dropout_rate: float = 0.2):
        super().__init__()
        self.n_biomarkers = n_biomarkers
        self.system_groups = system_groups
        self.n_systems = len(system_groups)
        
        # Individual biomarker encoders
        self.biomarker_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(embedding_dim, embedding_dim)
            ) for _ in range(n_biomarkers)
        ])
        
        # System-level aggregators with attention
        self.system_attention = nn.ModuleDict({
            system: nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=4,
                dropout=dropout_rate,
                batch_first=True
            ) for system in system_groups.keys()
        })
        
        # System embeddings
        self.system_embeddings = nn.ModuleDict({
            system: nn.Linear(embedding_dim * len(indices), embedding_dim)
            for system, indices in system_groups.items()
        })
        
        # Cross-system interaction layer
        self.cross_system_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Encode biomarkers into hierarchical system representations
        
        Args:
            x: Biomarker values [batch_size, n_biomarkers]
        
        Returns:
            system_features: Aggregated system features [batch_size, n_systems * embedding_dim]
            system_embeddings: Individual system embeddings
        """
        batch_size = x.size(0)
        
        # Encode individual biomarkers
        biomarker_embeddings = []
        for i in range(self.n_biomarkers):
            embedding = self.biomarker_encoders[i](x[:, i:i+1])
            biomarker_embeddings.append(embedding)
        
        # Group into biological systems
        system_embeddings = {}
        system_features_list = []
        
        for system_name, biomarker_indices in self.system_groups.items():
            # Gather biomarkers for this system
            system_biomarkers = torch.stack([
                biomarker_embeddings[idx] for idx in biomarker_indices
            ], dim=1)  # [batch_size, n_biomarkers_in_system, embedding_dim]
            
            # Apply attention within system
            attended_features, _ = self.system_attention[system_name](
                system_biomarkers, system_biomarkers, system_biomarkers
            )
            
            # Aggregate system features
            system_feature = attended_features.reshape(batch_size, -1)
            system_embedding = self.system_embeddings[system_name](system_feature)
            
            system_embeddings[system_name] = system_embedding
            system_features_list.append(system_embedding)
        
        # Stack all system features
        all_systems = torch.stack(system_features_list, dim=1)  # [batch_size, n_systems, embedding_dim]
        
        # Cross-system interactions
        interacted_systems, _ = self.cross_system_attention(
            all_systems, all_systems, all_systems
        )
        
        # Flatten for output
        system_features = interacted_systems.reshape(batch_size, -1)
        
        return system_features, system_embeddings


class HENAWModel(nn.Module):
    """
    HENAW (Hierarchical Elastic Net with Adaptive Weighting) Model
    
    Implements the full HENAW algorithm for biological age prediction:
    BA_HENAW = f(X_blood) = α₀ + Σᵢ₌₁ⁿ αᵢ·h(xᵢ) + Σⱼ₌₁ᵐ βⱼ·g(xᵢ,xⱼ)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Extract configuration
        self.n_biomarkers = config['model']['input_dim']
        self.hidden_dims = config['model']['hidden_dims']
        self.dropout_rate = config['model']['dropout_rate']
        self.system_embedding_dim = config['model']['system_embedding_dim']
        
        # Convert system groups to indices
        self.system_groups = self._create_system_groups(config)
        
        # Core modules
        self.adaptive_weighting = AdaptiveWeightingModule(
            n_biomarkers=self.n_biomarkers,
            lambda_param=config['model']['adaptive_weighting']['lambda_param'],
            temperature=config['model']['adaptive_weighting']['temperature']
        )
        
        self.system_encoder = BiologicalSystemEncoder(
            n_biomarkers=self.n_biomarkers,
            system_groups=self.system_groups,
            embedding_dim=self.system_embedding_dim,
            dropout_rate=self.dropout_rate
        )
        
        # Feature transformation network h(x)
        self.feature_transform = nn.Sequential(
            nn.Linear(self.n_biomarkers, self.hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dims[0]),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dims[1]),
            nn.Dropout(self.dropout_rate)
        )
        
        # Interaction features network g(x_i, x_j)
        n_interactions = len(config['feature_engineering'].get('interactions', []))
        n_ratios = len(config['feature_engineering'].get('ratios', []))
        self.n_engineered_features = n_interactions + n_ratios
        
        if self.n_engineered_features > 0:
            self.interaction_transform = nn.Sequential(
                nn.Linear(self.n_engineered_features, 32),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(32, 16)
            )
        else:
            self.interaction_transform = None
        
        # Calculate total feature dimension
        n_systems = len(self.system_groups)
        system_feature_dim = n_systems * self.system_embedding_dim
        individual_feature_dim = self.hidden_dims[1]
        interaction_feature_dim = 16 if self.n_engineered_features > 0 else 0
        
        total_feature_dim = system_feature_dim + individual_feature_dim + interaction_feature_dim
        
        # Prediction heads
        # Biological age prediction head
        self.age_predictor = nn.Sequential(
            nn.Linear(total_feature_dim, self.hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dims[2], 1)
        )
        
        # Mortality risk prediction head (Cox proportional hazards)
        self.mortality_predictor = nn.Sequential(
            nn.Linear(total_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 1)  # Log hazard ratio
        )
        
        # Morbidity prediction heads (multi-task)
        self.morbidity_predictors = nn.ModuleDict({
            'cardiovascular': self._create_morbidity_head(total_feature_dim),
            'diabetes': self._create_morbidity_head(total_feature_dim),
            'cancer': self._create_morbidity_head(total_feature_dim),
            'dementia': self._create_morbidity_head(total_feature_dim)
        })
        
        # Elastic net parameters
        self.register_buffer('l1_lambda', torch.tensor(config['training']['elastic_net']['lambda_reg']))
        self.register_buffer('elastic_alpha', torch.tensor(config['training']['elastic_net']['alpha']))
        
        # Initialize weights
        self._initialize_weights()
        
    def _create_system_groups(self, config: Dict) -> Dict[str, List[int]]:
        """Convert biomarker names to indices for system grouping"""
        biomarker_names = ['crp', 'hba1c', 'creatinine', 'albumin', 
                          'lymphocyte_pct', 'rdw', 'ggt', 'ast', 'alt']
        name_to_idx = {name: idx for idx, name in enumerate(biomarker_names)}
        
        system_groups = {}
        for system, biomarkers in config['biological_systems'].items():
            indices = [name_to_idx[b] for b in biomarkers if b in name_to_idx]
            if indices:
                system_groups[system] = indices
                
        return system_groups
    
    def _create_morbidity_head(self, input_dim: int) -> nn.Module:
        """Create a morbidity prediction head"""
        return nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def _initialize_weights(self) -> None:
        """Initialize model weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def compute_engineered_features(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Compute interaction and ratio features with safe operations
        
        Args:
            x: Input biomarkers [batch_size, n_biomarkers]
        
        Returns:
            Engineered features or None if no engineering configured
        """
        if self.n_engineered_features == 0:
            return None
        
        if x.size(1) < 9:  # Need at least 9 biomarkers for all features
            logger.warning(f"Not enough biomarkers ({x.size(1)}) for feature engineering")
            return None
        
        features = []
        
        try:
            # Interaction features (from config) with safety checks
            # CRP × HbA1c (indices 0, 1)
            crp_hba1c = x[:, 0] * x[:, 1]
            # Clamp to prevent extreme values
            crp_hba1c = torch.clamp(crp_hba1c, min=-1e6, max=1e6)
            features.append(crp_hba1c)
            
            # Creatinine × Albumin (indices 2, 3)
            creat_alb = x[:, 2] * x[:, 3]
            creat_alb = torch.clamp(creat_alb, min=-1e6, max=1e6)
            features.append(creat_alb)
            
            # AST/ALT ratio (indices 7, 8) with robust safe division
            ast = x[:, 7]
            alt = x[:, 8]
            
            # Use more robust division protection
            # When ALT is close to zero, use default ratio of 1.0
            ast_alt_ratio = torch.where(
                torch.abs(alt) < 1e-6,  # Use larger epsilon for stability
                torch.ones_like(ast),   # Default ratio of 1.0 when ALT is near zero
                ast / alt                # Normal division when ALT is safe
            )
            
            # Clamp ratio to reasonable clinical bounds (0.1 to 10)
            ast_alt_ratio = torch.clamp(ast_alt_ratio, min=0.1, max=10.0)
            features.append(ast_alt_ratio)
            
            # Check for any NaN or Inf values
            result = torch.stack(features, dim=1)
            if torch.isnan(result).any() or torch.isinf(result).any():
                logger.warning("NaN or Inf detected in engineered features, replacing with zeros")
                result = torch.where(torch.isnan(result) | torch.isinf(result), 
                                   torch.zeros_like(result), result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error computing engineered features: {e}")
            # Return zero features as fallback
            return torch.zeros(x.size(0), self.n_engineered_features, device=x.device)
    
    def forward(self, 
                x: torch.Tensor,
                age: Optional[torch.Tensor] = None,
                return_intermediates: bool = False) -> HENAWOutput:
        """
        Forward pass of HENAW model
        
        Args:
            x: Input biomarkers [batch_size, n_biomarkers]
            age: Chronological age for adaptive weighting [batch_size, 1]
            return_intermediates: Whether to return intermediate representations
        
        Returns:
            HENAWOutput with predictions and optional intermediate values
        """
        batch_size = x.size(0)
        
        # Apply adaptive weighting
        weights = self.adaptive_weighting(x, age)
        x_weighted = x * weights
        
        # Transform individual features h(x)
        individual_features = self.feature_transform(x_weighted)
        
        # Encode biological systems
        system_features, system_embeddings = self.system_encoder(x_weighted)
        
        # Compute engineered features g(x_i, x_j)
        engineered = self.compute_engineered_features(x)
        if engineered is not None and self.interaction_transform is not None:
            interaction_features = self.interaction_transform(engineered)
            # Concatenate all features
            combined_features = torch.cat([
                system_features,
                individual_features,
                interaction_features
            ], dim=1)
        else:
            combined_features = torch.cat([
                system_features,
                individual_features
            ], dim=1)
        
        # Predictions
        biological_age = self.age_predictor(combined_features)
        mortality_risk = self.mortality_predictor(combined_features)
        
        # Morbidity predictions
        morbidity_risks = {
            disease: predictor(combined_features)
            for disease, predictor in self.morbidity_predictors.items()
        }
        
        # Create output
        output = HENAWOutput(
            biological_age=biological_age,
            mortality_risk=mortality_risk,
            morbidity_risks=morbidity_risks
        )
        
        if return_intermediates:
            output.system_embeddings = system_features
            output.biomarker_weights = weights
            output.feature_importance = self.compute_feature_importance(x, biological_age)
        
        return output
    
    def compute_feature_importance(self, x: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute feature importance using gradient-based method
        
        Args:
            x: Input features
            predictions: Model predictions
        
        Returns:
            Feature importance scores
        """
        x.requires_grad_(True)
        
        # Backward pass
        grad_outputs = torch.ones_like(predictions)
        gradients = torch.autograd.grad(
            outputs=predictions,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Importance as gradient magnitude
        importance = torch.abs(gradients)
        
        return importance
    
    def elastic_net_loss(self) -> torch.Tensor:
        """
        Compute elastic net regularization with hierarchical constraints
        R_elastic = α·||w||₁ + (1-α)·||w||₂²
        """
        l1_loss = 0.0
        l2_loss = 0.0
        
        # Get all weight parameters
        for name, param in self.named_parameters():
            if 'weight' in name:
                l1_loss += torch.sum(torch.abs(param))
                l2_loss += torch.sum(param ** 2)
        
        # Combine L1 and L2
        elastic_loss = self.elastic_alpha * l1_loss + (1 - self.elastic_alpha) * l2_loss
        
        # Apply hierarchical constraint
        # System-level weights should be bounded by individual weights
        hierarchical_penalty = 0.0
        
        # Add hierarchical constraints between system and individual encoders
        system_weights = []
        individual_weights = []
        
        for name, param in self.system_encoder.named_parameters():
            if 'weight' in name:
                system_weights.append(param.flatten())
        
        for name, param in self.feature_transform.named_parameters():
            if 'weight' in name:
                individual_weights.append(param.flatten())
        
        if system_weights and individual_weights:
            system_norm = torch.norm(torch.cat(system_weights))
            individual_norm = torch.norm(torch.cat(individual_weights))
            
            # Penalize if system norm exceeds individual norm by factor C
            C = self.config['training']['elastic_net']['hierarchical_constraint']
            hierarchical_penalty = F.relu(system_norm - C * individual_norm)
        
        total_regularization = self.l1_lambda * (elastic_loss + hierarchical_penalty)
        
        return total_regularization


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function for HENAW
    L = L_age + λ₁·L_mortality + λ₂·L_morbidity + λ₃·R_elastic
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.loss_weights = config['training']['loss_weights']
        
        # Individual loss functions
        self.age_loss = nn.MSELoss()
        self.mortality_loss = CoxProportionalHazardLoss()
        self.morbidity_loss = nn.BCELoss()
        
    def forward(self,
                output: HENAWOutput,
                targets: Dict[str, torch.Tensor],
                model: HENAWModel) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss
        
        Args:
            output: Model predictions
            targets: Ground truth targets
            model: HENAW model for regularization
        
        Returns:
            total_loss: Combined loss
            loss_components: Individual loss components for logging
        """
        losses = {}
        
        # Age prediction loss
        if 'chronological_age' in targets:
            losses['age'] = self.age_loss(
                output.biological_age.squeeze(),
                targets['chronological_age']
            )
        else:
            losses['age'] = torch.tensor(0.0, device=output.biological_age.device)
        
        # Mortality loss (if survival data available)
        if output.mortality_risk is not None and 'survival_time' in targets:
            losses['mortality'] = self.mortality_loss(
                output.mortality_risk.squeeze(),
                targets['survival_time'],
                targets.get('event_indicator', None)
            )
        else:
            losses['mortality'] = torch.tensor(0.0, device=output.biological_age.device)
        
        # Morbidity losses
        morbidity_total = torch.tensor(0.0, device=output.biological_age.device)
        if output.morbidity_risks is not None:
            for disease, risk in output.morbidity_risks.items():
                if f'{disease}_label' in targets:
                    disease_loss = self.morbidity_loss(
                        risk.squeeze(),
                        targets[f'{disease}_label'].float()
                    )
                    losses[f'morbidity_{disease}'] = disease_loss
                    morbidity_total += disease_loss
        
        losses['morbidity'] = morbidity_total / max(len(output.morbidity_risks), 1)
        
        # Elastic net regularization
        losses['regularization'] = model.elastic_net_loss()
        
        # Combine losses
        total_loss = (
            self.loss_weights['age_prediction'] * losses['age'] +
            self.loss_weights['mortality'] * losses['mortality'] +
            self.loss_weights['morbidity'] * losses['morbidity'] +
            losses['regularization']
        )
        
        return total_loss, losses


class CoxProportionalHazardLoss(nn.Module):
    """Cox proportional hazards loss for survival analysis"""
    
    def forward(self,
                risk_scores: torch.Tensor,
                survival_times: torch.Tensor,
                event_indicators: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute negative partial log-likelihood for Cox model
        
        Args:
            risk_scores: Predicted log hazard ratios [batch_size]
            survival_times: Time to event [batch_size]
            event_indicators: Binary event indicator (1=event, 0=censored) [batch_size]
        
        Returns:
            Cox loss
        """
        if event_indicators is None:
            event_indicators = torch.ones_like(survival_times)
        
        # Sort by survival time
        sorted_times, sorted_indices = torch.sort(survival_times, descending=True)
        sorted_risks = risk_scores[sorted_indices]
        sorted_events = event_indicators[sorted_indices]
        
        # Compute log-sum-exp of risks for at-risk sets
        max_risk = sorted_risks.max()
        log_cumsum_exp = torch.logcumsumexp(sorted_risks - max_risk, dim=0) + max_risk
        
        # Partial log-likelihood
        pl = sorted_risks - log_cumsum_exp
        
        # Only consider actual events (not censored)
        loss = -torch.sum(pl * sorted_events) / (torch.sum(sorted_events) + 1e-8)
        
        return loss


if __name__ == "__main__":
    # Test the model with dummy data
    import yaml
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = HENAWModel(config)
    
    # Create dummy input
    batch_size = 32
    n_biomarkers = 9
    x = torch.randn(batch_size, n_biomarkers)
    age = torch.randn(batch_size, 1) * 20 + 50  # Ages around 50±20
    
    # Forward pass
    output = model(x, age, return_intermediates=True)
    
    print(f"Biological age shape: {output.biological_age.shape}")
    print(f"Mortality risk shape: {output.mortality_risk.shape}")
    print(f"Number of morbidity predictions: {len(output.morbidity_risks)}")
    
    # Test loss computation
    loss_fn = MultiTaskLoss(config)
    targets = {
        'chronological_age': torch.randn(batch_size) * 20 + 50,
        'survival_time': torch.randn(batch_size) * 10,
        'event_indicator': torch.randint(0, 2, (batch_size,)),
        'cardiovascular_label': torch.randint(0, 2, (batch_size,)),
        'diabetes_label': torch.randint(0, 2, (batch_size,))
    }
    
    total_loss, loss_components = loss_fn(output, targets, model)
    print(f"\nTotal loss: {total_loss.item():.4f}")
    for name, value in loss_components.items():
        print(f"{name}: {value.item():.4f}")