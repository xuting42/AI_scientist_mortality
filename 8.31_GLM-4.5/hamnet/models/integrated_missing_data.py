"""
Integrated Missing Data Handler for HAMNet

This module provides a unified interface for all missing data imputation methods
and integrates them seamlessly with the HAMNet architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import numpy as np
from .hamnet import HAMNet, HAMNetConfig
from .gan_imputation import GANImputationConfig, GANImputer
from .graph_imputation import GraphImputationConfig, GraphImputer
from .advanced_imputation import AdvancedImputationConfig, AdvancedImputer


@dataclass
class IntegratedMissingDataConfig:
    """Configuration for integrated missing data handling"""
    # HAMNet configuration
    hamnet_config: HAMNetConfig = None
    
    # Imputation method configurations
    gan_config: GANImputationConfig = None
    graph_config: GraphImputationConfig = None
    advanced_config: AdvancedImputationConfig = None
    
    # Strategy selection
    primary_method: str = "adaptive"  # adaptive, gan, graph, vae, matrix_factorization, temporal, multi_task
    fallback_methods: List[str] = None
    
    # Adaptive strategy parameters
    missing_threshold: float = 0.3  # Threshold for method selection
    confidence_threshold: float = 0.8
    
    # Ensemble parameters
    ensemble_weights: Dict[str, float] = None
    uncertainty_weight: float = 0.2
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    
    def __post_init__(self):
        if self.hamnet_config is None:
            self.hamnet_config = HAMNetConfig()
        if self.gan_config is None:
            self.gan_config = GANImputationConfig()
        if self.graph_config is None:
            self.graph_config = GraphImputationConfig()
        if self.advanced_config is None:
            self.advanced_config = AdvancedImputationConfig()
        if self.fallback_methods is None:
            self.fallback_methods = ["graph", "vae", "multi_task"]
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                "gan": 0.3,
                "graph": 0.3,
                "vae": 0.2,
                "multi_task": 0.2
            }


class MissingDataAnalyzer:
    """Analyze missing data patterns and recommend imputation strategies"""
    
    def __init__(self, config: IntegratedMissingDataConfig):
        self.config = config
        
    def analyze_missing_patterns(self, masks: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Analyze missing data patterns
        
        Args:
            masks: Dictionary of binary masks for each modality
            
        Returns:
            Dictionary with pattern analysis results
        """
        analysis = {}
        
        # Overall missing statistics
        total_values = sum(masks[k].numel() for k in masks)
        total_missing = sum((1 - masks[k]).sum() for k in masks)
        overall_missing_rate = total_missing / total_values
        
        analysis['overall_missing_rate'] = overall_missing_rate
        
        # Modality-specific statistics
        modality_stats = {}
        for modality in masks:
            mask = masks[modality]
            total = mask.numel()
            missing = (1 - mask).sum()
            missing_rate = missing / total
            
            modality_stats[modality] = {
                'total_values': total.item(),
                'missing_values': missing.item(),
                'missing_rate': missing_rate.item()
            }
        
        analysis['modality_stats'] = modality_stats
        
        # Pattern analysis
        pattern_analysis = self._analyze_patterns(masks)
        analysis['pattern_analysis'] = pattern_analysis
        
        # Recommended method
        recommended_method = self._recommend_method(analysis)
        analysis['recommended_method'] = recommended_method
        
        return analysis
    
    def _analyze_patterns(self, masks: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze specific missing data patterns"""
        patterns = {}
        
        # Check for complete modality missingness
        completely_missing = []
        partially_missing = []
        complete_modalities = []
        
        for modality in masks:
            mask = masks[modality]
            missing_rate = (1 - mask).float().mean()
            
            if missing_rate == 1.0:
                completely_missing.append(modality)
            elif missing_rate > 0.0:
                partially_missing.append(modality)
            else:
                complete_modalities.append(modality)
        
        patterns['completely_missing'] = completely_missing
        patterns['partially_missing'] = partially_missing
        patterns['complete_modalities'] = complete_modalities
        
        # Check for structured missingness (e.g., blocks)
        if len(partially_missing) > 0:
            # Analyze correlation between missing patterns
            missing_correlation = self._compute_missing_correlation(masks)
            patterns['missing_correlation'] = missing_correlation
        
        return patterns
    
    def _compute_missing_correlation(self, masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute correlation between missing patterns across modalities"""
        # Flatten masks
        flat_masks = []
        for modality in sorted(masks.keys()):
            flat_masks.append(masks[modality].flatten().unsqueeze(1))
        
        # Stack and compute correlation
        stacked = torch.cat(flat_masks, dim=1)
        correlation = torch.corrcoef(stacked.T)
        
        return correlation
    
    def _recommend_method(self, analysis: Dict[str, Any]) -> str:
        """Recommend imputation method based on analysis"""
        missing_rate = analysis['overall_missing_rate']
        pattern_analysis = analysis['pattern_analysis']
        
        # Rule-based recommendations
        if missing_rate < 0.1:
            return "mean"  # Simple imputation for low missingness
        elif missing_rate < 0.3:
            if len(pattern_analysis['completely_missing']) > 0:
                return "gan"  # GANs good for complete modality generation
            else:
                return "graph"  # Graph-based for partial missingness
        elif missing_rate < 0.5:
            if pattern_analysis.get('missing_correlation', torch.tensor(0.0)).mean() > 0.5:
                return "multi_task"  # Multi-task for correlated missingness
            else:
                return "vae"  # VAE for high missingness with uncertainty
        else:
            return "ensemble"  # Ensemble for very high missingness


class IntegratedImputationModule(nn.Module):
    """Integrated imputation module with HAMNet"""
    
    def __init__(self, config: IntegratedMissingDataConfig):
        super().__init__()
        self.config = config
        
        # Initialize HAMNet
        self.hamnet = HAMNet(config.hamnet_config)
        
        # Initialize imputation methods
        self.imputers = {}
        
        # GAN-based imputer
        if config.gan_config is not None:
            self.imputers['gan'] = GANImputer(config.gan_config)
        
        # Graph-based imputer
        if config.graph_config is not None:
            self.imputers['graph'] = GraphImputer(config.graph_config)
        
        # Advanced imputation methods
        if config.advanced_config is not None:
            self.imputers['vae'] = AdvancedImputer(config.advanced_config, method="vae")
            self.imputers['matrix_factorization'] = AdvancedImputer(
                config.advanced_config, method="matrix_factorization"
            )
            self.imputers['temporal'] = AdvancedImputer(
                config.advanced_config, method="temporal"
            )
            self.imputers['multi_task'] = AdvancedImputer(
                config.advanced_config, method="multi_task"
            )
        
        # Missing data analyzer
        self.analyzer = MissingDataAnalyzer(config)
        
        # Attention-based missing data weighting
        self.missing_attention = nn.MultiheadAttention(
            embed_dim=config.hamnet_config.embedding_dim,
            num_heads=config.hamnet_config.num_heads,
            dropout=config.hamnet_config.dropout,
            batch_first=True
        )
        
        # Uncertainty propagation
        self.uncertainty_propagation = nn.Sequential(
            nn.Linear(config.hamnet_config.embedding_dim, config.hamnet_config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hamnet_config.hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        # Modality dropout for robust training
        self.modality_dropout = config.hamnet_config.dropout
        
    def forward(self, inputs: Dict[str, torch.Tensor],
                masks: Optional[Dict[str, torch.Tensor]] = None,
                training: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with integrated imputation
        
        Args:
            inputs: Dictionary of modality inputs
            masks: Optional dictionary of modality masks
            training: Whether in training mode
            
        Returns:
            Dictionary with predictions and metadata
        """
        if masks is None:
            masks = {}
        
        # Analyze missing patterns
        missing_analysis = self.analyzer.analyze_missing_patterns(masks)
        
        # Select imputation method
        selected_method = self._select_imputation_method(missing_analysis, training)
        
        # Impute missing data
        imputed_data = self._impute_data(inputs, masks, selected_method)
        
        # Apply modality dropout during training
        if training and self.modality_dropout > 0:
            imputed_data = self._apply_modality_dropout(imputed_data)
        
        # Forward through HAMNet
        hamnet_output = self.hamnet(imputed_data, masks)
        
        # Apply missing data attention weighting
        if any(masks.values()):
            hamnet_output = self._apply_missing_attention(hamnet_output, masks)
        
        # Propagate uncertainty
        if self.config.hamnet_config.enable_uncertainty:
            uncertainty = self._propagate_uncertainty(hamnet_output, missing_analysis)
            hamnet_output.update(uncertainty)
        
        # Add metadata
        hamnet_output.update({
            'imputed_data': imputed_data,
            'missing_analysis': missing_analysis,
            'selected_method': selected_method
        })
        
        return hamnet_output
    
    def _select_imputation_method(self, analysis: Dict[str, Any], training: bool) -> str:
        """Select imputation method based on analysis"""
        if self.config.primary_method == "adaptive":
            recommended = analysis['recommended_method']
            
            if recommended in self.imputers:
                return recommended
            else:
                # Use fallback method
                for fallback in self.config.fallback_methods:
                    if fallback in self.imputers:
                        return fallback
        elif self.config.primary_method in self.imputers:
            return self.config.primary_method
        
        # Default to first available method
        return list(self.imputers.keys())[0]
    
    def _impute_data(self, inputs: Dict[str, torch.Tensor],
                    masks: Dict[str, torch.Tensor],
                    method: str) -> Dict[str, torch.Tensor]:
        """Impute missing data using selected method"""
        if method == "ensemble":
            return self._ensemble_imputation(inputs, masks)
        elif method in self.imputers:
            imputer = self.imputers[method]
            return imputer.impute(inputs, masks)
        else:
            # Simple mean imputation as fallback
            return self._mean_imputation(inputs, masks)
    
    def _ensemble_imputation(self, inputs: Dict[str, torch.Tensor],
                           masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Ensemble imputation combining multiple methods"""
        ensemble_results = {}
        uncertainties = {}
        
        # Get predictions from each method
        for method, weight in self.config.ensemble_weights.items():
            if method in self.imputers:
                result = self.imputers[method].impute(inputs, masks)
                
                for modality in result:
                    if modality not in ensemble_results:
                        ensemble_results[modality] = []
                        uncertainties[modality] = []
                    
                    ensemble_results[modality].append(result[modality])
                    
                    # Extract uncertainty if available
                    if f"{modality}_uncertainty" in result:
                        uncertainties[modality].append(result[f"{modality}_uncertainty"])
        
        # Weighted average
        final_results = {}
        for modality in ensemble_results:
            predictions = torch.stack(ensemble_results[modality])
            weights = torch.tensor([self.config.ensemble_weights.get(m, 0.1) 
                                  for m in self.config.ensemble_weights.keys() 
                                  if m in self.imputers], 
                                 device=predictions.device)
            weights = weights / weights.sum()
            
            weighted_pred = (predictions * weights.unsqueeze(-1)).sum(dim=0)
            final_results[modality] = weighted_pred
            
            # Combine uncertainties
            if modality in uncertainties and uncertainties[modality]:
                unc_stack = torch.stack(uncertainties[modality])
                weighted_unc = (unc_stack * weights.unsqueeze(-1)).sum(dim=0)
                final_results[f"{modality}_uncertainty"] = weighted_unc
        
        return final_results
    
    def _mean_imputation(self, inputs: Dict[str, torch.Tensor],
                        masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Simple mean imputation fallback"""
        imputed = {}
        
        for modality in inputs:
            data = inputs[modality]
            if modality in masks:
                mask = masks[modality]
                
                # Compute mean of observed values
                observed_data = data[mask]
                if len(observed_data) > 0:
                    mean_val = observed_data.mean()
                else:
                    mean_val = torch.zeros(1, device=data.device)
                
                # Impute missing values
                imputed_data = data.clone()
                imputed_data[~mask] = mean_val
                imputed[modality] = imputed_data
            else:
                imputed[modality] = data
        
        return imputed
    
    def _apply_modality_dropout(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply modality dropout for robust training"""
        dropped_data = {}
        
        for modality in data:
            if torch.rand(1) < self.modality_dropout:
                # Drop entire modality
                dropped_data[modality] = torch.zeros_like(data[modality])
            else:
                dropped_data[modality] = data[modality]
        
        return dropped_data
    
    def _apply_missing_attention(self, hamnet_output: Dict[str, torch.Tensor],
                               masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply attention-based weighting for missing data"""
        # This would integrate with HAMNet's attention mechanisms
        # For now, we'll adjust the final predictions based on missing data patterns
        if 'predictions' in hamnet_output:
            predictions = hamnet_output['predictions']
            
            # Compute missing data penalty
            missing_penalty = 0.0
            for modality in masks:
                missing_rate = (1 - masks[modality]).float().mean()
                missing_penalty += missing_penalty * 0.1
            
            # Adjust predictions
            adjusted_predictions = predictions * (1 + missing_penalty)
            hamnet_output['predictions'] = adjusted_predictions
        
        return hamnet_output
    
    def _propagate_uncertainty(self, hamnet_output: Dict[str, torch.Tensor],
                             missing_analysis: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Propagate uncertainty from imputation to final predictions"""
        # Extract embedding from HAMNet
        if hasattr(self.hamnet, 'get_embedding'):
            embedding = self.hamnet.get_embedding(hamnet_output)
        else:
            # Use a simplified approach
            embedding = torch.randn(1, self.config.hamnet_config.embedding_dim,
                                  device=next(self.parameters()).device)
        
        # Compute uncertainty based on missing data
        missing_rate = missing_analysis['overall_missing_rate']
        base_uncertainty = self.uncertainty_propagation(embedding)
        
        # Scale by missing rate
        scaled_uncertainty = base_uncertainty * (1 + missing_rate)
        
        return {
            'imputation_uncertainty': scaled_uncertainty,
            'missing_rate': torch.tensor(missing_rate, device=embedding.device)
        }
    
    def train_imputers(self, dataloaders: Dict[str, Any], num_epochs: int = None):
        """Train all imputation methods"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        for method, imputer in self.imputers.items():
            if method in dataloaders:
                print(f"Training {method} imputer...")
                imputer.train(dataloaders[method], num_epochs)
    
    def evaluate_imputation(self, test_data: Dict[str, torch.Tensor],
                          test_masks: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """Evaluate different imputation methods"""
        results = {}
        
        for method, imputer in self.imputers.items():
            print(f"Evaluating {method}...")
            imputed = imputer.impute(test_data, test_masks)
            
            # Compute reconstruction error
            errors = {}
            for modality in test_data:
                if modality in test_masks:
                    mask = test_masks[modality]
                    original = test_data[modality][mask]
                    reconstructed = imputed[modality][mask]
                    
                    mse = F.mse_loss(reconstructed, original).item()
                    mae = F.l1_loss(reconstructed, original).item()
                    
                    errors[modality] = {'mse': mse, 'mae': mae}
            
            results[method] = errors
        
        return results
    
    def save(self, path: str):
        """Save the integrated model"""
        save_dict = {
            'config': self.config,
            'hamnet_state_dict': self.hamnet.state_dict(),
            'imputer_states': {}
        }
        
        for method, imputer in self.imputers.items():
            save_dict['imputer_states'][method] = {
                'state_dict': imputer.model.state_dict(),
                'optimizer_state_dict': imputer.optimizer.state_dict(),
                'training_history': imputer.training_history
            }
        
        torch.save(save_dict, path)
    
    def load(self, path: str):
        """Load the integrated model"""
        checkpoint = torch.load(path, map_location=self.config.device)
        
        self.hamnet.load_state_dict(checkpoint['hamnet_state_dict'])
        
        for method, state in checkpoint['imputer_states'].items():
            if method in self.imputers:
                self.imputers[method].model.load_state_dict(state['state_dict'])
                self.imputers[method].optimizer.load_state_dict(state['optimizer_state_dict'])
                self.imputers[method].training_history = state['training_history']


class IntegratedMissingDataHandler:
    """Main interface for integrated missing data handling"""
    
    def __init__(self, config: IntegratedMissingDataConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize integrated module
        self.module = IntegratedImputationModule(config).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.module.parameters(),
            lr=config.learning_rate,
            weight_decay=config.hamnet_config.weight_decay
        )
        
    def train(self, dataloaders: Dict[str, Any], num_epochs: int = None):
        """Train the integrated system"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        # Train imputers first
        print("Training imputation methods...")
        self.module.train_imputers(dataloaders, num_epochs // 2)
        
        # Train integrated system
        print("Training integrated system...")
        self.module.train()
        
        for epoch in range(num_epochs // 2):
            epoch_loss = 0.0
            
            for batch in dataloaders.get('integrated', []):
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.module(
                    batch['data'], batch['masks'], training=True
                )
                
                # Compute loss
                loss = self._compute_integrated_loss(outputs, batch)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs//2}, Loss: {epoch_loss:.4f}")
    
    def _compute_integrated_loss(self, outputs: Dict[str, torch.Tensor],
                                batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute integrated training loss"""
        # Main prediction loss
        if 'predictions' in outputs and 'targets' in batch:
            pred_loss = F.mse_loss(outputs['predictions'], batch['targets'])
        else:
            pred_loss = torch.tensor(0.0, device=self.device)
        
        # Imputation quality loss
        if 'imputed_data' in outputs and 'data' in batch:
            imp_loss = 0.0
            for modality in outputs['imputed_data']:
                if modality in batch['data'] and modality in batch['masks']:
                    mask = batch['masks'][modality]
                    original = batch['data'][modality]
                    imputed = outputs['imputed_data'][modality]
                    
                    imp_loss += F.mse_loss(imputed * mask, original * mask)
            
            imp_loss = imp_loss / len(outputs['imputed_data'])
        else:
            imp_loss = torch.tensor(0.0, device=self.device)
        
        # Uncertainty calibration loss
        if 'imputation_uncertainty' in outputs:
            unc_loss = outputs['imputation_uncertainty'].mean()
        else:
            unc_loss = torch.tensor(0.0, device=self.device)
        
        return pred_loss + 0.5 * imp_loss + 0.1 * unc_loss
    
    def predict(self, data: Dict[str, torch.Tensor],
               masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make predictions with integrated imputation"""
        self.module.eval()
        
        with torch.no_grad():
            outputs = self.module(data, masks, training=False)
        
        return outputs
    
    def analyze_missing_data(self, masks: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        return self.module.analyzer.analyze_missing_patterns(masks)
    
    def save(self, path: str):
        """Save the handler"""
        self.module.save(path)
    
    def load(self, path: str):
        """Load the handler"""
        self.module.load(path)