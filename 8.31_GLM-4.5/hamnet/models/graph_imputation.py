"""
Graph-based Missing Data Imputation for HAMNet

This module implements graph neural networks for missing data imputation using
patient similarity graphs, community detection, and attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import numpy as np
# Note: PyTorch Geometric dependencies are optional
try:
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    GATConv = None

# Note: sklearn dependencies are optional
try:
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    SpectralClustering = None
    cosine_similarity = None


@dataclass
class GraphImputationConfig:
    """Configuration for graph-based imputation"""
    # Model architecture
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Graph construction
    similarity_threshold: float = 0.7
    k_neighbors: int = 10
    graph_type: str = "knn"  # knn, epsilon, fully_connected
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 100
    
    # Community detection
    num_communities: int = 5
    community_weight: float = 0.5
    
    # Data dimensions
    clinical_dim: int = 100
    imaging_dim: int = 512
    genetic_dim: int = 1000
    lifestyle_dim: int = 50
    
    # Modality settings
    modalities: List[str] = None
    modality_dims: Dict[str, int] = None
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    
    def __post_init__(self):
        if self.modalities is None:
            self.modalities = ["clinical", "imaging", "genetic", "lifestyle"]
        if self.modality_dims is None:
            self.modality_dims = {
                "clinical": self.clinical_dim,
                "imaging": self.imaging_dim,
                "genetic": self.genetic_dim,
                "lifestyle": self.lifestyle_dim
            }


class PatientSimilarityGraph:
    """Construct patient similarity graphs from multi-modal data"""
    
    def __init__(self, config: GraphImputationConfig):
        self.config = config
        self.adjacency_matrix = None
        self.node_features = None
        self.communities = None
        
    def construct_graph(self, data: Dict[str, torch.Tensor],
                       masks: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct patient similarity graph
        
        Args:
            data: Dictionary of modality data
            masks: Dictionary of binary masks
            
        Returns:
            Tuple of (edge_index, edge_weight)
        """
        # Combine all modalities into feature matrix
        features = self._combine_modalities(data, masks)
        
        # Compute similarity matrix
        similarity_matrix = self._compute_similarity(features)
        
        # Construct graph based on specified type
        if self.config.graph_type == "knn":
            edge_index, edge_weight = self._construct_knn_graph(similarity_matrix)
        elif self.config.graph_type == "epsilon":
            edge_index, edge_weight = self._construct_epsilon_graph(similarity_matrix)
        else:  # fully_connected
            edge_index, edge_weight = self._construct_fully_connected_graph(similarity_matrix)
        
        # Detect communities
        self.communities = self._detect_communities(similarity_matrix)
        
        self.adjacency_matrix = similarity_matrix
        self.node_features = features
        
        return edge_index, edge_weight
    
    def _combine_modalities(self, data: Dict[str, torch.Tensor],
                          masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine multi-modal data into feature matrix"""
        features_list = []
        
        for modality in self.config.modalities:
            if modality in data:
                mod_data = data[modality]
                if modality in masks:
                    mask = masks[modality].unsqueeze(-1)
                    mod_data = mod_data * mask  # Zero out missing values
                features_list.append(mod_data)
        
        return torch.cat(features_list, dim=-1)
    
    def _compute_similarity(self, features: torch.Tensor) -> torch.Tensor:
        """Compute pairwise similarity between patients"""
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.mm(features_norm, features_norm.T)
        
        # Apply threshold
        similarity = torch.clamp(similarity, 0, 1)
        
        return similarity
    
    def _construct_knn_graph(self, similarity_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct k-nearest neighbor graph"""
        num_nodes = similarity_matrix.shape[0]
        k = min(self.config.k_neighbors, num_nodes - 1)
        
        # Get top-k neighbors for each node
        top_k_values, top_k_indices = torch.topk(similarity_matrix, k + 1, dim=-1)
        
        # Remove self-loops
        mask = top_k_indices != torch.arange(num_nodes, device=similarity_matrix.device).unsqueeze(-1)
        top_k_indices = top_k_indices[mask].view(num_nodes, k)
        top_k_values = top_k_values[mask].view(num_nodes, k)
        
        # Create edge index and weights
        edge_indices = []
        edge_weights = []
        
        for i in range(num_nodes):
            for j in range(k):
                neighbor = top_k_indices[i, j]
                weight = top_k_values[i, j]
                
                # Add edge in both directions
                edge_indices.extend([(i, neighbor), (neighbor, i)])
                edge_weights.extend([weight, weight])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).T
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        
        return edge_index, edge_weight
    
    def _construct_epsilon_graph(self, similarity_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct epsilon-neighborhood graph"""
        threshold = self.config.similarity_threshold
        
        # Create edges for similarities above threshold
        edge_indices = torch.where(similarity_matrix > threshold)
        edge_weights = similarity_matrix[edge_indices]
        
        # Remove self-loops
        mask = edge_indices[0] != edge_indices[1]
        edge_indices = edge_indices[:, mask]
        edge_weights = edge_weights[mask]
        
        return edge_indices, edge_weights
    
    def _construct_fully_connected_graph(self, similarity_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct fully connected graph"""
        num_nodes = similarity_matrix.shape[0]
        
        # Create all possible edges (excluding self-loops)
        edge_indices = []
        edge_weights = []
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                weight = similarity_matrix[i, j]
                edge_indices.extend([(i, j), (j, i)])
                edge_weights.extend([weight, weight])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).T
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        
        return edge_index, edge_weight
    
    def _detect_communities(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        """Detect communities using spectral clustering"""
        if not SKLEARN_AVAILABLE:
            # Fallback: simple k-means-like clustering using PyTorch
            return self._simple_clustering(similarity_matrix)
        
        # Convert to numpy for sklearn
        sim_np = similarity_matrix.cpu().numpy()
        
        # Apply spectral clustering
        clustering = SpectralClustering(
            n_clusters=self.config.num_communities,
            affinity='precomputed',
            random_state=42
        )
        
        communities = clustering.fit_predict(sim_np)
        
        return torch.tensor(communities, dtype=torch.long, device=similarity_matrix.device)
    
    def _simple_clustering(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        """Simple clustering implementation using PyTorch"""
        num_nodes = similarity_matrix.shape[0]
        num_clusters = min(self.config.num_communities, num_nodes)
        
        # Initialize random cluster centers
        centers = similarity_matrix[torch.randperm(num_nodes)[:num_clusters]]
        
        # Simple k-means-like algorithm
        communities = torch.zeros(num_nodes, dtype=torch.long, device=similarity_matrix.device)
        
        for _ in range(10):  # Max iterations
            # Assign each node to nearest cluster center
            for i in range(num_nodes):
                similarities = torch.cosine_similarity(
                    similarity_matrix[i:i+1], centers, dim=1
                )
                communities[i] = torch.argmax(similarities)
            
            # Update cluster centers
            new_centers = []
            for c in range(num_clusters):
                cluster_mask = communities == c
                if cluster_mask.any():
                    cluster_sims = similarity_matrix[cluster_mask]
                    new_center = cluster_sims.mean(dim=0)
                else:
                    new_center = centers[c]
                new_centers.append(new_center)
            
            centers = torch.stack(new_centers)
        
        return communities


class GraphAttentionLayer(nn.Module):
    """Graph attention layer for missing data imputation"""
    
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 4,
                 dropout: float = 0.1, attention_dropout: float = 0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.output_dim = output_dim
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            # Fallback to standard attention if PyTorch Geometric is not available
            self.use_fallback = True
            self.fallback_attention = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads,
                dropout=attention_dropout,
                batch_first=True
            )
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            # Use GATConv if PyTorch Geometric is available
            self.use_fallback = False
            self.attention = nn.ModuleList([
                GATConv(input_dim, output_dim // num_heads, heads=1,
                       dropout=attention_dropout, concat=False)
                for _ in range(num_heads)
            ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(output_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for graph attention layer
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_weight: Optional edge weights
            
        Returns:
            Updated node features
        """
        if self.use_fallback:
            # Fallback attention mechanism
            # Create adjacency matrix from edge_index
            num_nodes = x.shape[0]
            adj_matrix = torch.zeros(num_nodes, num_nodes, device=x.device)
            adj_matrix[edge_index[0], edge_index[1]] = 1
            
            # Apply self-attention
            attended, _ = self.fallback_attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
            attended = attended.squeeze(1)
            
            # Apply edge weighting if provided
            if edge_weight is not None:
                # Simple aggregation of neighbor information
                neighbor_sum = torch.zeros_like(x)
                for i, (src, dst) in enumerate(edge_index.T):
                    neighbor_sum[dst] += x[src] * edge_weight[i]
                attended = 0.5 * attended + 0.5 * neighbor_sum
            
            output = self.projection(attended)
        else:
            # Apply multi-head attention
            head_outputs = []
            for attention_head in self.attention:
                head_output = attention_head(x, edge_index, edge_weight)
                head_outputs.append(head_output)
            
            # Concatenate heads
            output = torch.cat(head_outputs, dim=-1)
        
        # Apply dropout and batch norm
        output = self.dropout(output)
        output = self.batch_norm(output)
        
        return output


class CommunityAwareGNN(nn.Module):
    """Community-aware graph neural network for imputation"""
    
    def __init__(self, config: GraphImputationConfig):
        super().__init__()
        self.config = config
        
        # Calculate input dimension
        total_dim = sum(config.modality_dims.values())
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        
        # First layer
        self.conv_layers.append(
            GraphAttentionLayer(
                total_dim, config.hidden_dim,
                config.num_heads, config.dropout,
                config.attention_dropout
            )
        )
        
        # Hidden layers
        for _ in range(config.num_layers - 2):
            self.conv_layers.append(
                GraphAttentionLayer(
                    config.hidden_dim, config.hidden_dim,
                    config.num_heads, config.dropout,
                    config.attention_dropout
                )
            )
        
        # Output layer
        self.conv_layers.append(
            GraphAttentionLayer(
                config.hidden_dim, total_dim,
                config.num_heads, config.dropout,
                config.attention_dropout
            )
        )
        
        # Community embeddings
        self.community_embeddings = nn.Embedding(
            config.num_communities, config.hidden_dim
        )
        
        # Community attention
        self.community_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads // 2,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # Modality-specific predictors
        self.modality_predictors = nn.ModuleDict()
        for modality in config.modalities:
            self.modality_predictors[modality] = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, config.modality_dims[modality])
            )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None,
                communities: Optional[torch.Tensor] = None,
                missing_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for community-aware GNN
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_weight: Optional edge weights
            communities: Community assignments
            missing_mask: Missing data mask
            
        Returns:
            Dictionary of imputed values for each modality
        """
        # Store original features for residual connection
        original_x = x.clone()
        
        # Apply graph convolution layers
        for i, conv_layer in enumerate(self.conv_layers):
            residual = x if i > 0 else None
            x = conv_layer(x, edge_index, edge_weight)
            
            # Add residual connection
            if residual is not None:
                x = x + residual
            
            # Apply community-aware attention
            if communities is not None and i < len(self.conv_layers) - 1:
                x = self._apply_community_attention(x, communities)
        
        # Split features back into modalities
        imputed_data = self._split_modalities(x, original_x, missing_mask)
        
        return imputed_data
    
    def _apply_community_attention(self, x: torch.Tensor,
                                 communities: torch.Tensor) -> torch.Tensor:
        """Apply community-aware attention mechanism"""
        # Get community embeddings
        community_emb = self.community_embeddings(communities)
        
        # Apply community attention
        attended_x, _ = self.community_attention(x.unsqueeze(1), 
                                               community_emb.unsqueeze(1),
                                               community_emb.unsqueeze(1))
        
        # Weighted combination
        alpha = self.config.community_weight
        output = alpha * attended_x.squeeze(1) + (1 - alpha) * x
        
        return output
    
    def _split_modalities(self, x: torch.Tensor, original_x: torch.Tensor,
                         missing_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Split combined features back into modalities"""
        imputed_data = {}
        
        # Calculate split indices
        split_indices = []
        current_idx = 0
        for modality in self.config.modalities:
            dim = self.config.modality_dims[modality]
            split_indices.append((current_idx, current_idx + dim))
            current_idx += dim
        
        # Split features
        for i, modality in enumerate(self.config.modalities):
            start_idx, end_idx = split_indices[i]
            mod_features = x[:, start_idx:end_idx]
            
            # Apply modality-specific prediction
            if missing_mask is not None and modality in missing_mask:
                # Only predict missing values
                mod_missing = missing_mask[modality]
                if mod_missing.any():
                    # Get features for missing values
                    missing_features = x[mod_missing]
                    
                    # Predict missing values
                    predicted = self.modality_predictors[modality](missing_features)
                    
                    # Create output tensor
                    imputed = original_x[:, start_idx:end_idx].clone()
                    imputed[mod_missing] = predicted
                    imputed_data[modality] = imputed
                else:
                    # No missing values, use original
                    imputed_data[modality] = original_x[:, start_idx:end_idx]
            else:
                # Predict all values
                predicted = self.modality_predictors[modality](mod_features)
                imputed_data[modality] = predicted
        
        return imputed_data


class GraphImputer:
    """Main graph-based imputation class"""
    
    def __init__(self, config: GraphImputationConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize models
        self.graph_constructor = PatientSimilarityGraph(config)
        self.gnn = CommunityAwareGNN(config).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.gnn.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Training history
        self.training_history = {
            'reconstruction_loss': [],
            'graph_loss': [],
            'community_loss': []
        }
        
    def train_step(self, data: Dict[str, torch.Tensor],
                   masks: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        batch_size = next(iter(data.values())).shape[0]
        
        # Move to device
        data = {k: v.to(self.device) for k, v in data.items()}
        masks = {k: v.to(self.device) for k, v in masks.items()}
        
        # Construct graph
        edge_index, edge_weight = self.graph_constructor.construct_graph(data, masks)
        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device)
        
        # Get communities
        communities = self.graph_constructor.communities
        
        # Combine modalities
        combined_features = self.graph_constructor._combine_modalities(data, masks)
        
        # Forward pass
        self.optimizer.zero_grad()
        
        imputed_data = self.gnn(
            combined_features, edge_index, edge_weight,
            communities, masks
        )
        
        # Compute reconstruction loss
        reconstruction_loss = 0.0
        for modality in self.config.modalities:
            if modality in data:
                # Only compute loss for missing values
                if modality in masks:
                    missing_mask = masks[modality]
                    if missing_mask.any():
                        original_missing = data[modality][missing_mask]
                        imputed_missing = imputed_data[modality][missing_mask]
                        reconstruction_loss += F.mse_loss(imputed_missing, original_missing)
        
        # Compute graph regularization loss
        graph_loss = self._compute_graph_loss(combined_features, edge_index, edge_weight)
        
        # Compute community loss
        community_loss = self._compute_community_loss(combined_features, communities)
        
        # Total loss
        total_loss = reconstruction_loss + 0.1 * graph_loss + 0.1 * community_loss
        
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'reconstruction_loss': reconstruction_loss.item(),
            'graph_loss': graph_loss.item(),
            'community_loss': community_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def _compute_graph_loss(self, x: torch.Tensor, edge_index: torch.Tensor,
                           edge_weight: torch.Tensor) -> torch.Tensor:
        """Compute graph regularization loss"""
        # Smoothness regularization: encourage connected nodes to have similar features
        row, col = edge_index
        diff = x[row] - x[col]
        smoothness_loss = (edge_weight.unsqueeze(-1) * diff ** 2).sum()
        
        return smoothness_loss
    
    def _compute_community_loss(self, x: torch.Tensor,
                               communities: torch.Tensor) -> torch.Tensor:
        """Compute community regularization loss"""
        # Encourage nodes in same community to have similar features
        community_loss = 0.0
        
        for comm_id in torch.unique(communities):
            comm_mask = communities == comm_id
            if comm_mask.sum() > 1:
                comm_features = x[comm_mask]
                comm_mean = comm_features.mean(dim=0)
                comm_loss = F.mse_loss(comm_features, comm_mean.unsqueeze(0))
                community_loss += comm_loss
        
        return community_loss
    
    def impute(self, data: Dict[str, torch.Tensor],
               masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Impute missing data"""
        self.gnn.eval()
        
        with torch.no_grad():
            # Construct graph
            edge_index, edge_weight = self.graph_constructor.construct_graph(data, masks)
            edge_index = edge_index.to(self.device)
            edge_weight = edge_weight.to(self.device)
            
            # Get communities
            communities = self.graph_constructor.communities
            
            # Combine modalities
            combined_features = self.graph_constructor._combine_modalities(data, masks)
            
            # Impute
            imputed_data = self.gnn(
                combined_features, edge_index, edge_weight,
                communities, masks
            )
        
        return imputed_data
    
    def train(self, dataloader, num_epochs: int = None):
        """Train the graph imputer"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        self.gnn.train()
        
        for epoch in range(num_epochs):
            epoch_losses = {
                'reconstruction_loss': [],
                'graph_loss': [],
                'community_loss': [],
                'total_loss': []
            }
            
            for batch_idx, (data, masks) in enumerate(dataloader):
                losses = self.train_step(data, masks)
                
                for key, value in losses.items():
                    epoch_losses[key].append(value)
            
            # Log epoch averages
            epoch_avg = {k: np.mean(v) for k, v in epoch_losses.items()}
            self.training_history['reconstruction_loss'].append(epoch_avg['reconstruction_loss'])
            self.training_history['graph_loss'].append(epoch_avg['graph_loss'])
            self.training_history['community_loss'].append(epoch_avg['community_loss'])
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Reconstruction Loss: {epoch_avg['reconstruction_loss']:.4f}")
                print(f"  Graph Loss: {epoch_avg['graph_loss']:.4f}")
                print(f"  Community Loss: {epoch_avg['community_loss']:.4f}")
                print(f"  Total Loss: {epoch_avg['total_loss']:.4f}")
    
    def save(self, path: str):
        """Save the model"""
        torch.save({
            'gnn_state_dict': self.gnn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, path)
    
    def load(self, path: str):
        """Load the model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.gnn.load_state_dict(checkpoint['gnn_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']