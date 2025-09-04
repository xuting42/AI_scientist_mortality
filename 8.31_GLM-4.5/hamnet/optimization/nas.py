"""
Neural Architecture Search (NAS) Framework for HAMNet Optimization

This module provides comprehensive neural architecture search capabilities
for optimizing the HAMNet architecture with various search strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from enum import Enum
import random
import math
from collections import defaultdict
import time

from ..models.hamnet import HAMNet, HAMNetConfig
from ..utils.metrics import ModelMetrics
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SearchStrategy(Enum):
    """Supported NAS search strategies."""
    RANDOM = "random"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT = "reinforcement"
    DIFFERENTIABLE = "differentiable"


class OperationType(Enum):
    """Supported operation types for architecture search."""
    CONV1D = "conv1d"
    CONV2D = "conv2d"
    DEPTHWISE_CONV = "depthwise_conv"
    SEPARABLE_CONV = "separable_conv"
    MAX_POOL = "max_pool"
    AVG_POOL = "avg_pool"
    IDENTITY = "identity"
    ZERO = "zero"
    ATTENTION = "attention"
    GRU = "gru"
    LSTM = "lstm"
    TRANSFORMER = "transformer"


@dataclass
class ArchitectureConfig:
    """Configuration for neural architecture search."""
    search_strategy: SearchStrategy = SearchStrategy.RANDOM
    max_layers: int = 20
    min_layers: int = 5
    max_hidden_size: int = 1024
    min_hidden_size: int = 64
    max_attention_heads: int = 16
    min_attention_heads: int = 2
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    tournament_size: int = 3
    early_stop_patience: int = 20
    max_trials: int = 1000
    validation_split: float = 0.2
    search_space: List[OperationType] = field(default_factory=lambda: [
        OperationType.CONV1D, OperationType.CONV2D, OperationType.ATTENTION,
        OperationType.GRU, OperationType.TRANSFORMER, OperationType.IDENTITY
    ])


@dataclass
class ArchitectureGene:
    """Represents an architecture gene in evolutionary search."""
    layer_type: OperationType
    hidden_size: int
    num_heads: Optional[int] = None
    dropout: float = 0.1
    activation: str = "gelu"
    normalization: str = "layer_norm"
    skip_connection: bool = True
    
    def mutate(self, config: ArchitectureConfig) -> 'ArchitectureGene':
        """Mutate the gene."""
        mutated = ArchitectureGene(**self.__dict__)
        
        # Mutate layer type
        if random.random() < config.mutation_rate:
            mutated.layer_type = random.choice(config.search_space)
        
        # Mutate hidden size
        if random.random() < config.mutation_rate:
            mutated.hidden_size = random.randint(
                config.min_hidden_size, config.max_hidden_size
            )
        
        # Mutate attention heads if applicable
        if mutated.layer_type in [OperationType.ATTENTION, OperationType.TRANSFORMER]:
            if random.random() < config.mutation_rate:
                mutated.num_heads = random.randint(
                    config.min_attention_heads, config.max_attention_heads
                )
        
        # Mutate dropout
        if random.random() < config.mutation_rate:
            mutated.dropout = random.uniform(0.0, 0.5)
        
        # Mutate activation
        if random.random() < config.mutation_rate:
            mutated.activation = random.choice(["gelu", "relu", "swish", "tanh"])
        
        # Mutate normalization
        if random.random() < config.mutation_rate:
            mutated.normalization = random.choice(["layer_norm", "batch_norm", "instance_norm"])
        
        # Mutate skip connection
        if random.random() < config.mutation_rate:
            mutated.skip_connection = random.choice([True, False])
        
        return mutated


@dataclass
class ArchitectureIndividual:
    """Represents an individual architecture in evolutionary search."""
    genes: List[ArchitectureGene]
    fitness: float = 0.0
    complexity: float = 0.0
    accuracy: float = 0.0
    parameters: int = 0
    flops: float = 0.0
    
    def crossover(self, other: 'ArchitectureIndividual', config: ArchitectureConfig) -> 'ArchitectureIndividual':
        """Perform crossover with another individual."""
        if random.random() > config.crossover_rate:
            return self
        
        # Single-point crossover
        crossover_point = random.randint(1, len(self.genes) - 1)
        
        child_genes = (
            self.genes[:crossover_point] + 
            other.genes[crossover_point:]
        )
        
        return ArchitectureIndividual(genes=child_genes)
    
    def mutate(self, config: ArchitectureConfig) -> 'ArchitectureIndividual':
        """Mutate the individual."""
        mutated_genes = []
        
        for gene in self.genes:
            if random.random() < config.mutation_rate:
                mutated_genes.append(gene.mutate(config))
            else:
                mutated_genes.append(gene)
        
        return ArchitectureIndividual(genes=mutated_genes)
    
    def to_model_config(self) -> HAMNetConfig:
        """Convert to HAMNet configuration."""
        # Convert genes to HAMNet layers
        layers = []
        
        for i, gene in enumerate(self.genes):
            layer_config = {
                "type": gene.layer_type.value,
                "hidden_size": gene.hidden_size,
                "dropout": gene.dropout,
                "activation": gene.activation,
                "normalization": gene.normalization,
                "skip_connection": gene.skip_connection
            }
            
            if gene.num_heads is not None:
                layer_config["num_heads"] = gene.num_heads
            
            layers.append(layer_config)
        
        return HAMNetConfig(
            input_size=512,  # Will be updated based on actual input
            hidden_size=256,  # Default, will be overridden by genes
            num_layers=len(self.genes),
            layers=layers,
            dropout_rate=0.1
        )


class SearchSpace:
    """Manages the architecture search space."""
    
    def __init__(self, config: ArchitectureConfig):
        self.config = config
        self.operation_types = config.search_space
        
    def random_architecture(self) -> ArchitectureIndividual:
        """Generate a random architecture."""
        num_layers = random.randint(self.config.min_layers, self.config.max_layers)
        genes = []
        
        for _ in range(num_layers):
            gene = ArchitectureGene(
                layer_type=random.choice(self.operation_types),
                hidden_size=random.randint(
                    self.config.min_hidden_size, self.config.max_hidden_size
                ),
                num_heads=random.randint(
                    self.config.min_attention_heads, self.config.max_attention_heads
                ) if random.choice([True, False]) else None,
                dropout=random.uniform(0.0, 0.5),
                activation=random.choice(["gelu", "relu", "swish", "tanh"]),
                normalization=random.choice(["layer_norm", "batch_norm", "instance_norm"]),
                skip_connection=random.choice([True, False])
            )
            genes.append(gene)
        
        return ArchitectureIndividual(genes=genes)
    
    def generate_population(self, size: int) -> List[ArchitectureIndividual]:
        """Generate initial population."""
        return [self.random_architecture() for _ in range(size)]


class BaseSearchStrategy(ABC):
    """Base class for search strategies."""
    
    def __init__(self, config: ArchitectureConfig, search_space: SearchSpace):
        self.config = config
        self.search_space = search_space
        self.best_architecture = None
        self.best_fitness = float('-inf')
        self.search_history = []
        
    @abstractmethod
    def search(self, fitness_fn, max_iterations: int = None) -> ArchitectureIndividual:
        """Perform architecture search."""
        pass
    
    def evaluate_architecture(self, architecture: ArchitectureIndividual, fitness_fn) -> float:
        """Evaluate a single architecture."""
        try:
            fitness = fitness_fn(architecture)
            self.search_history.append({
                "architecture": architecture,
                "fitness": fitness,
                "timestamp": time.time()
            })
            
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_architecture = architecture
            
            return fitness
        except Exception as e:
            logger.error(f"Error evaluating architecture: {e}")
            return float('-inf')


class RandomSearch(BaseSearchStrategy):
    """Random search strategy."""
    
    def search(self, fitness_fn, max_iterations: int = None) -> ArchitectureIndividual:
        """Perform random search."""
        max_trials = max_iterations or self.config.max_trials
        
        logger.info(f"Starting random search with {max_trials} trials")
        
        for trial in range(max_trials):
            architecture = self.search_space.random_architecture()
            fitness = self.evaluate_architecture(architecture, fitness_fn)
            
            if trial % 50 == 0:
                logger.info(f"Trial {trial}: Best fitness = {self.best_fitness:.4f}")
        
        logger.info(f"Random search completed. Best fitness: {self.best_fitness:.4f}")
        return self.best_architecture


class EvolutionarySearch(BaseSearchStrategy):
    """Evolutionary search strategy."""
    
    def __init__(self, config: ArchitectureConfig, search_space: SearchSpace):
        super().__init__(config, search_space)
        self.population = []
        
    def tournament_selection(self, fitnesses: List[float]) -> int:
        """Select individual using tournament selection."""
        tournament = random.sample(range(len(fitnesses)), self.config.tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament]
        return tournament[tournament_fitnesses.index(max(tournament_fitnesses))]
    
    def search(self, fitness_fn, max_iterations: int = None) -> ArchitectureIndividual:
        """Perform evolutionary search."""
        generations = max_iterations or self.config.generations
        
        # Initialize population
        self.population = self.search_space.generate_population(self.config.population_size)
        
        logger.info(f"Starting evolutionary search with {generations} generations")
        
        best_fitness = float('-inf')
        patience_counter = 0
        
        for generation in range(generations):
            # Evaluate population
            fitnesses = []
            for individual in self.population:
                fitness = self.evaluate_architecture(individual, fitness_fn)
                fitnesses.append(fitness)
            
            # Track best
            current_best = max(fitnesses)
            if current_best > best_fitness:
                best_fitness = current_best
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.early_stop_patience:
                logger.info(f"Early stopping at generation {generation}")
                break
            
            # Create new generation
            new_population = []
            
            # Elitism: keep best individual
            best_idx = fitnesses.index(max(fitnesses))
            new_population.append(self.population[best_idx])
            
            # Generate offspring
            while len(new_population) < self.config.population_size:
                # Selection
                parent1_idx = self.tournament_selection(fitnesses)
                parent2_idx = self.tournament_selection(fitnesses)
                
                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]
                
                # Crossover and mutation
                child = parent1.crossover(parent2, self.config)
                child = child.mutate(self.config)
                
                new_population.append(child)
            
            self.population = new_population
            
            if generation % 10 == 0:
                logger.info(f"Generation {generation}: Best fitness = {best_fitness:.4f}")
        
        logger.info(f"Evolutionary search completed. Best fitness: {best_fitness:.4f}")
        return self.best_architecture


class BayesianOptimizationSearch(BaseSearchStrategy):
    """Bayesian optimization search strategy."""
    
    def __init__(self, config: ArchitectureConfig, search_space: SearchSpace):
        super().__init__(config, search_space)
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            self.gp_minimize = gp_minimize
            self.skopt_available = True
        except ImportError:
            logger.warning("scikit-optimize not available. Using random search instead.")
            self.skopt_available = False
    
    def search(self, fitness_fn, max_iterations: int = None) -> ArchitectureIndividual:
        """Perform Bayesian optimization search."""
        if not self.skopt_available:
            logger.warning("Falling back to random search")
            random_search = RandomSearch(self.config, self.search_space)
            return random_search.search(fitness_fn, max_iterations)
        
        n_calls = max_iterations or self.config.max_trials
        
        logger.info(f"Starting Bayesian optimization with {n_calls} calls")
        
        # Define search space
        space = [
            Integer(self.config.min_layers, self.config.max_layers, name='num_layers'),
            Integer(self.config.min_hidden_size, self.config.max_hidden_size, name='hidden_size'),
            Real(0.0, 0.5, name='dropout'),
            Categorical(['gelu', 'relu', 'swish', 'tanh'], name='activation'),
            Categorical(['layer_norm', 'batch_norm', 'instance_norm'], name='normalization')
        ]
        
        def objective(params):
            num_layers, hidden_size, dropout, activation, normalization = params
            
            # Create architecture from parameters
            genes = []
            for i in range(num_layers):
                gene = ArchitectureGene(
                    layer_type=random.choice(self.operation_types),
                    hidden_size=hidden_size,
                    dropout=dropout,
                    activation=activation,
                    normalization=normalization,
                    skip_connection=random.choice([True, False])
                )
                genes.append(gene)
            
            architecture = ArchitectureIndividual(genes=genes)
            
            # Evaluate architecture (negative because skopt minimizes)
            fitness = self.evaluate_architecture(architecture, fitness_fn)
            return -fitness
        
        # Run optimization
        result = self.gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            n_initial_points=20,
            random_state=42
        )
        
        logger.info(f"Bayesian optimization completed. Best fitness: {-result.fun:.4f}")
        return self.best_architecture


class HAMNetNAS:
    """HAMNet Neural Architecture Search framework."""
    
    def __init__(self, config: ArchitectureConfig):
        self.config = config
        self.search_space = SearchSpace(config)
        self.search_strategy = self._create_search_strategy()
        self.search_results = []
        
    def _create_search_strategy(self) -> BaseSearchStrategy:
        """Create search strategy based on configuration."""
        if self.config.search_strategy == SearchStrategy.RANDOM:
            return RandomSearch(self.config, self.search_space)
        elif self.config.search_strategy == SearchStrategy.EVOLUTIONARY:
            return EvolutionarySearch(self.config, self.search_space)
        elif self.config.search_strategy == SearchStrategy.BAYESIAN:
            return BayesianOptimizationSearch(self.config, self.search_space)
        else:
            raise ValueError(f"Unsupported search strategy: {self.config.search_strategy}")
    
    def search(self, fitness_fn, max_iterations: int = None) -> ArchitectureIndividual:
        """Perform neural architecture search."""
        logger.info(f"Starting NAS with {self.config.search_strategy.value} strategy")
        
        best_architecture = self.search_strategy.search(fitness_fn, max_iterations)
        
        # Store results
        self.search_results = self.search_strategy.search_history
        
        return best_architecture
    
    def get_top_architectures(self, n: int = 10) -> List[ArchitectureIndividual]:
        """Get top n architectures from search results."""
        sorted_results = sorted(self.search_results, key=lambda x: x['fitness'], reverse=True)
        return [result['architecture'] for result in sorted_results[:n]]
    
    def analyze_search_results(self) -> Dict[str, Any]:
        """Analyze search results and provide insights."""
        if not self.search_results:
            return {}
        
        fitnesses = [result['fitness'] for result in self.search_results]
        
        analysis = {
            "total_evaluations": len(self.search_results),
            "best_fitness": max(fitnesses),
            "mean_fitness": np.mean(fitnesses),
            "std_fitness": np.std(fitnesses),
            "convergence_trend": self._analyze_convergence(),
            "architecture_diversity": self._analyze_diversity()
        }
        
        return analysis
    
    def _analyze_convergence(self) -> List[float]:
        """Analyze convergence trend."""
        if not self.search_results:
            return []
        
        # Calculate moving average of best fitness
        window_size = min(50, len(self.search_results) // 10)
        if window_size < 2:
            return [result['fitness'] for result in self.search_results]
        
        convergence = []
        for i in range(len(self.search_results)):
            start_idx = max(0, i - window_size + 1)
            window_fitnesses = [self.search_results[j]['fitness'] for j in range(start_idx, i + 1)]
            convergence.append(max(window_fitnesses))
        
        return convergence
    
    def _analyze_diversity(self) -> Dict[str, float]:
        """Analyze architecture diversity."""
        if not self.search_results:
            return {}
        
        # Calculate diversity metrics
        layer_types = defaultdict(int)
        hidden_sizes = []
        dropout_rates = []
        
        for result in self.search_results:
            architecture = result['architecture']
            for gene in architecture.genes:
                layer_types[gene.layer_type] += 1
                hidden_sizes.append(gene.hidden_size)
                dropout_rates.append(gene.dropout)
        
        diversity = {
            "unique_layer_types": len(layer_types),
            "hidden_size_std": np.std(hidden_sizes) if hidden_sizes else 0,
            "dropout_rate_std": np.std(dropout_rates) if dropout_rates else 0,
            "layer_type_distribution": dict(layer_types)
        }
        
        return diversity
    
    def save_search_results(self, filepath: str):
        """Save search results to file."""
        import pickle
        
        results = {
            "config": self.config,
            "search_results": self.search_results,
            "best_architecture": self.search_strategy.best_architecture,
            "best_fitness": self.search_strategy.best_fitness
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Search results saved to {filepath}")
    
    def load_search_results(self, filepath: str):
        """Load search results from file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        
        self.config = results['config']
        self.search_results = results['search_results']
        self.search_strategy.best_architecture = results['best_architecture']
        self.search_strategy.best_fitness = results['best_fitness']
        
        logger.info(f"Search results loaded from {filepath}")


def create_fitness_fn(data_loader, val_data_loader, device: str = 'cuda') -> callable:
    """Create fitness function for architecture evaluation."""
    
    def fitness_fn(architecture: ArchitectureIndividual) -> float:
        """Evaluate architecture fitness."""
        try:
            # Convert architecture to model config
            model_config = architecture.to_model_config()
            
            # Create model
            model = HAMNet(model_config).to(device)
            
            # Simple training loop for evaluation
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Quick training (limited epochs for efficiency)
            model.train()
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(data_loader):
                if batch_idx >= 50:  # Limit batches for efficiency
                    break
                
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Validation
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_data_loader):
                    if batch_idx >= 20:  # Limit batches for efficiency
                        break
                    
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            # Calculate fitness (lower is better, so return negative)
            avg_train_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
            
            # Combine training and validation loss
            fitness = -(avg_train_loss + avg_val_loss)
            
            # Add complexity penalty
            complexity_penalty = 0.001 * len(architecture.genes)
            fitness -= complexity_penalty
            
            return fitness
            
        except Exception as e:
            logger.error(f"Error in fitness evaluation: {e}")
            return float('-inf')
    
    return fitness_fn


# Example usage
def example_nas_usage():
    """Example of how to use the NAS framework."""
    # Configuration
    config = ArchitectureConfig(
        search_strategy=SearchStrategy.EVOLUTIONARY,
        max_layers=15,
        min_layers=5,
        population_size=30,
        generations=50,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    
    # Create NAS framework
    nas = HAMNetNAS(config)
    
    # Mock data loaders (in practice, these would be real data)
    # train_loader = ...
    # val_loader = ...
    # fitness_fn = create_fitness_fn(train_loader, val_loader)
    
    # Perform search
    # best_architecture = nas.search(fitness_fn)
    
    # Analyze results
    # analysis = nas.analyze_search_results()
    # print(f"Best fitness: {analysis['best_fitness']:.4f}")
    
    # Get top architectures
    # top_architectures = nas.get_top_architectures(5)
    
    return nas


if __name__ == "__main__":
    # Run example
    nas = example_nas_usage()
    print("NAS framework initialized successfully")