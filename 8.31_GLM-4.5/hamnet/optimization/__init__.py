"""
HAMNet Optimization Framework

This module provides comprehensive model optimization and performance improvement
components for HAMNet models, including neural architecture search, hyperparameter
optimization, model compression, and production deployment tools.
"""

from .nas import (
    HAMNetNAS,
    ArchitectureConfig,
    SearchStrategy,
    OperationType,
    ArchitectureGene,
    ArchitectureIndividual,
    SearchSpace,
    BaseSearchStrategy,
    RandomSearch,
    EvolutionarySearch,
    BayesianOptimizationSearch,
    create_fitness_fn
)

from .hyperparameter_optimization import (
    HAMNetHyperparameterOptimizer,
    HyperparameterConfig,
    OptimizationStrategy,
    Hyperparameter,
    ParameterType,
    TrialResult,
    BaseOptimizer,
    RandomOptimizer,
    GridSearchOptimizer,
    BayesianOptimizer,
    HyperbandOptimizer,
    create_train_fn
)

from .pruning_quantization import (
    HAMNetOptimizer,
    PruningConfig,
    QuantizationConfig,
    PruningMethod,
    QuantizationMode,
    OptimizationResult,
    Pruner,
    MagnitudePruner,
    GradientPruner,
    StructuredPruner,
    MovementPruner,
    Quantizer
)

from .knowledge_distillation import (
    HAMNetKnowledgeDistillation,
    DistillationConfig,
    DistillationMethod,
    DistillationLoss,
    DistillationLossFunction,
    SoftTargetLoss,
    FeatureMatchingLoss,
    AttentionTransferLoss,
    RelationshipKnowledgeLoss,
    ContrastiveDistillationLoss,
    KnowledgeDistillationTrainer,
    MultiTeacherDistillationTrainer,
    SelfDistillationTrainer,
    DistillationResult
)

from .regularization import (
    HAMNetRegularizer,
    RegularizationConfig,
    RegularizationType,
    Regularizer,
    DropoutRegularizer,
    WeightDecayRegularizer,
    SpectralNormRegularizer,
    DropConnectRegularizer,
    DropPathRegularizer,
    LabelSmoothingRegularizer,
    MixupRegularizer,
    CutMixRegularizer,
    AdversarialRegularizer,
    VirtualAdversarialRegularizer,
    ConsistencyRegularizer,
    ManifoldRegularizer,
    ContrastiveRegularizer,
    InformationBottleneckRegularizer,
    VariationalRegularizer,
    RegularizationManager
)

from .training_optimization import (
    HAMNetTrainingOptimizer,
    TrainingOptimizationConfig,
    PrecisionMode,
    DistributedBackend,
    OptimizationStrategy,
    GradientAccumulator,
    MixedPrecisionTrainer,
    GradientCheckpointing,
    MemoryEfficientAttention,
    DistributedTrainingManager,
    TrainingOptimizer
)

from .inference_optimization import (
    HAMNetInferenceOptimizer,
    InferenceOptimizationConfig,
    InferenceBackend,
    OptimizationLevel,
    PrecisionMode as InferencePrecisionMode,
    InferenceProfiler,
    ONNXExporter,
    TensorRTOptimizer,
    BatchInferenceManager,
    InferenceOptimizer
)

from .deployment import (
    HAMNetDeploymentManager,
    DeploymentConfig,
    DeploymentEnvironment,
    DeploymentMode,
    HealthStatus,
    DeploymentMetrics,
    HealthChecker,
    AutoScaler,
    DockerManager,
    KubernetesManager,
    HAMNetAPI
)

from .monitoring import (
    HAMNetMonitoring,
    MonitoringConfig,
    LogLevel,
    MetricType,
    ExperimentStatus,
    ABTestType,
    Metric,
    HAMNetMetrics,
    StructuredLogger,
    AlertManager,
    ABTest,
    ABTestManager
)

from .benchmarking import (
    HAMNetBenchmarkSuite,
    BenchmarkConfig,
    BenchmarkType,
    ProfilerType,
    BenchmarkResult,
    HardwareMonitor,
    LatencyBenchmark,
    ThroughputBenchmark,
    MemoryBenchmark,
    AccuracyBenchmark,
    ScalabilityBenchmark,
    ModelProfiler,
    BenchmarkVisualizer
)

__version__ = "1.0.0"
__author__ = "HAMNet Team"
__email__ = "hamnet@example.com"

__all__ = [
    # Neural Architecture Search
    "HAMNetNAS",
    "ArchitectureConfig",
    "SearchStrategy",
    "OperationType",
    "ArchitectureGene",
    "ArchitectureIndividual",
    "SearchSpace",
    "BaseSearchStrategy",
    "RandomSearch",
    "EvolutionarySearch",
    "BayesianOptimizationSearch",
    "create_fitness_fn",
    
    # Hyperparameter Optimization
    "HAMNetHyperparameterOptimizer",
    "HyperparameterConfig",
    "OptimizationStrategy",
    "Hyperparameter",
    "ParameterType",
    "TrialResult",
    "BaseOptimizer",
    "RandomOptimizer",
    "GridSearchOptimizer",
    "BayesianOptimizer",
    "HyperbandOptimizer",
    "create_train_fn",
    
    # Model Pruning and Quantization
    "HAMNetOptimizer",
    "PruningConfig",
    "QuantizationConfig",
    "PruningMethod",
    "QuantizationMode",
    "OptimizationResult",
    "Pruner",
    "MagnitudePruner",
    "GradientPruner",
    "StructuredPruner",
    "MovementPruner",
    "Quantizer",
    
    # Knowledge Distillation
    "HAMNetKnowledgeDistillation",
    "DistillationConfig",
    "DistillationMethod",
    "DistillationLoss",
    "DistillationLossFunction",
    "SoftTargetLoss",
    "FeatureMatchingLoss",
    "AttentionTransferLoss",
    "RelationshipKnowledgeLoss",
    "ContrastiveDistillationLoss",
    "KnowledgeDistillationTrainer",
    "MultiTeacherDistillationTrainer",
    "SelfDistillationTrainer",
    "DistillationResult",
    
    # Advanced Regularization
    "HAMNetRegularizer",
    "RegularizationConfig",
    "RegularizationType",
    "Regularizer",
    "DropoutRegularizer",
    "WeightDecayRegularizer",
    "SpectralNormRegularizer",
    "DropConnectRegularizer",
    "DropPathRegularizer",
    "LabelSmoothingRegularizer",
    "MixupRegularizer",
    "CutMixRegularizer",
    "AdversarialRegularizer",
    "VirtualAdversarialRegularizer",
    "ConsistencyRegularizer",
    "ManifoldRegularizer",
    "ContrastiveRegularizer",
    "InformationBottleneckRegularizer",
    "VariationalRegularizer",
    "RegularizationManager",
    
    # Training Optimization
    "HAMNetTrainingOptimizer",
    "TrainingOptimizationConfig",
    "PrecisionMode",
    "DistributedBackend",
    "OptimizationStrategy",
    "GradientAccumulator",
    "MixedPrecisionTrainer",
    "GradientCheckpointing",
    "MemoryEfficientAttention",
    "DistributedTrainingManager",
    "TrainingOptimizer",
    
    # Inference Optimization
    "HAMNetInferenceOptimizer",
    "InferenceOptimizationConfig",
    "InferenceBackend",
    "OptimizationLevel",
    "InferencePrecisionMode",
    "InferenceProfiler",
    "ONNXExporter",
    "TensorRTOptimizer",
    "BatchInferenceManager",
    "InferenceOptimizer",
    
    # Production Deployment
    "HAMNetDeploymentManager",
    "DeploymentConfig",
    "DeploymentEnvironment",
    "DeploymentMode",
    "HealthStatus",
    "DeploymentMetrics",
    "HealthChecker",
    "AutoScaler",
    "DockerManager",
    "KubernetesManager",
    "HAMNetAPI",
    
    # Monitoring and A/B Testing
    "HAMNetMonitoring",
    "MonitoringConfig",
    "LogLevel",
    "MetricType",
    "ExperimentStatus",
    "ABTestType",
    "Metric",
    "HAMNetMetrics",
    "StructuredLogger",
    "AlertManager",
    "ABTest",
    "ABTestManager",
    
    # Benchmarking and Profiling
    "HAMNetBenchmarkSuite",
    "BenchmarkConfig",
    "BenchmarkType",
    "ProfilerType",
    "BenchmarkResult",
    "HardwareMonitor",
    "LatencyBenchmark",
    "ThroughputBenchmark",
    "MemoryBenchmark",
    "AccuracyBenchmark",
    "ScalabilityBenchmark",
    "ModelProfiler",
    "BenchmarkVisualizer"
]


def create_comprehensive_optimizer(
    model_config,
    enable_nas=True,
    enable_hyperparameter_opt=True,
    enable_pruning=True,
    enable_distillation=True,
    enable_regularization=True,
    enable_training_opt=True,
    enable_inference_opt=True,
    enable_monitoring=True,
    enable_benchmarking=True
):
    """
    Create a comprehensive optimization pipeline for HAMNet models.
    
    Args:
        model_config: HAMNet configuration
        enable_nas: Enable neural architecture search
        enable_hyperparameter_opt: Enable hyperparameter optimization
        enable_pruning: Enable model pruning and quantization
        enable_distillation: Enable knowledge distillation
        enable_regularization: Enable advanced regularization
        enable_training_opt: Enable training optimization
        enable_inference_opt: Enable inference optimization
        enable_monitoring: Enable monitoring and A/B testing
        enable_benchmarking: Enable benchmarking and profiling
    
    Returns:
        Dictionary containing all enabled optimizers
    """
    optimizers = {}
    
    if enable_nas:
        nas_config = ArchitectureConfig()
        optimizers["nas"] = HAMNetNAS(nas_config)
    
    if enable_hyperparameter_opt:
        hyperopt_config = HyperparameterConfig()
        optimizers["hyperparameter"] = HAMNetHyperparameterOptimizer(hyperopt_config)
    
    if enable_pruning:
        pruning_config = PruningConfig()
        quant_config = QuantizationConfig()
        optimizers["pruning"] = HAMNetOptimizer(pruning_config, quant_config)
    
    if enable_distillation:
        distill_config = DistillationConfig()
        optimizers["distillation"] = HAMNetKnowledgeDistillation(None, distill_config)
    
    if enable_regularization:
        reg_config = RegularizationConfig()
        optimizers["regularization"] = HAMNetRegularizer(reg_config)
    
    if enable_training_opt:
        train_config = TrainingOptimizationConfig()
        optimizers["training"] = HAMNetTrainingOptimizer(model_config, train_config)
    
    if enable_inference_opt:
        infer_config = InferenceOptimizationConfig()
        optimizers["inference"] = HAMNetInferenceOptimizer(model_config, infer_config)
    
    if enable_monitoring:
        monitor_config = MonitoringConfig()
        optimizers["monitoring"] = HAMNetMonitoring(monitor_config)
    
    if enable_benchmarking:
        bench_config = BenchmarkConfig()
        optimizers["benchmarking"] = HAMNetBenchmarkSuite(None, bench_config)
    
    return optimizers


def optimize_hamnet_pipeline(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimization_config=None
):
    """
    Run a complete optimization pipeline for HAMNet models.
    
    Args:
        model: HAMNet model to optimize
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        optimization_config: Custom optimization configuration
    
    Returns:
        Optimized model and optimization results
    """
    if optimization_config is None:
        optimization_config = {}
    
    logger.info("Starting comprehensive HAMNet optimization pipeline")
    
    results = {}
    
    # Step 1: Neural Architecture Search (if enabled)
    if optimization_config.get("enable_nas", True):
        logger.info("Step 1: Neural Architecture Search")
        nas_config = ArchitectureConfig()
        nas = HAMNetNAS(nas_config)
        
        fitness_fn = create_fitness_fn(train_loader, val_loader)
        best_architecture = nas.search(fitness_fn, max_iterations=50)
        
        results["nas"] = {
            "best_architecture": best_architecture,
            "search_summary": nas.analyze_search_results()
        }
        
        # Create new model with best architecture
        model_config = best_architecture.to_model_config()
        model = HAMNet(model_config)
    
    # Step 2: Hyperparameter Optimization (if enabled)
    if optimization_config.get("enable_hyperparameter_opt", True):
        logger.info("Step 2: Hyperparameter Optimization")
        hyperopt_config = HyperparameterConfig()
        hyperopt = HAMNetHyperparameterOptimizer(hyperopt_config)
        
        train_fn = create_train_fn(train_loader, val_loader)
        best_trial = hyperopt.optimize(train_fn, test_fn=create_train_fn(test_loader, val_loader))
        
        results["hyperparameter"] = {
            "best_trial": best_trial,
            "optimization_summary": hyperopt.analyze_results()
        }
    
    # Step 3: Advanced Regularization (if enabled)
    if optimization_config.get("enable_regularization", True):
        logger.info("Step 3: Advanced Regularization")
        reg_config = RegularizationConfig()
        regularizer = HAMNetRegularizer(reg_config)
        
        # Apply regularization to model
        model = regularizer.prepare_model(model)
        
        results["regularization"] = {
            "enabled_techniques": [t.value for t in reg_config.enabled_techniques],
            "regularization_summary": regularizer.get_regularization_summary()
        }
    
    # Step 4: Training Optimization (if enabled)
    if optimization_config.get("enable_training_opt", True):
        logger.info("Step 4: Training Optimization")
        train_config = TrainingOptimizationConfig()
        train_optimizer = HAMNetTrainingOptimizer(model.config, train_config)
        
        optimized_model, training_summary = train_optimizer.train(
            train_loader, val_loader, epochs=100
        )
        
        results["training"] = {
            "model": optimized_model,
            "training_summary": training_summary
        }
        
        model = optimized_model
    
    # Step 5: Model Pruning and Quantization (if enabled)
    if optimization_config.get("enable_pruning", True):
        logger.info("Step 5: Model Pruning and Quantization")
        pruning_config = PruningConfig()
        quant_config = QuantizationConfig()
        
        pruning_optimizer = HAMNetOptimizer(pruning_config, quant_config)
        optimized_model, optimization_result = pruning_optimizer.optimize_model(
            model, train_loader, val_loader, test_loader
        )
        
        results["pruning"] = {
            "model": optimized_model,
            "optimization_result": optimization_result,
            "optimization_summary": pruning_optimizer.get_optimization_summary()
        }
        
        model = optimized_model
    
    # Step 6: Knowledge Distillation (if enabled)
    if optimization_config.get("enable_distillation", True):
        logger.info("Step 6: Knowledge Distillation")
        distill_config = DistillationConfig()
        
        # Use original model as teacher
        distillation = HAMNetKnowledgeDistillation(model, distill_config)
        student_model, distillation_result = distillation.distill(
            train_loader, val_loader, compression_ratio=0.5
        )
        
        results["distillation"] = {
            "student_model": student_model,
            "distillation_result": distillation_result,
            "distillation_summary": distillation.get_distillation_summary()
        }
        
        model = student_model
    
    # Step 7: Inference Optimization (if enabled)
    if optimization_config.get("enable_inference_opt", True):
        logger.info("Step 7: Inference Optimization")
        infer_config = InferenceOptimizationConfig()
        
        inference_optimizer = HAMNetInferenceOptimizer(model.config, infer_config)
        
        # Create sample input for optimization
        sample_input = next(iter(test_loader))[0][:1]
        optimized_paths = inference_optimizer.optimize_model(model, sample_input, "optimized_models")
        
        results["inference"] = {
            "optimized_paths": optimized_paths,
            "optimization_summary": inference_optimizer.get_optimization_summary()
        }
    
    # Step 8: Benchmarking (if enabled)
    if optimization_config.get("enable_benchmarking", True):
        logger.info("Step 8: Benchmarking")
        bench_config = BenchmarkConfig()
        
        benchmark_suite = HAMNetBenchmarkSuite(model, bench_config)
        
        # Get test data for accuracy benchmark
        test_data = next(iter(test_loader))
        benchmark_results = benchmark_suite.run_benchmarks(test_data)
        
        results["benchmarking"] = {
            "benchmark_results": benchmark_results,
            "output_dir": bench_config.output_dir
        }
    
    # Step 9: Setup Monitoring (if enabled)
    if optimization_config.get("enable_monitoring", True):
        logger.info("Step 9: Setting up Monitoring")
        monitor_config = MonitoringConfig()
        
        monitoring = HAMNetMonitoring(monitor_config)
        monitoring.start_monitoring()
        
        results["monitoring"] = {
            "monitoring": monitoring,
            "monitoring_config": monitor_config
        }
    
    logger.info("HAMNet optimization pipeline completed successfully")
    
    return {
        "optimized_model": model,
        "results": results,
        "summary": generate_optimization_summary(results)
    }


def generate_optimization_summary(results):
    """Generate a summary of optimization results."""
    summary = {
        "total_optimizations_applied": len(results),
        "optimization_types": list(results.keys()),
        "model_improvements": {},
        "recommendations": []
    }
    
    # Analyze improvements
    if "benchmarking" in results:
        bench_results = results["benchmarking"]["benchmark_results"]
        baseline_metrics = {}
        
        for result in bench_results:
            if result.benchmark_type.value == "latency":
                for metrics in result.results.values():
                    baseline_metrics["latency"] = metrics["mean_latency_ms"]
            elif result.benchmark_type.value == "memory":
                for metrics in result.results.values():
                    baseline_metrics["memory"] = metrics.get("peak_memory_mb", 0)
            elif result.benchmark_type.value == "accuracy":
                baseline_metrics["accuracy"] = result.results.get("r2_score", 0)
    
    summary["baseline_metrics"] = baseline_metrics
    
    # Generate recommendations
    if "nas" in results:
        summary["recommendations"].append("Consider further architecture exploration with different search strategies")
    
    if "hyperparameter" in results:
        summary["recommendations"].append("Monitor hyperparameter performance over time for continuous improvement")
    
    if "pruning" in results:
        pruning_result = results["pruning"]["optimization_result"]
        if pruning_result.accuracy_drop > 0.02:
            summary["recommendations"].append("Accuracy drop significant, consider fine-tuning pruned model")
    
    if "distillation" in results:
        distillation_result = results["distillation"]["distillation_result"]
        if distillation_result.student_accuracy < distillation_result.teacher_accuracy * 0.95:
            summary["recommendations"].append("Student model performance lower than expected, adjust distillation parameters")
    
    if "inference" in results:
        summary["recommendations"].append("Deploy optimized model with appropriate serving infrastructure")
    
    return summary


# Example usage functions
def example_basic_optimization():
    """Example of basic optimization pipeline."""
    from ..models.hamnet import HAMNet, HAMNetConfig
    
    # Create model
    model_config = HAMNetConfig(
        hidden_size=256,
        num_layers=4,
        input_size=512
    )
    model = HAMNet(model_config)
    
    # Create simple optimization pipeline
    optimizers = create_comprehensive_optimizer(
        model_config,
        enable_nas=False,
        enable_hyperparameter_opt=True,
        enable_pruning=True,
        enable_distillation=False,
        enable_regularization=True,
        enable_training_opt=True,
        enable_inference_opt=True,
        enable_monitoring=True,
        enable_benchmarking=True
    )
    
    print(f"Created {len(optimizers)} optimizers")
    return optimizers


def example_production_optimization():
    """Example of production-ready optimization."""
    print("HAMNet Optimization Framework")
    print("=============================")
    print("Available optimization components:")
    print("1. Neural Architecture Search (NAS)")
    print("2. Hyperparameter Optimization")
    print("3. Model Pruning & Quantization")
    print("4. Knowledge Distillation")
    print("5. Advanced Regularization")
    print("6. Training Optimization")
    print("7. Inference Optimization")
    print("8. Production Deployment")
    print("9. Monitoring & A/B Testing")
    print("10. Benchmarking & Profiling")
    print()
    print("Use optimize_hamnet_pipeline() for complete optimization")
    print("Use create_comprehensive_optimizer() for custom optimization")
    
    return example_basic_optimization()


if __name__ == "__main__":
    example_production_optimization()