"""
Comprehensive Benchmarking and Profiling Tools for HAMNet

This module provides comprehensive benchmarking and profiling capabilities
for analyzing model performance, memory usage, and computational efficiency.
"""

import os
import sys
import json
import time
import psutil
import GPUtil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler as profiler
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
import threading
import asyncio
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import thop
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ..models.hamnet import HAMNet, HAMNetConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BenchmarkType(Enum):
    """Types of benchmarks."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    ACCURACY = "accuracy"
    SCALABILITY = "scalability"
    ENERGY = "energy"
    ROBUSTNESS = "robustness"


class ProfilerType(Enum):
    """Types of profilers."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    PYTORCH = "pytorch"
    CUSTOM = "custom"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    # General settings
    benchmark_types: List[BenchmarkType] = field(default_factory=lambda: [
        BenchmarkType.LATENCY, BenchmarkType.THROUGHPUT, BenchmarkType.MEMORY, BenchmarkType.ACCURACY
    ])
    
    # Test settings
    num_iterations: int = 100
    warmup_iterations: int = 10
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 16, 32, 64])
    sequence_lengths: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    
    # Profiling settings
    enable_profiling: bool = True
    profiler_types: List[ProfilerType] = field(default_factory=lambda: [ProfilerType.PYTORCH, ProfilerType.GPU])
    profile_memory: bool = True
    record_shapes: bool = True
    
    # Output settings
    output_dir: str = "benchmark_results"
    save_plots: bool = True
    save_csv: bool = True
    save_json: bool = True
    detailed_report: bool = True
    
    # Hardware monitoring
    monitor_hardware: bool = True
    hardware_interval: float = 0.1  # seconds
    
    # Advanced settings
    enable_autotuning: bool = False
    autotuning_trials: int = 50
    compare_baseline: bool = True
    baseline_config: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkResult:
    """Result from a benchmark."""
    benchmark_type: BenchmarkType
    model_name: str
    config: Dict[str, Any]
    results: Dict[str, Any]
    timestamp: float
    hardware_info: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_type": self.benchmark_type.value,
            "model_name": self.model_name,
            "config": self.config,
            "results": self.results,
            "timestamp": self.timestamp,
            "hardware_info": self.hardware_info
        }


class HardwareMonitor:
    """Monitors hardware resource usage."""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.monitoring = False
        self.data = deque(maxlen=10000)
        self.thread = None
    
    def start(self):
        """Start monitoring."""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop monitoring."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1)
    
    def _monitor_loop(self):
        """Monitoring loop."""
        while self.monitoring:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                cpu_freq = psutil.cpu_freq()
                
                # Memory metrics
                memory = psutil.virtual_memory()
                
                # GPU metrics
                gpu_metrics = {}
                try:
                    gpus = GPUtil.getGPUs()
                    for i, gpu in enumerate(gpus):
                        gpu_metrics[f"gpu_{i}"] = {
                            "usage": gpu.load * 100,
                            "memory_used": gpu.memoryUsed,
                            "memory_total": gpu.memoryTotal,
                            "temperature": gpu.temperature
                        }
                except:
                    pass
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                
                # Network metrics
                network = psutil.net_io_counters()
                
                # Collect data
                data_point = {
                    "timestamp": time.time(),
                    "cpu_percent": cpu_percent,
                    "cpu_freq": cpu_freq.current if cpu_freq else 0,
                    "memory_percent": memory.percent,
                    "memory_used": memory.used,
                    "memory_total": memory.total,
                    "disk_percent": disk.percent,
                    "disk_used": disk.used,
                    "disk_total": disk.total,
                    "network_bytes_sent": network.bytes_sent,
                    "network_bytes_recv": network.bytes_recv,
                    "gpu_metrics": gpu_metrics
                }
                
                self.data.append(data_point)
                time.sleep(self.interval)
            
            except Exception as e:
                logger.error(f"Error in hardware monitoring: {e}")
                time.sleep(self.interval)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        if not self.data:
            return {}
        
        df = pd.DataFrame(list(self.data))
        
        summary = {
            "duration_seconds": df["timestamp"].max() - df["timestamp"].min(),
            "avg_cpu_percent": df["cpu_percent"].mean(),
            "max_cpu_percent": df["cpu_percent"].max(),
            "avg_memory_percent": df["memory_percent"].mean(),
            "max_memory_percent": df["memory_percent"].max(),
            "avg_memory_used_mb": df["memory_used"].mean() / (1024 * 1024),
            "max_memory_used_mb": df["memory_used"].max() / (1024 * 1024)
        }
        
        # GPU summary
        if "gpu_metrics" in df.columns and not df["gpu_metrics"].isna().all():
            gpu_data = []
            for gpu_dict in df["gpu_metrics"].dropna():
                for gpu_name, gpu_info in gpu_dict.items():
                    gpu_data.append({
                        "name": gpu_name,
                        "usage": gpu_info["usage"],
                        "memory_used": gpu_info["memory_used"]
                    })
            
            if gpu_data:
                gpu_df = pd.DataFrame(gpu_data)
                summary["gpu_summary"] = {
                    "avg_usage": gpu_df["usage"].mean(),
                    "max_usage": gpu_df["usage"].max(),
                    "avg_memory_used_mb": gpu_df["memory_used"].mean() / (1024 * 1024),
                    "max_memory_used_mb": gpu_df["memory_used"].max() / (1024 * 1024)
                }
        
        return summary


class LatencyBenchmark:
    """Benchmarks model latency."""
    
    def __init__(self, model: nn.Module, config: BenchmarkConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
    
    def run(self, input_shapes: List[Tuple[int, ...]]) -> BenchmarkResult:
        """Run latency benchmark."""
        logger.info("Running latency benchmark...")
        
        results = {}
        
        for shape in input_shapes:
            batch_size, seq_len = shape[0], shape[1]
            
            # Create input
            input_data = torch.randn(shape, device=self.device)
            
            # Warmup
            for _ in range(self.config.warmup_iterations):
                with torch.no_grad():
                    _ = self.model(input_data)
            
            # Benchmark
            latencies = []
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            for _ in range(self.config.num_iterations):
                start_time = time.time()
                
                with torch.no_grad():
                    _ = self.model(input_data)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                latency = time.time() - start_time
                latencies.append(latency)
            
            # Calculate statistics
            latencies = np.array(latencies)
            key = f"batch_{batch_size}_seq_{seq_len}"
            results[key] = {
                "mean_latency_ms": np.mean(latencies) * 1000,
                "median_latency_ms": np.median(latencies) * 1000,
                "std_latency_ms": np.std(latencies) * 1000,
                "p95_latency_ms": np.percentile(latencies, 95) * 1000,
                "p99_latency_ms": np.percentile(latencies, 99) * 1000,
                "min_latency_ms": np.min(latencies) * 1000,
                "max_latency_ms": np.max(latencies) * 1000,
                "throughput_per_second": batch_size / np.mean(latencies)
            }
        
        return BenchmarkResult(
            benchmark_type=BenchmarkType.LATENCY,
            model_name=self.model.__class__.__name__,
            config={"input_shapes": input_shapes},
            results=results,
            timestamp=time.time(),
            hardware_info=self._get_hardware_info()
        )
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "device": str(self.device)
        }
        
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return info


class ThroughputBenchmark:
    """Benchmarks model throughput."""
    
    def __init__(self, model: nn.Module, config: BenchmarkConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
    
    def run(self, batch_sizes: List[int], seq_length: int = 512) -> BenchmarkResult:
        """Run throughput benchmark."""
        logger.info("Running throughput benchmark...")
        
        results = {}
        
        for batch_size in batch_sizes:
            # Create input
            input_data = torch.randn(batch_size, seq_length, device=self.device)
            
            # Warmup
            for _ in range(self.config.warmup_iterations):
                with torch.no_grad():
                    _ = self.model(input_data)
            
            # Benchmark
            total_time = 0
            total_samples = 0
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            start_time = time.time()
            
            for _ in range(self.config.num_iterations):
                with torch.no_grad():
                    _ = self.model(input_data)
                
                total_samples += batch_size
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            total_time = time.time() - start_time
            
            # Calculate throughput
            throughput = total_samples / total_time
            
            key = f"batch_{batch_size}"
            results[key] = {
                "throughput_samples_per_second": throughput,
                "total_samples": total_samples,
                "total_time_seconds": total_time,
                "avg_latency_ms": (total_time / self.config.num_iterations) * 1000
            }
        
        return BenchmarkResult(
            benchmark_type=BenchmarkType.THROUGHPUT,
            model_name=self.model.__class__.__name__,
            config={"batch_sizes": batch_sizes, "seq_length": seq_length},
            results=results,
            timestamp=time.time(),
            hardware_info=self._get_hardware_info()
        )
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "device": str(self.device)
        }


class MemoryBenchmark:
    """Benchmarks model memory usage."""
    
    def __init__(self, model: nn.Module, config: BenchmarkConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
    
    def run(self, input_shapes: List[Tuple[int, ...]]) -> BenchmarkResult:
        """Run memory benchmark."""
        logger.info("Running memory benchmark...")
        
        results = {}
        
        for shape in input_shapes:
            batch_size, seq_len = shape[0], shape[1]
            
            # Reset memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # Create input
            input_data = torch.randn(shape, device=self.device)
            
            # Forward pass
            with torch.no_grad():
                _ = self.model(input_data)
            
            # Get memory usage
            memory_info = {}
            
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(self.device)
                memory_reserved = torch.cuda.memory_reserved(self.device)
                peak_memory = torch.cuda.max_memory_allocated(self.device)
                
                memory_info = {
                    "memory_allocated_mb": memory_allocated / (1024 * 1024),
                    "memory_reserved_mb": memory_reserved / (1024 * 1024),
                    "peak_memory_mb": peak_memory / (1024 * 1024)
                }
            else:
                process = psutil.Process()
                memory_info = {
                    "memory_rss_mb": process.memory_info().rss / (1024 * 1024),
                    "memory_vms_mb": process.memory_info().vms / (1024 * 1024)
                }
            
            # Model size
            model_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            memory_info["model_size_mb"] = model_size / (1024 * 1024)
            
            key = f"batch_{batch_size}_seq_{seq_len}"
            results[key] = memory_info
        
        return BenchmarkResult(
            benchmark_type=BenchmarkType.MEMORY,
            model_name=self.model.__class__.__name__,
            config={"input_shapes": input_shapes},
            results=results,
            timestamp=time.time(),
            hardware_info=self._get_hardware_info()
        )
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "device": str(self.device)
        }
        
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return info


class AccuracyBenchmark:
    """Benchmarks model accuracy."""
    
    def __init__(self, model: nn.Module, config: BenchmarkConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
    
    def run(self, test_data: Tuple[torch.Tensor, torch.Tensor]) -> BenchmarkResult:
        """Run accuracy benchmark."""
        logger.info("Running accuracy benchmark...")
        
        inputs, targets = test_data
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(inputs)
        
        # Calculate metrics
        predictions = outputs
        mse = mean_squared_error(targets.cpu().numpy(), predictions.cpu().numpy())
        mae = mean_absolute_error(targets.cpu().numpy(), predictions.cpu().numpy())
        r2 = r2_score(targets.cpu().numpy(), predictions.cpu().numpy())
        
        # Additional metrics
        mape = np.mean(np.abs((targets.cpu().numpy() - predictions.cpu().numpy()) / 
                             (targets.cpu().numpy() + 1e-8))) * 100
        
        results = {
            "mse": mse,
            "mae": mae,
            "r2_score": r2,
            "mape": mape,
            "num_samples": len(inputs),
            "input_shape": list(inputs.shape),
            "output_shape": list(outputs.shape)
        }
        
        return BenchmarkResult(
            benchmark_type=BenchmarkType.ACCURACY,
            model_name=self.model.__class__.__name__,
            config={"test_data_shape": list(inputs.shape)},
            results=results,
            timestamp=time.time(),
            hardware_info=self._get_hardware_info()
        )
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        return {
            "device": str(self.device)
        }


class ScalabilityBenchmark:
    """Benchmarks model scalability."""
    
    def __init__(self, model: nn.Module, config: BenchmarkConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
    
    def run(self, batch_sizes: List[int], seq_length: int = 512) -> BenchmarkResult:
        """Run scalability benchmark."""
        logger.info("Running scalability benchmark...")
        
        results = {}
        
        for batch_size in batch_sizes:
            # Create input
            input_data = torch.randn(batch_size, seq_length, device=self.device)
            
            # Measure time and memory for different batch sizes
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            
            with torch.no_grad():
                _ = self.model(input_data)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            # Get metrics
            metrics = {
                "batch_size": batch_size,
                "time_seconds": end_time - start_time,
                "throughput_samples_per_second": batch_size / (end_time - start_time)
            }
            
            if torch.cuda.is_available():
                metrics["peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
                metrics["memory_efficiency"] = batch_size / (torch.cuda.max_memory_allocated() / (1024 * 1024))
            
            key = f"batch_{batch_size}"
            results[key] = metrics
        
        # Calculate scaling efficiency
        baseline_batch = batch_sizes[0]
        baseline_time = results[f"batch_{baseline_batch}"]["time_seconds"]
        
        for batch_size in batch_sizes[1:]:
            key = f"batch_{batch_size}"
            expected_time = baseline_time * (batch_size / baseline_batch)
            actual_time = results[key]["time_seconds"]
            scaling_efficiency = expected_time / actual_time
            results[key]["scaling_efficiency"] = scaling_efficiency
        
        return BenchmarkResult(
            benchmark_type=BenchmarkType.SCALABILITY,
            model_name=self.model.__class__.__name__,
            config={"batch_sizes": batch_sizes, "seq_length": seq_length},
            results=results,
            timestamp=time.time(),
            hardware_info=self._get_hardware_info()
        )
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "device": str(self.device)
        }


class ModelProfiler:
    """Profiles model performance in detail."""
    
    def __init__(self, model: nn.Module, config: BenchmarkConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
    
    def profile_pytorch(self, input_data: torch.Tensor, output_dir: str) -> Dict[str, Any]:
        """Profile using PyTorch profiler."""
        logger.info("Running PyTorch profiler...")
        
        output_dir = Path(output_dir) / "pytorch_profile"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA
            ],
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=True
        ) as prof:
            with profiler.record_function("model_inference"):
                with torch.no_grad():
                    _ = self.model(input_data)
        
        # Export results
        prof.export_chrome_trace(str(output_dir / "trace.json"))
        prof.export_memory_timeline(str(output_dir / "memory_timeline.html"))
        prof.export_stacks(str(output_dir / "stacks.txt"), "self_cuda_time_total")
        
        # Get summary
        summary = {
            "cpu_time_total": prof.key_averages().total_average().cpu_time_total,
            "cuda_time_total": prof.key_averages().total_average().cuda_time_total,
            "cpu_memory_usage": prof.key_averages().total_average().cpu_memory_usage,
            "cuda_memory_usage": prof.key_averages().total_average().cuda_memory_usage,
            "self_cpu_time_total": prof.key_averages().total_average().self_cpu_time_total,
            "self_cuda_time_total": prof.key_averages().total_average().self_cuda_time_total
        }
        
        # Get top operations
        top_ops = []
        for event in prof.key_averages():
            top_ops.append({
                "name": event.key,
                "cpu_time": event.cpu_time_total,
                "cuda_time": event.cuda_time_total,
                "cpu_memory": event.cpu_memory_usage,
                "cuda_memory": event.cuda_memory_usage,
                "call_count": event.count
            })
        
        # Sort by CUDA time
        top_ops.sort(key=lambda x: x["cuda_time"], reverse=True)
        summary["top_operations"] = top_ops[:20]
        
        return summary
    
    def profile_flops(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """Profile FLOPs and parameters."""
        logger.info("Calculating FLOPs and parameters...")
        
        # Use thop to calculate FLOPs
        flops, params = thop.profile(
            self.model, 
            inputs=(input_data,),
            verbose=False
        )
        
        flops_mac = flops / 2  # Convert to MAC (multiply-accumulate)
        
        return {
            "flops": flops,
            "flops_mac": flops_mac,
            "parameters": params,
            "flops_per_parameter": flops / params if params > 0 else 0
        }
    
    def profile_layer_wise(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """Profile layer-wise performance."""
        logger.info("Running layer-wise profiling...")
        
        layer_stats = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                # Calculate FLOPs for this layer
                flops = 0
                if isinstance(module, nn.Linear):
                    flops = 2 * module.in_features * module.out_features
                elif isinstance(module, nn.Conv1d):
                    flops = 2 * module.in_channels * module.out_channels * module.kernel_size[0]
                elif isinstance(module, nn.MultiheadAttention):
                    # Simplified attention FLOPs
                    seq_len = input[0].size(1)
                    embed_dim = module.embed_dim
                    flops = 4 * seq_len * embed_dim * embed_dim
                
                layer_stats[name] = {
                    "type": module.__class__.__name__,
                    "output_shape": list(output.shape) if torch.is_tensor(output) else [],
                    "flops": flops,
                    "parameters": sum(p.numel() for p in module.parameters())
                }
            return hook
        
        # Register hooks
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Run forward pass
        with torch.no_grad():
            _ = self.model(input_data)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return layer_stats


class BenchmarkVisualizer:
    """Visualizes benchmark results."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_latency_results(self, results: List[BenchmarkResult]):
        """Plot latency benchmark results."""
        fig = go.Figure()
        
        for result in results:
            if result.benchmark_type == BenchmarkType.LATENCY:
                x_data = []
                y_data = []
                error_data = []
                
                for key, metrics in result.results.items():
                    batch_size = int(key.split("_")[1])
                    seq_len = int(key.split("_")[3])
                    
                    x_data.append(f"Batch {batch_size}, Seq {seq_len}")
                    y_data.append(metrics["mean_latency_ms"])
                    error_data.append(metrics["std_latency_ms"])
                
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    name=result.model_name,
                    error_y=dict(type="data", array=error_data),
                    mode="markers+lines"
                ))
        
        fig.update_layout(
            title="Latency Benchmark Results",
            xaxis_title="Configuration",
            yaxis_title="Mean Latency (ms)",
            yaxis_type="log"
        )
        
        fig.write_html(self.output_dir / "latency_benchmark.html")
    
    def plot_throughput_results(self, results: List[BenchmarkResult]):
        """Plot throughput benchmark results."""
        fig = go.Figure()
        
        for result in results:
            if result.benchmark_type == BenchmarkType.THROUGHPUT:
                x_data = []
                y_data = []
                
                for key, metrics in result.results.items():
                    batch_size = int(key.split("_")[1])
                    
                    x_data.append(batch_size)
                    y_data.append(metrics["throughput_samples_per_second"])
                
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    name=result.model_name,
                    mode="markers+lines"
                ))
        
        fig.update_layout(
            title="Throughput Benchmark Results",
            xaxis_title="Batch Size",
            yaxis_title="Throughput (samples/second)"
        )
        
        fig.write_html(self.output_dir / "throughput_benchmark.html")
    
    def plot_memory_results(self, results: List[BenchmarkResult]):
        """Plot memory benchmark results."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Memory Usage", "Model Size"),
            vertical_spacing=0.1
        )
        
        for result in results:
            if result.benchmark_type == BenchmarkType.MEMORY:
                x_data = []
                y_data_memory = []
                y_data_model = []
                
                for key, metrics in result.results.items():
                    batch_size = int(key.split("_")[1])
                    
                    x_data.append(batch_size)
                    y_data_memory.append(metrics.get("peak_memory_mb", 0))
                    y_data_model.append(metrics.get("model_size_mb", 0))
                
                fig.add_trace(
                    go.Scatter(x=x_data, y=y_data_memory, name=result.model_name),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=x_data, y=y_data_model, name=f"{result.model_name} (model)"),
                    row=2, col=1
                )
        
        fig.update_layout(
            title="Memory Benchmark Results",
            height=600
        )
        
        fig.write_html(self.output_dir / "memory_benchmark.html")
    
    def plot_accuracy_results(self, results: List[BenchmarkResult]):
        """Plot accuracy benchmark results."""
        metrics = ["mse", "mae", "r2_score", "mape"]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metrics,
            vertical_spacing=0.1
        )
        
        for i, metric in enumerate(metrics):
            row = i // 2 + 1
            col = i % 2 + 1
            
            for result in results:
                if result.benchmark_type == BenchmarkType.ACCURACY:
                    value = result.results.get(metric, 0)
                    
                    fig.add_trace(
                        go.Bar(
                            x=[result.model_name],
                            y=[value],
                            name=result.model_name,
                            showlegend=False if i > 0 else True
                        ),
                        row=row, col=col
                    )
        
        fig.update_layout(
            title="Accuracy Benchmark Results",
            height=600
        )
        
        fig.write_html(self.output_dir / "accuracy_benchmark.html")
    
    def plot_scalability_results(self, results: List[BenchmarkResult]):
        """Plot scalability benchmark results."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Throughput vs Batch Size", "Scaling Efficiency"),
            vertical_spacing=0.1
        )
        
        for result in results:
            if result.benchmark_type == BenchmarkType.SCALABILITY:
                x_data = []
                y_data_throughput = []
                y_data_efficiency = []
                
                for key, metrics in result.results.items():
                    batch_size = metrics["batch_size"]
                    
                    x_data.append(batch_size)
                    y_data_throughput.append(metrics["throughput_samples_per_second"])
                    y_data_efficiency.append(metrics.get("scaling_efficiency", 1.0))
                
                fig.add_trace(
                    go.Scatter(x=x_data, y=y_data_throughput, name=result.model_name),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=x_data, y=y_data_efficiency, name=result.model_name),
                    row=2, col=1
                )
        
        fig.update_layout(
            title="Scalability Benchmark Results",
            height=600
        )
        
        fig.write_html(self.output_dir / "scalability_benchmark.html")
    
    def generate_summary_report(self, results: List[BenchmarkResult]):
        """Generate comprehensive summary report."""
        report = {
            "timestamp": time.time(),
            "total_benchmarks": len(results),
            "benchmark_types": list(set(r.benchmark_type.value for r in results)),
            "models_tested": list(set(r.model_name for r in results)),
            "summary": {}
        }
        
        # Group by benchmark type
        by_type = defaultdict(list)
        for result in results:
            by_type[result.benchmark_type].append(result)
        
        # Generate summaries for each type
        for benchmark_type, type_results in by_type.items():
            report["summary"][benchmark_type.value] = self._summarize_benchmark_type(type_results)
        
        # Save report
        with open(self.output_dir / "summary_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _summarize_benchmark_type(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Summarize results for a specific benchmark type."""
        summary = {
            "num_results": len(results),
            "models": list(set(r.model_name for r in results)),
            "best_performance": {},
            "average_performance": {}
        }
        
        if results[0].benchmark_type == BenchmarkType.LATENCY:
            # Find best (lowest) latency
            all_latencies = []
            for result in results:
                for metrics in result.results.values():
                    all_latencies.append(metrics["mean_latency_ms"])
            
            summary["average_performance"]["mean_latency_ms"] = np.mean(all_latencies)
            summary["best_performance"]["min_latency_ms"] = np.min(all_latencies)
        
        elif results[0].benchmark_type == BenchmarkType.THROUGHPUT:
            # Find best (highest) throughput
            all_throughputs = []
            for result in results:
                for metrics in result.results.values():
                    all_throughputs.append(metrics["throughput_samples_per_second"])
            
            summary["average_performance"]["mean_throughput"] = np.mean(all_throughputs)
            summary["best_performance"]["max_throughput"] = np.max(all_throughputs)
        
        elif results[0].benchmark_type == BenchmarkType.ACCURACY:
            # Find best accuracy metrics
            all_mse = []
            all_r2 = []
            for result in results:
                all_mse.append(result.results["mse"])
                all_r2.append(result.results["r2_score"])
            
            summary["average_performance"]["mean_mse"] = np.mean(all_mse)
            summary["average_performance"]["mean_r2"] = np.mean(all_r2)
            summary["best_performance"]["min_mse"] = np.min(all_mse)
            summary["best_performance"]["max_r2"] = np.max(all_r2)
        
        return summary


class HAMNetBenchmarkSuite:
    """Comprehensive benchmark suite for HAMNet."""
    
    def __init__(self, model: nn.Module, config: BenchmarkConfig):
        self.model = model
        self.config = config
        self.results = []
        self.hardware_monitor = HardwareMonitor(config.hardware_interval)
        self.visualizer = BenchmarkVisualizer(config.output_dir)
    
    def run_benchmarks(self, test_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> List[BenchmarkResult]:
        """Run all configured benchmarks."""
        logger.info("Starting HAMNet benchmark suite...")
        
        # Start hardware monitoring
        if self.config.monitor_hardware:
            self.hardware_monitor.start()
        
        # Generate test configurations
        input_shapes = []
        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.sequence_lengths:
                input_shapes.append((batch_size, seq_len))
        
        # Run benchmarks
        for benchmark_type in self.config.benchmark_types:
            try:
                if benchmark_type == BenchmarkType.LATENCY:
                    benchmark = LatencyBenchmark(self.model, self.config)
                    result = benchmark.run(input_shapes)
                
                elif benchmark_type == BenchmarkType.THROUGHPUT:
                    benchmark = ThroughputBenchmark(self.model, self.config)
                    result = benchmark.run(self.config.batch_sizes)
                
                elif benchmark_type == BenchmarkType.MEMORY:
                    benchmark = MemoryBenchmark(self.model, self.config)
                    result = benchmark.run(input_shapes)
                
                elif benchmark_type == BenchmarkType.ACCURACY:
                    if test_data is None:
                        logger.warning("No test data provided, skipping accuracy benchmark")
                        continue
                    benchmark = AccuracyBenchmark(self.model, self.config)
                    result = benchmark.run(test_data)
                
                elif benchmark_type == BenchmarkType.SCALABILITY:
                    benchmark = ScalabilityBenchmark(self.model, self.config)
                    result = benchmark.run(self.config.batch_sizes)
                
                else:
                    logger.warning(f"Unsupported benchmark type: {benchmark_type}")
                    continue
                
                self.results.append(result)
                logger.info(f"Completed {benchmark_type.value} benchmark")
                
            except Exception as e:
                logger.error(f"Error running {benchmark_type.value} benchmark: {e}")
        
        # Run profiling if enabled
        if self.config.enable_profiling:
            self._run_profiling()
        
        # Stop hardware monitoring
        if self.config.monitor_hardware:
            self.hardware_monitor.stop()
        
        # Generate reports and visualizations
        self._generate_reports()
        
        return self.results
    
    def _run_profiling(self):
        """Run profiling benchmarks."""
        logger.info("Running profiling...")
        
        # Create sample input
        sample_input = torch.randn(8, 512, device=next(self.model.parameters()).device)
        
        # PyTorch profiler
        if ProfilerType.PYTORCH in self.config.profiler_types:
            profiler = ModelProfiler(self.model, self.config)
            profile_results = profiler.profile_pytorch(sample_input, self.config.output_dir)
            
            # Save profiling results
            with open(Path(self.config.output_dir) / "pytorch_profile_results.json", "w") as f:
                json.dump(profile_results, f, indent=2)
        
        # FLOPs profiling
        flops_results = profiler.profile_flops(sample_input)
        
        with open(Path(self.config.output_dir) / "flops_profile_results.json", "w") as f:
            json.dump(flops_results, f, indent=2)
        
        # Layer-wise profiling
        layer_results = profiler.profile_layer_wise(sample_input)
        
        with open(Path(self.config.output_dir) / "layer_profile_results.json", "w") as f:
            json.dump(layer_results, f, indent=2)
    
    def _generate_reports(self):
        """Generate reports and visualizations."""
        logger.info("Generating reports and visualizations...")
        
        # Generate plots
        self.visualizer.plot_latency_results(self.results)
        self.visualizer.plot_throughput_results(self.results)
        self.visualizer.plot_memory_results(self.results)
        self.visualizer.plot_accuracy_results(self.results)
        self.visualizer.plot_scalability_results(self.results)
        
        # Generate summary report
        summary_report = self.visualizer.generate_summary_report(self.results)
        
        # Save individual results
        if self.config.save_json:
            for i, result in enumerate(self.results):
                with open(Path(self.config.output_dir) / f"result_{i}_{result.benchmark_type.value}.json", "w") as f:
                    json.dump(result.to_dict(), f, indent=2)
        
        # Save CSV summary
        if self.config.save_csv:
            self._save_csv_summary()
        
        # Add hardware monitoring summary
        if self.config.monitor_hardware:
            hardware_summary = self.hardware_monitor.get_summary()
            with open(Path(self.config.output_dir) / "hardware_monitoring_summary.json", "w") as f:
                json.dump(hardware_summary, f, indent=2)
        
        logger.info(f"Benchmark results saved to {self.config.output_dir}")
    
    def _save_csv_summary(self):
        """Save summary as CSV."""
        summary_data = []
        
        for result in self.results:
            for config_key, metrics in result.results.items():
                row = {
                    "benchmark_type": result.benchmark_type.value,
                    "model_name": result.model_name,
                    "config": config_key,
                    "timestamp": result.timestamp
                }
                row.update(metrics)
                summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df.to_csv(Path(self.config.output_dir) / "benchmark_summary.csv", index=False)
    
    def compare_with_baseline(self, baseline_results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Compare current results with baseline."""
        if not baseline_results:
            return {}
        
        comparison = {
            "improvements": {},
            "degradations": {},
            "summary": {}
        }
        
        # Group results by benchmark type
        current_by_type = defaultdict(list)
        baseline_by_type = defaultdict(list)
        
        for result in self.results:
            current_by_type[result.benchmark_type].append(result)
        
        for result in baseline_results:
            baseline_by_type[result.benchmark_type].append(result)
        
        # Compare each benchmark type
        for benchmark_type in current_by_type:
            if benchmark_type not in baseline_by_type:
                continue
            
            current_results = current_by_type[benchmark_type]
            baseline_results_type = baseline_by_type[benchmark_type]
            
            # Calculate average improvements
            if benchmark_type == BenchmarkType.LATENCY:
                current_avg = np.mean([
                    metrics["mean_latency_ms"] 
                    for result in current_results 
                    for metrics in result.results.values()
                ])
                baseline_avg = np.mean([
                    metrics["mean_latency_ms"] 
                    for result in baseline_results_type 
                    for metrics in result.results.values()
                ])
                
                improvement = (baseline_avg - current_avg) / baseline_avg * 100
                
                if improvement > 0:
                    comparison["improvements"][benchmark_type.value] = improvement
                else:
                    comparison["degradations"][benchmark_type.value] = abs(improvement)
            
            elif benchmark_type == BenchmarkType.THROUGHPUT:
                current_avg = np.mean([
                    metrics["throughput_samples_per_second"] 
                    for result in current_results 
                    for metrics in result.results.values()
                ])
                baseline_avg = np.mean([
                    metrics["throughput_samples_per_second"] 
                    for result in baseline_results_type 
                    for metrics in result.results.values()
                ])
                
                improvement = (current_avg - baseline_avg) / baseline_avg * 100
                
                if improvement > 0:
                    comparison["improvements"][benchmark_type.value] = improvement
                else:
                    comparison["degradations"][benchmark_type.value] = abs(improvement)
        
        # Generate summary
        comparison["summary"]["total_improvements"] = len(comparison["improvements"])
        comparison["summary"]["total_degradations"] = len(comparison["degradations"])
        comparison["summary"]["overall_improvement"] = (
            sum(comparison["improvements"].values()) - sum(comparison["degradations"].values())
        )
        
        return comparison


# Example usage
def example_benchmarking():
    """Example of benchmarking usage."""
    # Model configuration
    model_config = HAMNetConfig(
        hidden_size=256,
        num_layers=4,
        input_size=512
    )
    
    # Create model
    model = HAMNet(model_config)
    model.eval()
    
    # Benchmark configuration
    benchmark_config = BenchmarkConfig(
        benchmark_types=[
            BenchmarkType.LATENCY,
            BenchmarkType.THROUGHPUT,
            BenchmarkType.MEMORY,
            BenchmarkType.SCALABILITY
        ],
        num_iterations=50,
        batch_sizes=[1, 8, 16, 32],
        sequence_lengths=[128, 256, 512],
        output_dir="benchmark_results",
        save_plots=True,
        save_csv=True,
        save_json=True,
        enable_profiling=True,
        monitor_hardware=True
    )
    
    # Create benchmark suite
    benchmark_suite = HAMNetBenchmarkSuite(model, benchmark_config)
    
    # Run benchmarks
    results = benchmark_suite.run_benchmarks()
    
    # Generate comparison report
    comparison = benchmark_suite.compare_with_baseline([])
    
    print(f"Benchmark completed with {len(results)} results")
    print(f"Comparison summary: {comparison}")
    
    return benchmark_suite


if __name__ == "__main__":
    # Run example
    benchmark_suite = example_benchmarking()
    print("Benchmarking framework initialized successfully")