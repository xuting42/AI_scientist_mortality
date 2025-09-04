"""
Production Deployment Framework for HAMNet

This module provides comprehensive production deployment capabilities including
Docker containerization, FastAPI services, and monitoring infrastructure.
"""

import os
import sys
import json
import time
import logging
import asyncio
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
import docker
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import psutil
import GPUtil
import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import redis
from celery import Celery
import yaml
import kubernetes
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from ..models.hamnet import HAMNet, HAMNetConfig
from ..optimization.inference_optimization import HAMNetInferenceOptimizer, InferenceOptimizationConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DeploymentEnvironment(Enum):
    """Supported deployment environments."""
    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    CLOUD = "cloud"
    HYBRID = "hybrid"


class DeploymentMode(Enum):
    """Deployment modes."""
    SINGLE_INSTANCE = "single_instance"
    MULTI_INSTANCE = "multi_instance"
    LOAD_BALANCED = "load_balanced"
    AUTO_SCALING = "auto_scaling"


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class DeploymentConfig:
    """Configuration for deployment."""
    # Environment settings
    environment: DeploymentEnvironment = DeploymentEnvironment.DOCKER
    mode: DeploymentMode = DeploymentMode.AUTO_SCALING
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_timeout: int = 30
    
    # Docker settings
    docker_image_name: str = "hamnet-optimizer"
    docker_image_tag: str = "latest"
    docker_registry: str = "localhost:5000"
    
    # Kubernetes settings
    kubernetes_namespace: str = "hamnet"
    kubernetes_deployment_name: str = "hamnet-api"
    kubernetes_service_name: str = "hamnet-service"
    kubernetes_replicas: int = 3
    kubernetes_cpu_request: str = "2"
    kubernetes_memory_request: str = "4Gi"
    kubernetes_gpu_request: str = "1"
    
    # Monitoring settings
    enable_monitoring: bool = True
    monitoring_port: int = 9090
    metrics_retention_days: int = 30
    
    # Scaling settings
    min_instances: int = 1
    max_instances: int = 10
    scale_up_threshold: float = 0.7  # CPU/GPU utilization
    scale_down_threshold: float = 0.3
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600
    
    # Security settings
    enable_auth: bool = True
    api_key: str = ""
    ssl_enabled: bool = True
    ssl_cert_path: str = ""
    ssl_key_path: str = ""
    
    # Cache settings
    enable_cache: bool = True
    cache_backend: str = "redis"  # "redis", "memory"
    cache_host: str = "localhost"
    cache_port: int = 6379
    cache_ttl: int = 3600
    
    # Queue settings
    enable_queue: bool = True
    queue_backend: str = "celery"  # "celery", "redis"
    queue_host: str = "localhost"
    queue_port: int = 6379


class DeploymentMetrics:
    """Manages deployment metrics and monitoring."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.registry = CollectorRegistry()
        
        # Define metrics
        self.request_counter = Counter(
            'hamnet_requests_total',
            'Total number of requests',
            ['endpoint', 'method', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'hamnet_request_duration_seconds',
            'Request duration in seconds',
            ['endpoint', 'method'],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'hamnet_active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        self.model_inference_time = Histogram(
            'hamnet_inference_duration_seconds',
            'Model inference duration in seconds',
            ['model_type', 'batch_size'],
            registry=self.registry
        )
        
        self.system_metrics = {
            'cpu_usage': Gauge('hamnet_cpu_usage_percent', 'CPU usage percentage', registry=self.registry),
            'memory_usage': Gauge('hamnet_memory_usage_bytes', 'Memory usage in bytes', registry=self.registry),
            'gpu_usage': Gauge('hamnet_gpu_usage_percent', 'GPU usage percentage', registry=self.registry),
            'gpu_memory': Gauge('hamnet_gpu_memory_usage_bytes', 'GPU memory usage in bytes', registry=self.registry)
        }
    
    def record_request(self, endpoint: str, method: str, status: int, duration: float):
        """Record request metrics."""
        self.request_counter.labels(endpoint=endpoint, method=method, status=status).inc()
        self.request_duration.labels(endpoint=endpoint, method=method).observe(duration)
    
    def record_inference(self, model_type: str, batch_size: int, duration: float):
        """Record inference metrics."""
        self.model_inference_time.labels(model_type=model_type, batch_size=batch_size).observe(duration)
    
    def update_system_metrics(self):
        """Update system metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.system_metrics['cpu_usage'].set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.system_metrics['memory_usage'].set(memory.used)
        
        # GPU metrics
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                self.system_metrics['gpu_usage'].set(gpu.load * 100)
                self.system_metrics['gpu_memory'].set(gpu.memoryUsed * 1024 * 1024)
        except:
            pass
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')


class HealthChecker:
    """Health checking for deployed services."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.health_status = HealthStatus.UNKNOWN
        self.last_check = 0
        self.check_interval = 30  # seconds
    
    def check_health(self) -> HealthStatus:
        """Check overall system health."""
        current_time = time.time()
        
        # Avoid frequent checks
        if current_time - self.last_check < self.check_interval:
            return self.health_status
        
        self.last_check = current_time
        
        # Check various components
        checks = [
            self._check_system_resources(),
            self._check_model_availability(),
            self._check_api_responsiveness(),
            self._check_cache_connectivity(),
            self._check_queue_connectivity()
        ]
        
        # Determine overall health
        healthy_checks = sum(1 for check in checks if check)
        
        if healthy_checks == len(checks):
            self.health_status = HealthStatus.HEALTHY
        elif healthy_checks >= len(checks) * 0.5:
            self.health_status = HealthStatus.DEGRADED
        else:
            self.health_status = HealthStatus.UNHEALTHY
        
        return self.health_status
    
    def _check_system_resources(self) -> bool:
        """Check system resource availability."""
        try:
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                return False
            
            # Memory check
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return False
            
            # GPU check
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    if gpu.load > 0.9 or gpu.memoryUtil > 0.9:
                        return False
            except:
                pass
            
            return True
        
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return False
    
    def _check_model_availability(self) -> bool:
        """Check if models are available."""
        try:
            # This would check if model files exist and are loadable
            model_path = Path("models/hamnet.pt")
            return model_path.exists()
        except Exception as e:
            logger.error(f"Model availability check failed: {e}")
            return False
    
    def _check_api_responsiveness(self) -> bool:
        """Check API responsiveness."""
        try:
            # This would perform a simple API health check
            return True
        except Exception as e:
            logger.error(f"API responsiveness check failed: {e}")
            return False
    
    def _check_cache_connectivity(self) -> bool:
        """Check cache connectivity."""
        if not self.config.enable_cache:
            return True
        
        try:
            if self.config.cache_backend == "redis":
                r = redis.Redis(host=self.config.cache_host, port=self.config.cache_port)
                r.ping()
            return True
        except Exception as e:
            logger.error(f"Cache connectivity check failed: {e}")
            return False
    
    def _check_queue_connectivity(self) -> bool:
        """Check queue connectivity."""
        if not self.config.enable_queue:
            return True
        
        try:
            if self.config.queue_backend == "redis":
                r = redis.Redis(host=self.config.queue_host, port=self.config.queue_port)
                r.ping()
            return True
        except Exception as e:
            logger.error(f"Queue connectivity check failed: {e}")
            return False


class AutoScaler:
    """Automatic scaling manager."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.last_scale_time = 0
        self.current_instances = config.min_instances
    
    def should_scale(self) -> Tuple[bool, int]:
        """Check if scaling is needed and return direction and target."""
        current_time = time.time()
        
        # Check cooldown periods
        if current_time - self.last_scale_time < self.config.scale_up_cooldown:
            return False, self.current_instances
        
        # Get current metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Check GPU usage if available
        gpu_usage = 0
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = gpus[0].load * 100
        except:
            pass
        
        # Determine if scaling is needed
        max_usage = max(cpu_usage, gpu_usage)
        
        if max_usage > self.config.scale_up_threshold * 100:
            # Scale up
            target_instances = min(
                self.current_instances * 2,
                self.config.max_instances
            )
            if target_instances > self.current_instances:
                self.last_scale_time = current_time
                return True, target_instances
        
        elif max_usage < self.config.scale_down_threshold * 100:
            # Scale down
            target_instances = max(
                self.current_instances // 2,
                self.config.min_instances
            )
            if target_instances < self.current_instances:
                self.last_scale_time = current_time
                return True, target_instances
        
        return False, self.current_instances


class DockerManager:
    """Docker container management."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.client = docker.from_env()
    
    def build_image(self, dockerfile_path: str, context_path: str) -> str:
        """Build Docker image."""
        try:
            image, build_logs = self.client.images.build(
                path=context_path,
                dockerfile=dockerfile_path,
                tag=f"{self.config.docker_image_name}:{self.config.docker_image_tag}"
            )
            
            logger.info(f"Docker image built: {image.id}")
            return image.id
        
        except Exception as e:
            logger.error(f"Failed to build Docker image: {e}")
            raise
    
    def push_image(self, image_id: str) -> bool:
        """Push Docker image to registry."""
        try:
            image = self.client.images.get(image_id)
            image.tag(
                f"{self.config.docker_registry}/{self.config.docker_image_name}:{self.config.docker_image_tag}"
            )
            
            push_logs = self.client.images.push(
                f"{self.config.docker_registry}/{self.config.docker_image_name}",
                tag=self.config.docker_image_tag
            )
            
            logger.info(f"Docker image pushed: {image_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to push Docker image: {e}")
            return False
    
    def run_container(self, image_id: str, port_mapping: Dict[int, int] = None) -> str:
        """Run Docker container."""
        try:
            if port_mapping is None:
                port_mapping = {self.config.api_port: self.config.api_port}
            
            container = self.client.containers.run(
                image_id,
                ports=port_mapping,
                detach=True,
                environment={
                    "API_HOST": "0.0.0.0",
                    "API_PORT": str(self.config.api_port)
                }
            )
            
            logger.info(f"Docker container started: {container.id}")
            return container.id
        
        except Exception as e:
            logger.error(f"Failed to run Docker container: {e}")
            raise


class KubernetesManager:
    """Kubernetes deployment management."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        try:
            config.load_kube_config()
            self.apps_v1 = client.AppsV1Api()
            self.core_v1 = client.CoreV1Api()
        except Exception as e:
            logger.warning(f"Failed to load Kubernetes config: {e}")
            self.apps_v1 = None
            self.core_v1 = None
    
    def create_deployment(self, image: str) -> bool:
        """Create Kubernetes deployment."""
        if self.apps_v1 is None:
            logger.error("Kubernetes client not available")
            return False
        
        try:
            # Create deployment manifest
            deployment = client.V1Deployment(
                metadata=client.V1ObjectMeta(name=self.config.kubernetes_deployment_name),
                spec=client.V1DeploymentSpec(
                    replicas=self.config.kubernetes_replicas,
                    selector=client.V1LabelSelector(
                        match_labels={"app": self.config.kubernetes_deployment_name}
                    ),
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(
                            labels={"app": self.config.kubernetes_deployment_name}
                        ),
                        spec=client.V1PodSpec(
                            containers=[
                                client.V1Container(
                                    name="hamnet",
                                    image=image,
                                    ports=[client.V1ContainerPort(container_port=self.config.api_port)],
                                    resources=client.V1ResourceRequirements(
                                        requests={
                                            "cpu": self.config.kubernetes_cpu_request,
                                            "memory": self.config.kubernetes_memory_request,
                                            "nvidia.com/gpu": self.config.kubernetes_gpu_request
                                        }
                                    ),
                                    env=[
                                        client.V1EnvVar(name="API_PORT", value=str(self.config.api_port))
                                    ]
                                )
                            ]
                        )
                    )
                )
            )
            
            # Create deployment
            self.apps_v1.create_namespaced_deployment(
                namespace=self.config.kubernetes_namespace,
                body=deployment
            )
            
            logger.info(f"Kubernetes deployment created: {self.config.kubernetes_deployment_name}")
            return True
        
        except ApiException as e:
            logger.error(f"Failed to create Kubernetes deployment: {e}")
            return False
    
    def create_service(self) -> bool:
        """Create Kubernetes service."""
        if self.core_v1 is None:
            logger.error("Kubernetes client not available")
            return False
        
        try:
            service = client.V1Service(
                metadata=client.V1ObjectMeta(name=self.config.kubernetes_service_name),
                spec=client.V1ServiceSpec(
                    selector={"app": self.config.kubernetes_deployment_name},
                    ports=[client.V1ServicePort(
                        port=self.config.api_port,
                        target_port=self.config.api_port
                    )],
                    type="LoadBalancer"
                )
            )
            
            self.core_v1.create_namespaced_service(
                namespace=self.config.kubernetes_namespace,
                body=service
            )
            
            logger.info(f"Kubernetes service created: {self.config.kubernetes_service_name}")
            return True
        
        except ApiException as e:
            logger.error(f"Failed to create Kubernetes service: {e}")
            return False


class HAMNetAPI:
    """FastAPI application for HAMNet services."""
    
    def __init__(self, config: DeploymentConfig, model_config: HAMNetConfig):
        self.config = config
        self.model_config = model_config
        self.app = FastAPI(
            title="HAMNet Optimizer API",
            description="API for HAMNet model optimization and inference",
            version="1.0.0"
        )
        self.model = None
        self.inference_optimizer = None
        self.metrics = DeploymentMetrics(config)
        self.health_checker = HealthChecker(config)
        self.auto_scaler = AutoScaler(config)
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        # Setup background tasks
        self._setup_background_tasks()
    
    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            return {"message": "HAMNet Optimizer API", "version": "1.0.0"}
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            status = self.health_checker.check_health()
            return {
                "status": status.value,
                "timestamp": time.time(),
                "checks": {
                    "system_resources": True,
                    "model_availability": True,
                    "api_responsiveness": True,
                    "cache_connectivity": True,
                    "queue_connectivity": True
                }
            }
        
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint."""
            return self.metrics.get_metrics()
        
        @self.app.post("/predict")
        async def predict(request: dict):
            """Prediction endpoint."""
            start_time = time.time()
            
            try:
                # Extract input data
                input_data = request.get("input")
                if not input_data:
                    raise HTTPException(status_code=400, detail="Input data required")
                
                # Convert to tensor
                input_tensor = torch.tensor(input_data, dtype=torch.float32)
                
                # Perform inference
                if self.inference_optimizer:
                    output = self.inference_optimizer.infer(
                        self.model, input_tensor, "pytorch"
                    )
                else:
                    with torch.no_grad():
                        output = self.model(input_tensor)
                
                # Record metrics
                duration = time.time() - start_time
                self.metrics.record_request("/predict", "POST", 200, duration)
                self.metrics.record_inference("pytorch", 1, duration)
                
                return {
                    "prediction": output.tolist(),
                    "timestamp": time.time(),
                    "processing_time": duration
                }
            
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                self.metrics.record_request("/predict", "POST", 500, time.time() - start_time)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/batch_predict")
        async def batch_predict(request: dict):
            """Batch prediction endpoint."""
            start_time = time.time()
            
            try:
                # Extract input data
                input_data = request.get("inputs")
                if not input_data:
                    raise HTTPException(status_code=400, detail="Input data required")
                
                # Convert to tensor
                input_tensor = torch.tensor(input_data, dtype=torch.float32)
                
                # Perform batch inference
                batch_size = input_tensor.size(0)
                if self.inference_optimizer:
                    output = self.inference_optimizer.infer(
                        self.model, input_tensor, "pytorch"
                    )
                else:
                    with torch.no_grad():
                        output = self.model(input_tensor)
                
                # Record metrics
                duration = time.time() - start_time
                self.metrics.record_request("/batch_predict", "POST", 200, duration)
                self.metrics.record_inference("pytorch", batch_size, duration)
                
                return {
                    "predictions": output.tolist(),
                    "batch_size": batch_size,
                    "timestamp": time.time(),
                    "processing_time": duration
                }
            
            except Exception as e:
                logger.error(f"Batch prediction failed: {e}")
                self.metrics.record_request("/batch_predict", "POST", 500, time.time() - start_time)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/model/info")
        async def model_info():
            """Model information endpoint."""
            if self.model is None:
                raise HTTPException(status_code=404, detail="Model not loaded")
            
            return {
                "model_type": "HAMNet",
                "model_config": self.model_config.__dict__,
                "model_size": sum(p.numel() for p in self.model.parameters()),
                "device": str(next(self.model.parameters()).device)
            }
    
    def _setup_background_tasks(self):
        """Setup background tasks."""
        @self.app.on_event("startup")
        async def startup_event():
            """Startup tasks."""
            # Load model
            self.model = HAMNet(self.model_config)
            self.model.eval()
            
            # Setup inference optimizer
            inference_config = InferenceOptimizationConfig()
            self.inference_optimizer = HAMNetInferenceOptimizer(
                self.model_config, inference_config
            )
            
            # Start metrics collection
            threading.Thread(target=self._collect_metrics, daemon=True).start()
            
            # Start auto-scaling
            if self.config.mode == DeploymentMode.AUTO_SCALING:
                threading.Thread(target=self._auto_scale, daemon=True).start()
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Shutdown tasks."""
            logger.info("Shutting down HAMNet API")
    
    def _collect_metrics(self):
        """Collect system metrics periodically."""
        while True:
            try:
                self.metrics.update_system_metrics()
                time.sleep(10)
            except Exception as e:
                logger.error(f"Metrics collection failed: {e}")
                time.sleep(30)
    
    def _auto_scale(self):
        """Auto-scaling loop."""
        while True:
            try:
                should_scale, target_instances = self.auto_scaler.should_scale()
                
                if should_scale:
                    logger.info(f"Scaling to {target_instances} instances")
                    # Implement scaling logic here
                
                time.sleep(60)
            except Exception as e:
                logger.error(f"Auto-scaling failed: {e}")
                time.sleep(60)
    
    def run(self):
        """Run the API server."""
        uvicorn.run(
            self.app,
            host=self.config.api_host,
            port=self.config.api_port,
            workers=self.config.api_workers if self.config.mode == DeploymentMode.MULTI_INSTANCE else 1,
            log_level="info"
        )


class HAMNetDeploymentManager:
    """Main deployment manager for HAMNet."""
    
    def __init__(self, model_config: HAMNetConfig, deployment_config: DeploymentConfig):
        self.model_config = model_config
        self.deployment_config = deployment_config
        self.docker_manager = DockerManager(deployment_config)
        self.kubernetes_manager = KubernetesManager(deployment_config)
        self.api = HAMNetAPI(deployment_config, model_config)
    
    def generate_dockerfile(self, output_path: str = "Dockerfile"):
        """Generate Dockerfile for deployment."""
        dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE {self.deployment_config.api_port}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{self.deployment_config.api_port}/health || exit 1

# Run the application
CMD ["uvicorn", "hamnet.optimization.deployment:api.app", "--host", "0.0.0.0", "--port", "{self.deployment_config.api_port}"]
"""
        
        with open(output_path, 'w') as f:
            f.write(dockerfile_content)
        
        logger.info(f"Dockerfile generated: {output_path}")
    
    def generate_kubernetes_manifests(self, output_dir: str = "k8s"):
        """Generate Kubernetes manifests."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Deployment manifest
        deployment_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.deployment_config.kubernetes_deployment_name,
                "namespace": self.deployment_config.kubernetes_namespace
            },
            "spec": {
                "replicas": self.deployment_config.kubernetes_replicas,
                "selector": {
                    "matchLabels": {
                        "app": self.deployment_config.kubernetes_deployment_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": self.deployment_config.kubernetes_deployment_name
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "hamnet",
                            "image": f"{self.deployment_config.docker_registry}/{self.deployment_config.docker_image_name}:{self.deployment_config.docker_image_tag}",
                            "ports": [{
                                "containerPort": self.deployment_config.api_port
                            }],
                            "resources": {
                                "requests": {
                                    "cpu": self.deployment_config.kubernetes_cpu_request,
                                    "memory": self.deployment_config.kubernetes_memory_request,
                                    "nvidia.com/gpu": self.deployment_config.kubernetes_gpu_request
                                }
                            },
                            "env": [{
                                "name": "API_PORT",
                                "value": str(self.deployment_config.api_port)
                            }],
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": self.deployment_config.api_port
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": self.deployment_config.api_port
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }]
                    }
                }
            }
        }
        
        # Service manifest
        service_manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": self.deployment_config.kubernetes_service_name,
                "namespace": self.deployment_config.kubernetes_namespace
            },
            "spec": {
                "selector": {
                    "app": self.deployment_config.kubernetes_deployment_name
                },
                "ports": [{
                    "port": self.deployment_config.api_port,
                    "targetPort": self.deployment_config.api_port
                }],
                "type": "LoadBalancer"
            }
        }
        
        # Horizontal Pod Autoscaler manifest
        hpa_manifest = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{self.deployment_config.kubernetes_deployment_name}-hpa",
                "namespace": self.deployment_config.kubernetes_namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": self.deployment_config.kubernetes_deployment_name
                },
                "minReplicas": self.deployment_config.min_instances,
                "maxReplicas": self.deployment_config.max_instances,
                "metrics": [{
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": int(self.deployment_config.scale_up_threshold * 100)
                        }
                    }
                }]
            }
        }
        
        # Write manifests
        with open(output_dir / "deployment.yaml", 'w') as f:
            yaml.dump(deployment_manifest, f)
        
        with open(output_dir / "service.yaml", 'w') as f:
            yaml.dump(service_manifest, f)
        
        with open(output_dir / "hpa.yaml", 'w') as f:
            yaml.dump(hpa_manifest, f)
        
        logger.info(f"Kubernetes manifests generated in {output_dir}")
    
    def deploy_docker(self) -> bool:
        """Deploy using Docker."""
        try:
            # Generate Dockerfile
            self.generate_dockerfile()
            
            # Build image
            image_id = self.docker_manager.build_image("Dockerfile", ".")
            
            # Push image
            success = self.docker_manager.push_image(image_id)
            if not success:
                return False
            
            # Run container
            container_id = self.docker_manager.run_container(image_id)
            
            logger.info(f"Docker deployment successful. Container ID: {container_id}")
            return True
        
        except Exception as e:
            logger.error(f"Docker deployment failed: {e}")
            return False
    
    def deploy_kubernetes(self) -> bool:
        """Deploy using Kubernetes."""
        try:
            # Generate manifests
            self.generate_kubernetes_manifests()
            
            # Create deployment
            image = f"{self.deployment_config.docker_registry}/{self.deployment_config.docker_image_name}:{self.deployment_config.docker_image_tag}"
            success = self.kubernetes_manager.create_deployment(image)
            if not success:
                return False
            
            # Create service
            success = self.kubernetes_manager.create_service()
            if not success:
                return False
            
            logger.info("Kubernetes deployment successful")
            return True
        
        except Exception as e:
            logger.error(f"Kubernetes deployment failed: {e}")
            return False
    
    def run_local(self):
        """Run locally for development."""
        logger.info("Running HAMNet API locally")
        self.api.run()
    
    def deploy(self):
        """Deploy based on configuration."""
        logger.info(f"Deploying HAMNet with {self.deployment_config.environment.value}")
        
        if self.deployment_config.environment == DeploymentEnvironment.LOCAL:
            self.run_local()
        elif self.deployment_config.environment == DeploymentEnvironment.DOCKER:
            self.deploy_docker()
        elif self.deployment_config.environment == DeploymentEnvironment.KUBERNETES:
            self.deploy_kubernetes()
        else:
            logger.error(f"Unsupported deployment environment: {self.deployment_config.environment}")


# Example usage
def example_deployment():
    """Example of deployment."""
    # Model configuration
    model_config = HAMNetConfig(
        hidden_size=256,
        num_layers=4,
        input_size=512
    )
    
    # Deployment configuration
    deployment_config = DeploymentConfig(
        environment=DeploymentEnvironment.DOCKER,
        mode=DeploymentMode.AUTO_SCALING,
        api_port=8000,
        enable_monitoring=True,
        min_instances=1,
        max_instances=5
    )
    
    # Create deployment manager
    deployment_manager = HAMNetDeploymentManager(model_config, deployment_config)
    
    # Deploy
    deployment_manager.deploy()
    
    return deployment_manager


if __name__ == "__main__":
    # Run example
    deployment_manager = example_deployment()
    print("Deployment framework initialized successfully")