"""
Monitoring, Logging, and A/B Testing Framework for HAMNet

This module provides comprehensive monitoring, logging, and A/B testing capabilities
for tracking model performance and conducting experiments.
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
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import uuid
import hashlib
from collections import defaultdict, deque
import sqlite3
import redis
from elasticsearch import Elasticsearch
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ..models.hamnet import HAMNet, HAMNetConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class ExperimentStatus(Enum):
    """Experiment statuses."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ABTestType(Enum):
    """A/B test types."""
    MODEL_A_B = "model_a_b"
    HYPERPARAMETER = "hyperparameter"
    ARCHITECTURE = "architecture"
    PREPROCESSING = "preprocessing"
    INFERENCE = "inference"


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and logging."""
    # Logging settings
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "hamnet.log"
    log_rotation: str = "1d"
    log_retention: str = "30d"
    
    # Metrics settings
    enable_metrics: bool = True
    metrics_port: int = 9090
    metrics_interval: int = 10  # seconds
    metrics_retention: int = 30  # days
    
    # Database settings
    database_type: str = "sqlite"  # "sqlite", "postgresql", "mysql"
    database_url: str = "sqlite:///hamnet.db"
    
    # Redis settings
    enable_redis: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Elasticsearch settings
    enable_elasticsearch: bool = False
    elasticsearch_host: str = "localhost"
    elasticsearch_port: int = 9200
    elasticsearch_index: str = "hamnet-logs"
    
    # Alerting settings
    enable_alerting: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["email", "slack"])
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "error_rate": 0.05,
        "latency_p95": 1.0,
        "memory_usage": 0.9,
        "gpu_usage": 0.9
    })
    
    # A/B testing settings
    enable_ab_testing: bool = True
    ab_test_db: str = "sqlite:///ab_tests.db"
    traffic_split: float = 0.5  # 50/50 split by default


class Metric:
    """Base metric class."""
    
    def __init__(self, name: str, metric_type: MetricType, description: str = ""):
        self.name = name
        self.type = metric_type
        self.description = description
        self.labels = {}
        self.registry = CollectorRegistry()
        
        if metric_type == MetricType.COUNTER:
            self.metric = Counter(name, description, registry=self.registry)
        elif metric_type == MetricType.GAUGE:
            self.metric = Gauge(name, description, registry=self.registry)
        elif metric_type == MetricType.HISTOGRAM:
            self.metric = Histogram(name, description, registry=self.registry)
        else:
            self.metric = Gauge(name, description, registry=self.registry)
    
    def record(self, value: float, **labels):
        """Record metric value."""
        if labels:
            self.metric.labels(**labels).observe(value)
        else:
            self.metric.observe(value)


class HAMNetMetrics:
    """HAMNet-specific metrics collection."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics = {}
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup HAMNet metrics."""
        # Training metrics
        self.metrics["training_loss"] = Metric(
            "hamnet_training_loss",
            MetricType.HISTOGRAM,
            "Training loss per epoch"
        )
        
        self.metrics["validation_loss"] = Metric(
            "hamnet_validation_loss",
            MetricType.HISTOGRAM,
            "Validation loss per epoch"
        )
        
        self.metrics["learning_rate"] = Metric(
            "hamnet_learning_rate",
            MetricType.GAUGE,
            "Current learning rate"
        )
        
        # Inference metrics
        self.metrics["inference_latency"] = Metric(
            "hamnet_inference_latency_seconds",
            MetricType.HISTOGRAM,
            "Inference latency in seconds"
        )
        
        self.metrics["inference_throughput"] = Metric(
            "hamnet_inference_throughput",
            MetricType.GAUGE,
            "Inference throughput per second"
        )
        
        self.metrics["prediction_accuracy"] = Metric(
            "hamnet_prediction_accuracy",
            MetricType.HISTOGRAM,
            "Prediction accuracy"
        )
        
        # System metrics
        self.metrics["gpu_usage"] = Metric(
            "hamnet_gpu_usage_percent",
            MetricType.GAUGE,
            "GPU usage percentage"
        )
        
        self.metrics["gpu_memory"] = Metric(
            "hamnet_gpu_memory_usage_bytes",
            MetricType.GAUGE,
            "GPU memory usage in bytes"
        )
        
        self.metrics["cpu_usage"] = Metric(
            "hamnet_cpu_usage_percent",
            MetricType.GAUGE,
            "CPU usage percentage"
        )
        
        self.metrics["memory_usage"] = Metric(
            "hamnet_memory_usage_bytes",
            MetricType.GAUGE,
            "Memory usage in bytes"
        )
    
    def record_training_metrics(self, epoch: int, train_loss: float, val_loss: float, lr: float):
        """Record training metrics."""
        self.metrics["training_loss"].record(train_loss, epoch=str(epoch))
        self.metrics["validation_loss"].record(val_loss, epoch=str(epoch))
        self.metrics["learning_rate"].record(lr)
    
    def record_inference_metrics(self, latency: float, accuracy: float, batch_size: int):
        """Record inference metrics."""
        self.metrics["inference_latency"].record(latency, batch_size=str(batch_size))
        self.metrics["prediction_accuracy"].record(accuracy)
        self.metrics["inference_throughput"].record(batch_size / latency)
    
    def record_system_metrics(self, cpu_usage: float, memory_usage: float, 
                             gpu_usage: float = 0, gpu_memory: float = 0):
        """Record system metrics."""
        self.metrics["cpu_usage"].record(cpu_usage)
        self.metrics["memory_usage"].record(memory_usage)
        self.metrics["gpu_usage"].record(gpu_usage)
        self.metrics["gpu_memory"].record(gpu_memory)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        summary = {}
        
        for name, metric in self.metrics.items():
            try:
                samples = list(metric.metric.collect())[0].samples
                if samples:
                    values = [sample.value for sample in samples]
                    summary[name] = {
                        "count": len(values),
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values)
                    }
            except:
                pass
        
        return summary


class StructuredLogger:
    """Structured logger for HAMNet."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger("hamnet")
        self._setup_logger()
        
        # Setup additional logging backends
        self.redis_client = None
        self.elasticsearch_client = None
        
        if config.enable_redis:
            self.redis_client = redis.Redis(
                host=config.redis_host, port=config.redis_port, db=config.redis_db
            )
        
        if config.enable_elasticsearch:
            self.elasticsearch_client = Elasticsearch([{
                "host": config.elasticsearch_host,
                "port": config.elasticsearch_port
            }])
    
    def _setup_logger(self):
        """Setup logger configuration."""
        self.logger.setLevel(getattr(logging, self.config.log_level.value.upper()))
        
        # Create formatter
        formatter = logging.Formatter(self.config.log_format)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def log(self, level: LogLevel, message: str, **kwargs):
        """Log structured message."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level.value,
            "message": message,
            **kwargs
        }
        
        # Log to standard logger
        getattr(self.logger, level.value.lower())(json.dumps(log_entry))
        
        # Log to Redis
        if self.redis_client:
            self.redis_client.lpush(
                "hamnet_logs",
                json.dumps(log_entry)
            )
        
        # Log to Elasticsearch
        if self.elasticsearch_client:
            self.elasticsearch_client.index(
                index=self.config.elasticsearch_index,
                body=log_entry
            )
    
    def log_training_event(self, event_type: str, epoch: int, metrics: Dict[str, float]):
        """Log training event."""
        self.log(
            LogLevel.INFO,
            f"Training event: {event_type}",
            event_type=event_type,
            epoch=epoch,
            metrics=metrics
        )
    
    def log_inference_event(self, request_id: str, latency: float, accuracy: float):
        """Log inference event."""
        self.log(
            LogLevel.INFO,
            "Inference completed",
            request_id=request_id,
            latency=latency,
            accuracy=accuracy
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any]):
        """Log error event."""
        self.log(
            LogLevel.ERROR,
            f"Error occurred: {str(error)}",
            error_type=type(error).__name__,
            context=context
        )


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alert_history = deque(maxlen=1000)
        self.alert_cooldown = {}
        self.cooldown_period = 300  # 5 minutes
    
    def check_alerts(self, metrics: Dict[str, float]):
        """Check alert conditions."""
        alerts = []
        
        for metric_name, threshold in self.config.alert_thresholds.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                
                if value > threshold:
                    alert_key = f"{metric_name}_{int(value)}"
                    
                    # Check cooldown
                    if alert_key in self.alert_cooldown:
                        if time.time() - self.alert_cooldown[alert_key] < self.cooldown_period:
                            continue
                    
                    # Create alert
                    alert = {
                        "id": str(uuid.uuid4()),
                        "metric": metric_name,
                        "value": value,
                        "threshold": threshold,
                        "timestamp": datetime.utcnow().isoformat(),
                        "severity": "warning" if value < threshold * 1.5 else "critical"
                    }
                    
                    alerts.append(alert)
                    self.alert_history.append(alert)
                    self.alert_cooldown[alert_key] = time.time()
        
        # Send alerts
        for alert in alerts:
            self._send_alert(alert)
        
        return alerts
    
    def _send_alert(self, alert: Dict[str, Any]):
        """Send alert notification."""
        logger.warning(f"Alert triggered: {alert}")
        
        # Send to configured channels
        if "email" in self.config.alert_channels:
            self._send_email_alert(alert)
        
        if "slack" in self.config.alert_channels:
            self._send_slack_alert(alert)
    
    def _send_email_alert(self, alert: Dict[str, Any]):
        """Send email alert (placeholder)."""
        logger.info(f"Email alert sent for {alert['metric']}")
    
    def _send_slack_alert(self, alert: Dict[str, Any]):
        """Send Slack alert (placeholder)."""
        logger.info(f"Slack alert sent for {alert['metric']}")


class ABTest:
    """A/B test implementation."""
    
    def __init__(self, test_id: str, name: str, test_type: ABTestType, 
                 config_a: Dict[str, Any], config_b: Dict[str, Any]):
        self.test_id = test_id
        self.name = name
        self.test_type = test_type
        self.config_a = config_a
        self.config_b = config_b
        self.status = ExperimentStatus.DRAFT
        self.traffic_split = 0.5
        self.results = {
            "group_a": [],
            "group_b": []
        }
        self.start_time = None
        self.end_time = None
    
    def assign_group(self, user_id: str) -> str:
        """Assign user to test group."""
        # Use hash for consistent assignment
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        
        if (hash_value % 100) < (self.traffic_split * 100):
            return "group_a"
        else:
            return "group_b"
    
    def record_result(self, group: str, result: Dict[str, Any]):
        """Record test result."""
        if group in ["group_a", "group_b"]:
            self.results[group].append(result)
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate test statistics."""
        stats = {}
        
        for group in ["group_a", "group_b"]:
            if not self.results[group]:
                continue
            
            results = self.results[group]
            
            # Extract metric values
            metrics = defaultdict(list)
            for result in results:
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        metrics[key].append(value)
            
            # Calculate statistics for each metric
            group_stats = {}
            for metric_name, values in metrics.items():
                group_stats[metric_name] = {
                    "count": len(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values)
                }
            
            stats[group] = group_stats
        
        # Calculate statistical significance
        if len(stats) == 2:
            stats["significance"] = self._calculate_significance()
        
        return stats
    
    def _calculate_significance(self) -> Dict[str, Any]:
        """Calculate statistical significance between groups."""
        significance = {}
        
        # Compare metrics between groups
        metrics_a = defaultdict(list)
        metrics_b = defaultdict(list)
        
        for result in self.results["group_a"]:
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    metrics_a[key].append(value)
        
        for result in self.results["group_b"]:
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    metrics_b[key].append(value)
        
        # Perform statistical tests
        for metric_name in set(metrics_a.keys()) & set(metrics_b.keys()):
            values_a = metrics_a[metric_name]
            values_b = metrics_b[metric_name]
            
            if len(values_a) > 1 and len(values_b) > 1:
                # T-test
                t_stat, p_value = stats.ttest_ind(values_a, values_b)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((len(values_a) - 1) * np.std(values_a) ** 2 + 
                                   (len(values_b) - 1) * np.std(values_b) ** 2) / (len(values_a) + len(values_b) - 2)
                effect_size = (np.mean(values_a) - np.mean(values_b)) / pooled_std
                
                significance[metric_name] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "effect_size": effect_size,
                    "significant": p_value < 0.05
                }
        
        return significance
    
    def start(self):
        """Start the test."""
        self.status = ExperimentStatus.RUNNING
        self.start_time = datetime.utcnow()
        logger.info(f"A/B test {self.test_id} started")
    
    def stop(self):
        """Stop the test."""
        self.status = ExperimentStatus.COMPLETED
        self.end_time = datetime.utcnow()
        logger.info(f"A/B test {self.test_id} completed")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary."""
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        
        return {
            "test_id": self.test_id,
            "name": self.name,
            "type": self.test_type.value,
            "status": self.status.value,
            "traffic_split": self.traffic_split,
            "duration_seconds": duration,
            "total_samples": len(self.results["group_a"]) + len(self.results["group_b"]),
            "group_a_samples": len(self.results["group_a"]),
            "group_b_samples": len(self.results["group_b"]),
            "statistics": self.calculate_statistics()
        }


class ABTestManager:
    """Manages A/B tests."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.tests = {}
        self._setup_database()
    
    def _setup_database(self):
        """Setup database for A/B tests."""
        if self.config.ab_test_db.startswith("sqlite"):
            self.conn = sqlite3.connect(self.config.ab_test_db.replace("sqlite:///", ""))
            self._create_tables()
    
    def _create_tables(self):
        """Create database tables."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_tests (
                test_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                config_a TEXT NOT NULL,
                config_b TEXT NOT NULL,
                traffic_split REAL NOT NULL,
                start_time TEXT,
                end_time TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                group_name TEXT NOT NULL,
                user_id TEXT NOT NULL,
                result_data TEXT NOT NULL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (test_id) REFERENCES ab_tests (test_id)
            )
        """)
        
        self.conn.commit()
    
    def create_test(self, name: str, test_type: ABTestType, 
                   config_a: Dict[str, Any], config_b: Dict[str, Any]) -> ABTest:
        """Create new A/B test."""
        test_id = str(uuid.uuid4())
        test = ABTest(test_id, name, test_type, config_a, config_b)
        
        # Save to database
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO ab_tests (test_id, name, type, status, config_a, config_b, traffic_split) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (test_id, name, test_type.value, test.status.value, 
             json.dumps(config_a), json.dumps(config_b), test.traffic_split)
        )
        self.conn.commit()
        
        self.tests[test_id] = test
        return test
    
    def get_test(self, test_id: str) -> Optional[ABTest]:
        """Get test by ID."""
        if test_id in self.tests:
            return self.tests[test_id]
        
        # Load from database
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM ab_tests WHERE test_id = ?", (test_id,))
        row = cursor.fetchone()
        
        if row:
            test = ABTest(
                row[0], row[1], ABTestType(row[2]), 
                json.loads(row[4]), json.loads(row[5])
            )
            test.status = ExperimentStatus(row[3])
            test.traffic_split = row[6]
            
            # Load results
            cursor.execute(
                "SELECT group_name, user_id, result_data FROM ab_test_results WHERE test_id = ?",
                (test_id,)
            )
            for result_row in cursor.fetchall():
                test.record_result(result_row[0], json.loads(result_row[2]))
            
            self.tests[test_id] = test
            return test
        
        return None
    
    def list_tests(self) -> List[Dict[str, Any]]:
        """List all tests."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT test_id, name, type, status FROM ab_tests")
        rows = cursor.fetchall()
        
        return [
            {
                "test_id": row[0],
                "name": row[1],
                "type": row[2],
                "status": row[3]
            }
            for row in rows
        ]
    
    def record_result(self, test_id: str, user_id: str, result: Dict[str, Any]):
        """Record test result."""
        test = self.get_test(test_id)
        if not test:
            return
        
        group = test.assign_group(user_id)
        test.record_result(group, result)
        
        # Save to database
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO ab_test_results (test_id, group_name, user_id, result_data) VALUES (?, ?, ?, ?)",
            (test_id, group, user_id, json.dumps(result))
        )
        self.conn.commit()
    
    def get_test_statistics(self, test_id: str) -> Dict[str, Any]:
        """Get test statistics."""
        test = self.get_test(test_id)
        if not test:
            return {}
        
        return test.get_summary()


class HAMNetMonitoring:
    """Main monitoring framework for HAMNet."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics = HAMNetMetrics(config)
        self.logger = StructuredLogger(config)
        self.alert_manager = AlertManager(config)
        self.ab_test_manager = ABTestManager(config)
        
        # Background tasks
        self.running = False
        self.metrics_thread = None
        self.alert_thread = None
    
    def start_monitoring(self):
        """Start monitoring."""
        self.running = True
        
        # Start metrics collection
        if self.config.enable_metrics:
            self.metrics_thread = threading.Thread(target=self._collect_metrics)
            self.metrics_thread.daemon = True
            self.metrics_thread.start()
        
        # Start alert checking
        if self.config.enable_alerting:
            self.alert_thread = threading.Thread(target=self._check_alerts)
            self.alert_thread.daemon = True
            self.alert_thread.start()
        
        logger.info("HAMNet monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
        
        if self.metrics_thread:
            self.metrics_thread.join(timeout=5)
        
        if self.alert_thread:
            self.alert_thread.join(timeout=5)
        
        logger.info("HAMNet monitoring stopped")
    
    def _collect_metrics(self):
        """Collect metrics periodically."""
        import psutil
        import GPUtil
        
        while self.running:
            try:
                # Collect system metrics
                cpu_usage = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # GPU metrics
                gpu_usage = 0
                gpu_memory = 0
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        gpu_usage = gpu.load * 100
                        gpu_memory = gpu.memoryUsed * 1024 * 1024
                except:
                    pass
                
                # Record metrics
                self.metrics.record_system_metrics(
                    cpu_usage, memory.used, gpu_usage, gpu_memory
                )
                
                # Check alerts
                if self.config.enable_alerting:
                    current_metrics = {
                        "cpu_usage": cpu_usage / 100,
                        "memory_usage": memory.percent / 100,
                        "gpu_usage": gpu_usage / 100
                    }
                    self.alert_manager.check_alerts(current_metrics)
                
                time.sleep(self.config.metrics_interval)
            
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                time.sleep(self.config.metrics_interval)
    
    def _check_alerts(self):
        """Check alert conditions periodically."""
        while self.running:
            try:
                # Get current metrics
                metrics_summary = self.metrics.get_metrics_summary()
                
                # Convert to alert format
                alert_metrics = {}
                for metric_name, summary in metrics_summary.items():
                    if "mean" in summary:
                        alert_metrics[metric_name] = summary["mean"]
                
                # Check alerts
                self.alert_manager.check_alerts(alert_metrics)
                
                time.sleep(60)  # Check every minute
            
            except Exception as e:
                logger.error(f"Error checking alerts: {e}")
                time.sleep(60)
    
    def log_training_progress(self, epoch: int, train_loss: float, val_loss: float, 
                             learning_rate: float, additional_metrics: Dict[str, float] = None):
        """Log training progress."""
        # Record metrics
        self.metrics.record_training_metrics(epoch, train_loss, val_loss, learning_rate)
        
        # Log event
        metrics_data = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": learning_rate
        }
        
        if additional_metrics:
            metrics_data.update(additional_metrics)
        
        self.logger.log_training_event("epoch_completed", epoch, metrics_data)
    
    def log_inference_request(self, request_id: str, input_data: Any, 
                             output_data: Any, latency: float, accuracy: float):
        """Log inference request."""
        # Record metrics
        batch_size = 1
        if hasattr(input_data, '__len__'):
            batch_size = len(input_data)
        
        self.metrics.record_inference_metrics(latency, accuracy, batch_size)
        
        # Log event
        self.logger.log_inference_event(request_id, latency, accuracy)
    
    def create_ab_test(self, name: str, test_type: ABTestType, 
                      config_a: Dict[str, Any], config_b: Dict[str, Any]) -> ABTest:
        """Create A/B test."""
        test = self.ab_test_manager.create_test(name, test_type, config_a, config_b)
        
        self.logger.log(
            LogLevel.INFO,
            f"A/B test created: {name}",
            test_id=test.test_id,
            test_type=test_type.value
        )
        
        return test
    
    def record_ab_test_result(self, test_id: str, user_id: str, result: Dict[str, Any]):
        """Record A/B test result."""
        self.ab_test_manager.record_result(test_id, user_id, result)
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get monitoring dashboard data."""
        dashboard = {
            "metrics_summary": self.metrics.get_metrics_summary(),
            "recent_alerts": list(self.alert_manager.alert_history)[-10:],
            "active_tests": self.ab_test_manager.list_tests(),
            "system_health": self._get_system_health()
        }
        
        return dashboard
    
    def _get_system_health(self) -> Dict[str, str]:
        """Get system health status."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            health_status = "healthy"
            
            if cpu_percent > 90 or memory.percent > 90:
                health_status = "critical"
            elif cpu_percent > 70 or memory.percent > 70:
                health_status = "warning"
            
            return {
                "status": health_status,
                "cpu_usage": f"{cpu_percent:.1f}%",
                "memory_usage": f"{memory.percent:.1f}%",
                "disk_usage": f"{psutil.disk_usage('/').percent:.1f}%"
            }
        
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {"status": "unknown", "error": str(e)}


# Example usage
def example_monitoring():
    """Example of monitoring usage."""
    # Configuration
    config = MonitoringConfig(
        log_level=LogLevel.INFO,
        enable_metrics=True,
        enable_alerting=True,
        enable_ab_testing=True
    )
    
    # Create monitoring framework
    monitoring = HAMNetMonitoring(config)
    
    # Start monitoring
    monitoring.start_monitoring()
    
    # Log some events
    monitoring.log_training_progress(1, 0.5, 0.4, 0.001)
    monitoring.log_inference_request("req_001", [1, 2, 3], [0.8, 0.2], 0.1, 0.95)
    
    # Create A/B test
    test = monitoring.create_ab_test(
        "Model Comparison",
        ABTestType.MODEL_A_B,
        {"model": "hamnet_v1", "learning_rate": 0.001},
        {"model": "hamnet_v2", "learning_rate": 0.0001}
    )
    
    # Start test
    test.start()
    
    # Record some results
    monitoring.record_ab_test_result(test.test_id, "user_001", {"accuracy": 0.95, "latency": 0.1})
    monitoring.record_ab_test_result(test.test_id, "user_002", {"accuracy": 0.92, "latency": 0.08})
    
    # Get dashboard
    dashboard = monitoring.get_monitoring_dashboard()
    print(f"Dashboard: {dashboard}")
    
    # Stop monitoring
    monitoring.stop_monitoring()
    
    return monitoring


if __name__ == "__main__":
    # Run example
    monitoring = example_monitoring()
    print("Monitoring framework initialized successfully")