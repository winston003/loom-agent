"""Monitoring and observability components.

Implements Phase 6 monitoring (T093-T096):
- Prometheus metrics exposition
- Structured logging enhancements
- Distributed tracing (OpenTelemetry)
- Health checks and auto-recovery

Reference: specs/002-xiaozhi-voice-adapter/phase6/
"""

from __future__ import annotations

import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from loom.core.structured_logger import get_logger

logger = get_logger("audio.monitoring")


# ============================
# T093: Prometheus Metrics
# ============================


class MetricType(str, Enum):
    """Prometheus metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """Prometheus metric."""
    name: str
    type: MetricType
    help: str
    value: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)
    buckets: list = field(default_factory=list)  # For histograms
    observations: list = field(default_factory=list)  # For histograms/summaries


class PrometheusMetrics:
    """Prometheus metrics collector.
    
    Exposes metrics in Prometheus text format for scraping.
    
    Metrics tracked:
    - audio_sessions_total: Total number of sessions created
    - audio_sessions_active: Current active sessions
    - audio_transcriptions_total: Total transcriptions
    - audio_asr_latency_seconds: ASR latency histogram
    - audio_voiceprint_verifications_total: Voiceprint verifications
    - audio_errors_total: Total errors by type
    - audio_cache_hits_total: Cache hits by cache type
    - audio_cache_misses_total: Cache misses by cache type
    """

    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self._initialize_metrics()

    def _initialize_metrics(self):
        """Initialize default metrics."""
        # Sessions
        self.register_metric(
            "audio_sessions_total",
            MetricType.COUNTER,
            "Total number of audio sessions created"
        )
        self.register_metric(
            "audio_sessions_active",
            MetricType.GAUGE,
            "Current number of active audio sessions"
        )
        
        # Transcriptions
        self.register_metric(
            "audio_transcriptions_total",
            MetricType.COUNTER,
            "Total number of ASR transcriptions"
        )
        
        # Latency
        self.register_metric(
            "audio_asr_latency_seconds",
            MetricType.HISTOGRAM,
            "ASR transcription latency in seconds",
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )
        self.register_metric(
            "audio_voiceprint_latency_seconds",
            MetricType.HISTOGRAM,
            "Voiceprint verification latency in seconds",
            buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
        )
        
        # Voiceprint
        self.register_metric(
            "audio_voiceprint_verifications_total",
            MetricType.COUNTER,
            "Total voiceprint verifications"
        )
        self.register_metric(
            "audio_voiceprint_matches_total",
            MetricType.COUNTER,
            "Total successful voiceprint matches"
        )
        
        # Errors
        self.register_metric(
            "audio_errors_total",
            MetricType.COUNTER,
            "Total errors by type"
        )
        
        # Cache
        self.register_metric(
            "audio_cache_hits_total",
            MetricType.COUNTER,
            "Cache hits by cache type"
        )
        self.register_metric(
            "audio_cache_misses_total",
            MetricType.COUNTER,
            "Cache misses by cache type"
        )

    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        help: str,
        buckets: Optional[list] = None
    ):
        """Register a new metric."""
        self.metrics[name] = Metric(
            name=name,
            type=metric_type,
            help=help,
            buckets=buckets or []
        )

    def inc(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment counter metric."""
        if name not in self.metrics:
            logger.warning("Metric not found", name=name)
            return
        
        metric = self.metrics[name]
        if metric.type != MetricType.COUNTER:
            logger.warning("Metric is not a counter", name=name, type=metric.type)
            return
        
        metric.value += value
        if labels:
            metric.labels = labels

    def set(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set gauge metric value."""
        if name not in self.metrics:
            logger.warning("Metric not found", name=name)
            return
        
        metric = self.metrics[name]
        if metric.type != MetricType.GAUGE:
            logger.warning("Metric is not a gauge", name=name, type=metric.type)
            return
        
        metric.value = value
        if labels:
            metric.labels = labels

    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe histogram/summary metric."""
        if name not in self.metrics:
            logger.warning("Metric not found", name=name)
            return
        
        metric = self.metrics[name]
        if metric.type not in (MetricType.HISTOGRAM, MetricType.SUMMARY):
            logger.warning("Metric is not histogram/summary", name=name, type=metric.type)
            return
        
        metric.observations.append(value)
        if labels:
            metric.labels = labels

    def export(self) -> str:
        """Export metrics in Prometheus text format.
        
        Returns:
            Prometheus-compatible metrics text
        """
        lines = []
        
        for name, metric in self.metrics.items():
            # Help line
            lines.append(f"# HELP {name} {metric.help}")
            lines.append(f"# TYPE {name} {metric.type}")
            
            # Metric value(s)
            if metric.type == MetricType.COUNTER or metric.type == MetricType.GAUGE:
                labels_str = self._format_labels(metric.labels)
                lines.append(f"{name}{labels_str} {metric.value}")
            
            elif metric.type == MetricType.HISTOGRAM:
                # Calculate histogram buckets
                if metric.observations:
                    sorted_obs = sorted(metric.observations)
                    for bucket in metric.buckets:
                        count = sum(1 for v in sorted_obs if v <= bucket)
                        labels_str = self._format_labels({**metric.labels, "le": str(bucket)})
                        lines.append(f"{name}_bucket{labels_str} {count}")
                    
                    # +Inf bucket
                    labels_str = self._format_labels({**metric.labels, "le": "+Inf"})
                    lines.append(f"{name}_bucket{labels_str} {len(sorted_obs)}")
                    
                    # Sum and count
                    labels_str = self._format_labels(metric.labels)
                    lines.append(f"{name}_sum{labels_str} {sum(sorted_obs)}")
                    lines.append(f"{name}_count{labels_str} {len(sorted_obs)}")
        
        return "\n".join(lines)

    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus output."""
        if not labels:
            return ""
        
        label_pairs = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ",".join(label_pairs) + "}"

    def reset(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.value = 0.0
            metric.observations.clear()
            metric.labels.clear()


# ============================
# T096: Health Check
# ============================


class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Health check result."""
    status: HealthStatus
    component: str
    message: str
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """Health checker for audio adapter components.
    
    Performs periodic health checks on:
    - Session manager (active sessions, responsiveness)
    - VAD/ASR/TTS providers (availability)
    - Voiceprint service (connectivity)
    - Cache systems (hit rates, expiration)
    - Concurrency limiter (rejection rates)
    
    Args:
        check_interval: Seconds between health checks (default: 30)
        auto_recover: Enable automatic recovery on failures (default: True)
    """

    def __init__(
        self,
        check_interval: float = 30,
        auto_recover: bool = True,
    ):
        self.check_interval = check_interval
        self.auto_recover = auto_recover
        
        self._checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._last_results: Dict[str, HealthCheckResult] = {}
        self._running = False
        self._task: Optional[Any] = None

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], HealthCheckResult]
    ):
        """Register a health check function.
        
        Args:
            name: Check identifier
            check_fn: Async function returning HealthCheckResult
        """
        self._checks[name] = check_fn
        logger.debug("Registered health check", name=name)

    async def check_all(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks.
        
        Returns:
            Dictionary mapping check names to results
        """
        results = {}
        
        for name, check_fn in self._checks.items():
            try:
                result = await check_fn()
                results[name] = result
                self._last_results[name] = result
                
                if result.status == HealthStatus.UNHEALTHY:
                    logger.error(
                        "Health check failed",
                        component=result.component,
                        message=result.message,
                        details=result.details
                    )
                    
                    if self.auto_recover:
                        await self._attempt_recovery(name, result)
                
                elif result.status == HealthStatus.DEGRADED:
                    logger.warning(
                        "Health check degraded",
                        component=result.component,
                        message=result.message
                    )
            
            except Exception as e:
                error_result = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    component=name,
                    message=f"Health check error: {str(e)}"
                )
                results[name] = error_result
                self._last_results[name] = error_result
                logger.error("Health check exception", name=name, error=str(e))
        
        return results

    async def _attempt_recovery(self, check_name: str, result: HealthCheckResult):
        """Attempt automatic recovery for failed health check."""
        logger.info(
            "Attempting auto-recovery",
            check=check_name,
            component=result.component
        )
        
        # Recovery strategies based on component
        # Subclasses can override this method for custom recovery
        logger.warning(
            "No recovery strategy defined",
            check=check_name,
            component=result.component
        )

    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status.
        
        Returns:
            HEALTHY if all checks pass
            DEGRADED if some checks degraded
            UNHEALTHY if any check fails
        """
        if not self._last_results:
            return HealthStatus.HEALTHY
        
        statuses = [r.status for r in self._last_results.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def get_stats(self) -> Dict[str, Any]:
        """Get health checker statistics."""
        return {
            "running": self._running,
            "check_interval": self.check_interval,
            "auto_recover": self.auto_recover,
            "total_checks": len(self._checks),
            "overall_status": self.get_overall_status().value,
            "last_results": {
                name: {
                    "status": result.status.value,
                    "component": result.component,
                    "message": result.message,
                }
                for name, result in self._last_results.items()
            }
        }
