"""
Resilience components for audio adapter.

Provides retry mechanisms, circuit breakers, and fallback strategies
for production-grade fault tolerance.

Components:
- RetryPolicy: Configurable retry with exponential backoff
- CircuitBreaker: Automatic failure detection and recovery
- FallbackStrategy: Degradation strategies for unavailable services
- RecoveryManager: Coordinated recovery workflows
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from functools import wraps
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ===== Retry Mechanism =====

class RetryStrategy(Enum):
    """Retry strategy types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    CONSTANT_BACKOFF = "constant_backoff"


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_attempts: int = 3
    initial_delay: float = 0.1  # seconds
    max_delay: float = 10.0  # seconds
    backoff_factor: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    
    # Exceptions to retry
    retryable_exceptions: tuple = (
        ConnectionError,
        TimeoutError,
        OSError,
    )
    
    # Exceptions to NOT retry (fail immediately)
    fatal_exceptions: tuple = (
        ValueError,
        TypeError,
        KeyError,
    )


class RetryPolicy:
    """
    Retry policy with exponential backoff.
    
    Usage:
        retry = RetryPolicy(max_attempts=3, initial_delay=0.1)
        
        result = await retry.execute(
            some_async_function,
            arg1, arg2,
            kwarg1=value1
        )
    
    Features:
    - Exponential backoff with jitter
    - Configurable retry strategies
    - Exception filtering
    - Attempt tracking and stats
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.stats = {
            'total_attempts': 0,
            'successful_retries': 0,
            'failed_retries': 0,
            'total_delay': 0.0,
        }
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.initial_delay * (self.config.backoff_factor ** attempt)
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.initial_delay * (attempt + 1)
        else:  # CONSTANT_BACKOFF
            delay = self.config.initial_delay
        
        # Cap at max_delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter (±10%)
        import random
        jitter = delay * 0.1 * (2 * random.random() - 1)
        return delay + jitter
    
    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
        
        Returns:
            Function result on success
        
        Raises:
            Last exception if all retries exhausted
        """
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            self.stats['total_attempts'] += 1
            
            try:
                result = await func(*args, **kwargs)
                
                if attempt > 0:
                    self.stats['successful_retries'] += 1
                    logger.info(
                        f"Retry succeeded on attempt {attempt + 1}/{self.config.max_attempts}",
                        extra={'function': func.__name__, 'attempt': attempt + 1}
                    )
                
                return result
            
            except self.config.fatal_exceptions as e:
                # Fatal exception - fail immediately
                logger.error(
                    f"Fatal exception in {func.__name__}: {e}",
                    extra={'exception': str(e)}
                )
                raise
            
            except self.config.retryable_exceptions as e:
                last_exception = e
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    self.stats['total_delay'] += delay
                    
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{self.config.max_attempts} "
                        f"failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s",
                        extra={
                            'function': func.__name__,
                            'attempt': attempt + 1,
                            'delay': delay,
                            'exception': str(e)
                        }
                    )
                    
                    await asyncio.sleep(delay)
                else:
                    self.stats['failed_retries'] += 1
                    logger.error(
                        f"All retry attempts exhausted for {func.__name__}",
                        extra={
                            'function': func.__name__,
                            'max_attempts': self.config.max_attempts,
                            'exception': str(e)
                        }
                    )
        
        # All retries exhausted
        raise last_exception
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics."""
        return {
            'total_attempts': self.stats['total_attempts'],
            'successful_retries': self.stats['successful_retries'],
            'failed_retries': self.stats['failed_retries'],
            'total_delay': round(self.stats['total_delay'], 2),
            'config': {
                'max_attempts': self.config.max_attempts,
                'strategy': self.config.strategy.value,
                'initial_delay': self.config.initial_delay,
                'max_delay': self.config.max_delay,
            }
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'total_attempts': 0,
            'successful_retries': 0,
            'failed_retries': 0,
            'total_delay': 0.0,
        }


def with_retry(config: Optional[RetryConfig] = None):
    """
    Decorator for adding retry logic to async functions.
    
    Usage:
        @with_retry(RetryConfig(max_attempts=3))
        async def fetch_data():
            # ... network call
            pass
    """
    retry_policy = RetryPolicy(config)
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_policy.execute(func, *args, **kwargs)
        
        # Attach stats to wrapped function
        wrapper.get_retry_stats = retry_policy.get_stats
        return wrapper
    
    return decorator


# ===== Circuit Breaker =====

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5  # failures before opening
    success_threshold: int = 2  # successes to close from half-open
    timeout: float = 30.0  # seconds before trying half-open
    
    # Sliding window for failure tracking
    window_size: int = 10  # track last N requests
    error_rate_threshold: float = 0.5  # 50% error rate


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Usage:
        breaker = CircuitBreaker(failure_threshold=5, timeout=30.0)
        
        async with breaker:
            result = await some_operation()
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, reject all requests
    - HALF_OPEN: Testing recovery, allow limited requests
    
    Features:
    - Automatic state transitions
    - Sliding window failure tracking
    - Error rate monitoring
    - Recovery testing
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        
        # Failure tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        
        # Sliding window for error rate
        self.recent_results: List[bool] = []  # True = success, False = failure
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rejected_requests': 0,
            'state_transitions': 0,
            'current_state': self.state.value,
        }
    
    def _update_window(self, success: bool):
        """Update sliding window with new result."""
        self.recent_results.append(success)
        
        # Keep window size limited
        if len(self.recent_results) > self.config.window_size:
            self.recent_results.pop(0)
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate from sliding window."""
        if not self.recent_results:
            return 0.0
        
        failures = sum(1 for r in self.recent_results if not r)
        return failures / len(self.recent_results)
    
    def _should_attempt_reset(self) -> bool:
        """Check if should attempt transition to HALF_OPEN."""
        if self.state != CircuitState.OPEN:
            return False
        
        if self.last_failure_time is None:
            return False
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.timeout
    
    def _transition_state(self, new_state: CircuitState):
        """Transition to new state."""
        old_state = self.state
        self.state = new_state
        self.stats['state_transitions'] += 1
        self.stats['current_state'] = new_state.value
        
        logger.warning(
            f"Circuit breaker state transition: {old_state.value} -> {new_state.value}",
            extra={
                'old_state': old_state.value,
                'new_state': new_state.value,
                'failure_count': self.failure_count,
                'error_rate': self._calculate_error_rate(),
            }
        )
    
    async def __aenter__(self):
        """Context manager entry - check if request should proceed."""
        self.stats['total_requests'] += 1
        
        # Check if should attempt reset
        if self._should_attempt_reset():
            self._transition_state(CircuitState.HALF_OPEN)
            self.failure_count = 0
            self.success_count = 0
        
        # Reject if circuit is open
        if self.state == CircuitState.OPEN:
            self.stats['rejected_requests'] += 1
            raise CircuitBreakerError(
                f"Circuit breaker is OPEN. "
                f"Last failure: {time.time() - self.last_failure_time:.1f}s ago"
            )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - record result and update state."""
        success = exc_type is None
        
        # Update sliding window
        self._update_window(success)
        
        if success:
            self.stats['successful_requests'] += 1
            self._on_success()
        else:
            self.stats['failed_requests'] += 1
            self._on_failure()
        
        return False  # Don't suppress exceptions
    
    def _on_success(self):
        """Handle successful request."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            
            if self.success_count >= self.config.success_threshold:
                # Recovered - close circuit
                self._transition_state(CircuitState.CLOSED)
                self.failure_count = 0
                self.success_count = 0
        
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            # Failed during recovery - reopen circuit
            self._transition_state(CircuitState.OPEN)
            self.success_count = 0
        
        elif self.state == CircuitState.CLOSED:
            # Check if should open circuit
            error_rate = self._calculate_error_rate()
            
            if (self.failure_count >= self.config.failure_threshold or
                error_rate >= self.config.error_rate_threshold):
                self._transition_state(CircuitState.OPEN)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            **self.stats,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'error_rate': round(self._calculate_error_rate(), 3),
            'last_failure_time': self.last_failure_time,
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'success_threshold': self.config.success_threshold,
                'timeout': self.config.timeout,
            }
        }
    
    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        self._transition_state(CircuitState.CLOSED)
        self.failure_count = 0
        self.success_count = 0
        self.recent_results.clear()
        logger.info("Circuit breaker manually reset to CLOSED")


class CircuitBreakerError(Exception):
    """Raised when circuit breaker rejects a request."""
    pass


# ===== Fallback Strategies =====

class FallbackStrategy(Enum):
    """Fallback strategy types."""
    RETURN_DEFAULT = "return_default"
    USE_CACHE = "use_cache"
    SKIP_OPERATION = "skip_operation"
    RAISE_ERROR = "raise_error"


@dataclass
class FallbackConfig:
    """Fallback configuration."""
    strategy: FallbackStrategy = FallbackStrategy.RETURN_DEFAULT
    default_value: Any = None
    cache_ttl: float = 300.0  # 5 minutes
    log_fallback: bool = True


class FallbackManager:
    """
    Manages fallback strategies for unavailable services.
    
    Usage:
        fallback = FallbackManager()
        
        # Register fallback for ASR
        fallback.register(
            'asr',
            FallbackConfig(
                strategy=FallbackStrategy.RETURN_DEFAULT,
                default_value={'text': '[ASR unavailable]', 'confidence': 0.0}
            )
        )
        
        # Use fallback
        result = await fallback.execute('asr', lambda: asr_service.transcribe())
    """
    
    def __init__(self):
        self.configs: Dict[str, FallbackConfig] = {}
        self.cache: Dict[str, Any] = {}
        self.stats = {
            'total_fallbacks': 0,
            'fallback_by_service': {},
        }
    
    def register(self, service: str, config: FallbackConfig):
        """Register fallback config for a service."""
        self.configs[service] = config
        self.stats['fallback_by_service'][service] = 0
        logger.info(
            f"Registered fallback for {service}: {config.strategy.value}",
            extra={'service': service, 'strategy': config.strategy.value}
        )
    
    async def execute(
        self,
        service: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute operation with fallback on failure.
        
        Args:
            service: Service name (for fallback lookup)
            operation: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Operation result or fallback value
        """
        try:
            result = await operation(*args, **kwargs)
            
            # Update cache for future fallback
            if service in self.configs:
                self.cache[service] = {
                    'value': result,
                    'timestamp': time.time()
                }
            
            return result
        
        except Exception as e:
            # Service failed - use fallback
            if service not in self.configs:
                # No fallback configured - re-raise
                logger.error(
                    f"No fallback configured for {service}",
                    extra={'service': service, 'error': str(e)}
                )
                raise
            
            config = self.configs[service]
            self.stats['total_fallbacks'] += 1
            self.stats['fallback_by_service'][service] += 1
            
            if config.log_fallback:
                logger.warning(
                    f"Service {service} failed, using fallback: {config.strategy.value}",
                    extra={
                        'service': service,
                        'error': str(e),
                        'strategy': config.strategy.value
                    }
                )
            
            return await self._apply_fallback(service, config, e)
    
    async def _apply_fallback(
        self,
        service: str,
        config: FallbackConfig,
        error: Exception
    ) -> Any:
        """Apply configured fallback strategy."""
        if config.strategy == FallbackStrategy.RETURN_DEFAULT:
            return config.default_value
        
        elif config.strategy == FallbackStrategy.USE_CACHE:
            cached = self.cache.get(service)
            
            if cached:
                # Check if cache is still valid
                age = time.time() - cached['timestamp']
                if age <= config.cache_ttl:
                    logger.info(
                        f"Using cached value for {service} (age: {age:.1f}s)",
                        extra={'service': service, 'cache_age': age}
                    )
                    return cached['value']
            
            # Cache miss or expired - return default
            logger.warning(
                f"Cache miss for {service}, using default",
                extra={'service': service}
            )
            return config.default_value
        
        elif config.strategy == FallbackStrategy.SKIP_OPERATION:
            logger.info(
                f"Skipping operation for {service}",
                extra={'service': service}
            )
            return None
        
        else:  # RAISE_ERROR
            raise error
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fallback statistics."""
        return {
            'total_fallbacks': self.stats['total_fallbacks'],
            'fallback_by_service': dict(self.stats['fallback_by_service']),
            'registered_services': list(self.configs.keys()),
        }


# ===== Recovery Manager =====

@dataclass
class RecoveryAction:
    """Recovery action definition."""
    name: str
    action: Callable
    rollback: Optional[Callable] = None
    timeout: float = 10.0


class RecoveryManager:
    """
    Coordinated recovery workflows with rollback support.
    
    Usage:
        recovery = RecoveryManager()
        
        # Register recovery actions
        recovery.register_action(RecoveryAction(
            name="restart_asr",
            action=lambda: asr_service.restart(),
            rollback=lambda: asr_service.stop(),
            timeout=10.0
        ))
        
        # Execute recovery
        success = await recovery.execute_recovery(['restart_asr'])
    """
    
    def __init__(self):
        self.actions: Dict[str, RecoveryAction] = {}
        self.recovery_history: List[Dict[str, Any]] = []
    
    def register_action(self, action: RecoveryAction):
        """Register a recovery action."""
        self.actions[action.name] = action
        logger.info(
            f"Registered recovery action: {action.name}",
            extra={'action': action.name}
        )
    
    async def execute_recovery(
        self,
        action_names: List[str],
        rollback_on_failure: bool = True
    ) -> bool:
        """
        Execute recovery workflow.
        
        Args:
            action_names: List of action names to execute in order
            rollback_on_failure: If True, rollback on any failure
        
        Returns:
            True if all actions succeeded, False otherwise
        """
        executed_actions: List[str] = []
        start_time = time.time()
        
        try:
            for action_name in action_names:
                if action_name not in self.actions:
                    logger.error(
                        f"Recovery action not found: {action_name}",
                        extra={'action': action_name}
                    )
                    raise ValueError(f"Unknown recovery action: {action_name}")
                
                action = self.actions[action_name]
                
                logger.info(
                    f"Executing recovery action: {action_name}",
                    extra={'action': action_name}
                )
                
                try:
                    # Execute with timeout
                    await asyncio.wait_for(
                        action.action(),
                        timeout=action.timeout
                    )
                    executed_actions.append(action_name)
                    
                except asyncio.TimeoutError:
                    logger.error(
                        f"Recovery action timed out: {action_name}",
                        extra={'action': action_name, 'timeout': action.timeout}
                    )
                    raise
            
            # All actions succeeded
            duration = time.time() - start_time
            self.recovery_history.append({
                'timestamp': time.time(),
                'actions': action_names,
                'success': True,
                'duration': duration,
            })
            
            logger.info(
                f"Recovery completed successfully in {duration:.2f}s",
                extra={'actions': action_names, 'duration': duration}
            )
            
            return True
        
        except Exception as e:
            # Recovery failed
            duration = time.time() - start_time
            self.recovery_history.append({
                'timestamp': time.time(),
                'actions': action_names,
                'success': False,
                'duration': duration,
                'error': str(e),
            })
            
            logger.error(
                f"Recovery failed: {e}",
                extra={'actions': action_names, 'error': str(e)}
            )
            
            # Rollback if requested
            if rollback_on_failure and executed_actions:
                await self._rollback(executed_actions)
            
            return False
    
    async def _rollback(self, action_names: List[str]):
        """Rollback executed actions in reverse order."""
        logger.warning(
            f"Rolling back {len(action_names)} actions",
            extra={'actions': action_names}
        )
        
        for action_name in reversed(action_names):
            action = self.actions[action_name]
            
            if action.rollback:
                try:
                    logger.info(
                        f"Rolling back action: {action_name}",
                        extra={'action': action_name}
                    )
                    await asyncio.wait_for(
                        action.rollback(),
                        timeout=action.timeout
                    )
                except Exception as e:
                    logger.error(
                        f"Rollback failed for {action_name}: {e}",
                        extra={'action': action_name, 'error': str(e)}
                    )
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent recovery history."""
        return self.recovery_history[-limit:]
