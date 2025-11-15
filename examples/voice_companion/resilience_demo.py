"""
Resilience demo for audio adapter.

Demonstrates:
- Retry mechanism with exponential backoff
- Circuit breaker pattern
- Fallback strategies
- Recovery workflows
"""

import asyncio
import random
from loom.adapters.audio.resilience import (
    RetryPolicy,
    RetryConfig,
    RetryStrategy,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    FallbackManager,
    FallbackConfig,
    FallbackStrategy,
    RecoveryManager,
    RecoveryAction,
)


# ===== Mock Services =====

class UnreliableASRService:
    """Mock ASR service with configurable failure rate."""
    
    def __init__(self, failure_rate: float = 0.3):
        self.failure_rate = failure_rate
        self.call_count = 0
    
    async def transcribe(self, audio_data: bytes) -> dict:
        """Transcribe audio (may fail randomly)."""
        self.call_count += 1
        await asyncio.sleep(0.1)  # Simulate processing
        
        if random.random() < self.failure_rate:
            raise ConnectionError(f"ASR service timeout (call #{self.call_count})")
        
        return {
            'text': f'Transcription result #{self.call_count}',
            'confidence': 0.95
        }


class FlakeyVoiceprintService:
    """Mock voiceprint service that fails periodically."""
    
    def __init__(self):
        self.call_count = 0
        self.is_available = True
    
    async def verify(self, audio_data: bytes, user_id: str) -> dict:
        """Verify voiceprint (may fail)."""
        self.call_count += 1
        await asyncio.sleep(0.05)
        
        # Fail every 5th call
        if self.call_count % 5 == 0:
            self.is_available = False
            raise TimeoutError(f"Voiceprint service unavailable (call #{self.call_count})")
        
        self.is_available = True
        return {
            'user_id': user_id,
            'similarity': 0.92,
            'verified': True
        }


# ===== Demo Functions =====

async def demo_retry_policy():
    """Demo: Retry mechanism with exponential backoff."""
    print("\n" + "="*60)
    print("DEMO 1: Retry Policy")
    print("="*60)
    
    # Create unreliable service (30% failure rate)
    asr = UnreliableASRService(failure_rate=0.3)
    
    # Create retry policy
    retry = RetryPolicy(RetryConfig(
        max_attempts=5,
        initial_delay=0.1,
        max_delay=2.0,
        backoff_factor=2.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF
    ))
    
    print("\n1. Testing retry with 30% failure rate...")
    print("   Max attempts: 5, Exponential backoff (0.1s → 2.0s)")
    
    try:
        result = await retry.execute(
            asr.transcribe,
            b"test audio data"
        )
        print(f"\n✓ Success: {result['text']}")
        print(f"  Total calls: {asr.call_count}")
    except Exception as e:
        print(f"\n✗ Failed after all retries: {e}")
    
    # Show stats
    stats = retry.get_stats()
    print(f"\nRetry Statistics:")
    print(f"  Total attempts: {stats['total_attempts']}")
    print(f"  Successful retries: {stats['successful_retries']}")
    print(f"  Failed retries: {stats['failed_retries']}")
    print(f"  Total delay: {stats['total_delay']}s")


async def demo_circuit_breaker():
    """Demo: Circuit breaker pattern."""
    print("\n" + "="*60)
    print("DEMO 2: Circuit Breaker")
    print("="*60)
    
    # Create service with high failure rate
    voiceprint = FlakeyVoiceprintService()
    
    # Create circuit breaker
    breaker = CircuitBreaker(CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout=5.0,
        window_size=10,
        error_rate_threshold=0.5
    ))
    
    print("\n1. Testing circuit breaker...")
    print("   Failure threshold: 3, Recovery timeout: 5s")
    
    for i in range(15):
        try:
            async with breaker:
                result = await voiceprint.verify(b"audio", "user123")
                print(f"\n  Call {i+1}: ✓ Success - {result['verified']}")
        
        except CircuitBreakerError as e:
            print(f"\n  Call {i+1}: ⊗ Rejected - {str(e)[:60]}")
        
        except Exception as e:
            print(f"\n  Call {i+1}: ✗ Failed - {e}")
        
        # Show state after key events
        if i == 4:
            stats = breaker.get_stats()
            print(f"\n  → State after 5 calls: {stats['current_state'].upper()}")
            print(f"     Failures: {stats['failed_requests']}, Error rate: {stats['error_rate']}")
        
        await asyncio.sleep(0.2)
    
    # Final stats
    stats = breaker.get_stats()
    print(f"\n2. Final Statistics:")
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Successful: {stats['successful_requests']}")
    print(f"   Failed: {stats['failed_requests']}")
    print(f"   Rejected: {stats['rejected_requests']}")
    print(f"   State transitions: {stats['state_transitions']}")
    print(f"   Current state: {stats['current_state'].upper()}")


async def demo_fallback_strategies():
    """Demo: Fallback strategies for degraded services."""
    print("\n" + "="*60)
    print("DEMO 3: Fallback Strategies")
    print("="*60)
    
    # Create fallback manager
    fallback = FallbackManager()
    
    # Register fallback for ASR
    fallback.register(
        'asr',
        FallbackConfig(
            strategy=FallbackStrategy.RETURN_DEFAULT,
            default_value={'text': '[ASR service unavailable]', 'confidence': 0.0}
        )
    )
    
    # Register fallback for voiceprint (use cache)
    fallback.register(
        'voiceprint',
        FallbackConfig(
            strategy=FallbackStrategy.USE_CACHE,
            default_value={'verified': False, 'similarity': 0.0},
            cache_ttl=10.0
        )
    )
    
    print("\n1. Testing ASR fallback (RETURN_DEFAULT)...")
    
    # Simulate successful call (updates cache)
    async def asr_success():
        return {'text': 'Hello world', 'confidence': 0.95}
    
    result = await fallback.execute('asr', asr_success)
    print(f"   ✓ Success: {result['text']}")
    
    # Simulate failed call (uses fallback)
    async def asr_failure():
        raise ConnectionError("ASR timeout")
    
    result = await fallback.execute('asr', asr_failure)
    print(f"   → Fallback: {result['text']}")
    
    print("\n2. Testing voiceprint fallback (USE_CACHE)...")
    
    # Successful call (populates cache)
    async def voiceprint_success():
        return {'verified': True, 'similarity': 0.92, 'user_id': 'user123'}
    
    result = await fallback.execute('voiceprint', voiceprint_success)
    print(f"   ✓ Success: verified={result['verified']}, similarity={result['similarity']}")
    
    # Failed call (uses cached value)
    async def voiceprint_failure():
        raise TimeoutError("Voiceprint service down")
    
    result = await fallback.execute('voiceprint', voiceprint_failure)
    print(f"   → Cached: verified={result['verified']}, similarity={result['similarity']}")
    
    # Show stats
    stats = fallback.get_stats()
    print(f"\n3. Fallback Statistics:")
    print(f"   Total fallbacks: {stats['total_fallbacks']}")
    print(f"   By service: {stats['fallback_by_service']}")


async def demo_recovery_workflow():
    """Demo: Recovery workflows with rollback."""
    print("\n" + "="*60)
    print("DEMO 4: Recovery Workflows")
    print("="*60)
    
    # Create recovery manager
    recovery = RecoveryManager()
    
    # Mock service state
    services = {
        'asr': {'running': False},
        'voiceprint': {'running': False},
        'vad': {'running': False}
    }
    
    # Register recovery actions
    recovery.register_action(RecoveryAction(
        name='restart_asr',
        action=lambda: restart_service('asr', services),
        rollback=lambda: stop_service('asr', services),
        timeout=2.0
    ))
    
    recovery.register_action(RecoveryAction(
        name='restart_voiceprint',
        action=lambda: restart_service('voiceprint', services),
        rollback=lambda: stop_service('voiceprint', services),
        timeout=2.0
    ))
    
    recovery.register_action(RecoveryAction(
        name='restart_vad',
        action=lambda: restart_service('vad', services),
        rollback=lambda: stop_service('vad', services),
        timeout=2.0
    ))
    
    print("\n1. Testing successful recovery workflow...")
    print("   Actions: restart_asr → restart_voiceprint → restart_vad")
    
    success = await recovery.execute_recovery([
        'restart_asr',
        'restart_voiceprint',
        'restart_vad'
    ])
    
    print(f"\n   Result: {'✓ Success' if success else '✗ Failed'}")
    print(f"   Service states: {services}")
    
    print("\n2. Testing recovery with rollback...")
    print("   Simulating failure in voiceprint restart")
    
    # Inject failure
    async def faulty_restart():
        raise RuntimeError("Voiceprint restart failed")
    
    recovery.register_action(RecoveryAction(
        name='restart_voiceprint_faulty',
        action=faulty_restart,
        rollback=lambda: stop_service('voiceprint', services),
        timeout=2.0
    ))
    
    success = await recovery.execute_recovery(
        ['restart_asr', 'restart_voiceprint_faulty'],
        rollback_on_failure=True
    )
    
    print(f"\n   Result: {'✓ Success' if success else '✗ Failed (rolled back)'}")
    print(f"   Service states after rollback: {services}")
    
    # Show history
    history = recovery.get_history(limit=5)
    print(f"\n3. Recovery History ({len(history)} attempts):")
    for i, record in enumerate(history, 1):
        status = "✓" if record['success'] else "✗"
        print(f"   {i}. {status} {record['actions']} - {record['duration']:.2f}s")


# Helper functions for recovery demo
async def restart_service(name: str, services: dict):
    """Mock service restart."""
    await asyncio.sleep(0.1)  # Simulate restart time
    services[name]['running'] = True
    print(f"      → Restarted {name}")


async def stop_service(name: str, services: dict):
    """Mock service stop."""
    await asyncio.sleep(0.05)
    services[name]['running'] = False
    print(f"      ← Stopped {name}")


# ===== Main =====

async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("RESILIENCE COMPONENTS DEMO")
    print("="*60)
    print("\nDemonstrating production-grade fault tolerance:")
    print("  • Retry mechanisms with backoff")
    print("  • Circuit breaker pattern")
    print("  • Fallback strategies")
    print("  • Recovery workflows")
    
    await demo_retry_policy()
    await asyncio.sleep(1)
    
    await demo_circuit_breaker()
    await asyncio.sleep(1)
    
    await demo_fallback_strategies()
    await asyncio.sleep(1)
    
    await demo_recovery_workflow()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nKey Takeaways:")
    print("  ✓ Retry policy handles transient failures")
    print("  ✓ Circuit breaker prevents cascade failures")
    print("  ✓ Fallback strategies maintain service availability")
    print("  ✓ Recovery manager coordinates complex workflows")


if __name__ == "__main__":
    asyncio.run(main())
