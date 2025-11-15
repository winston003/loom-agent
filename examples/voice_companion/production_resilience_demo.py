"""
Production resilience integration example.

Demonstrates how to use resilience components in AudioAdapter
for production-grade fault tolerance.
"""

import asyncio
from loom.adapters.audio.adapter import AudioAdapter, AudioAdapterConfig


async def demo_resilience_integration():
    """Demonstrate resilience features in AudioAdapter."""
    
    print("\n" + "="*70)
    print("PRODUCTION RESILIENCE INTEGRATION")
    print("="*70)
    
    # Create adapter (resilience components auto-initialized)
    config = AudioAdapterConfig(
        host="127.0.0.1",
        port=8766,
        sample_rate=16000,
        vad_threshold=0.5,
    )
    
    adapter = AudioAdapter(config)
    
    print("\n1. Starting AudioAdapter with resilience components...")
    await adapter.start()
    
    # Verify components initialized
    assert adapter.retry_policy is not None, "Retry policy not initialized"
    assert adapter.asr_circuit_breaker is not None, "ASR circuit breaker not initialized"
    assert adapter.voiceprint_circuit_breaker is not None, "Voiceprint circuit breaker not initialized"
    assert adapter.fallback_manager is not None, "Fallback manager not initialized"
    assert adapter.recovery_manager is not None, "Recovery manager not initialized"
    
    print("   ✓ Retry policy initialized (max_attempts=3, exponential backoff)")
    print("   ✓ ASR circuit breaker initialized (threshold=5 failures)")
    print("   ✓ Voiceprint circuit breaker initialized (threshold=3 failures)")
    print("   ✓ Fallback manager initialized (ASR + Voiceprint)")
    print("   ✓ Recovery manager initialized")
    
    # Show performance stats
    print("\n2. Performance optimization components:")
    perf_stats = adapter.get_performance_stats()
    
    print("   Caches initialized:")
    for cache_name, cache_stats in perf_stats["caches"].items():
        print(f"     • {cache_name}: size={cache_stats['size']}/{cache_stats['max_size']}, "
              f"hits={cache_stats['hits']}, misses={cache_stats['misses']}")
    
    print("\n   Concurrency limiter:")
    conc = perf_stats["concurrency"]
    print(f"     • Global: {conc['acquired_count']}/{conc['global_limit']}")
    print(f"     • Per-key: max {conc['per_key_limit']} per device")
    print(f"     • Rejected: {conc['rejected_count']}")
    
    # Show resilience stats
    print("\n3. Resilience component statistics:")
    res_stats = adapter.get_resilience_stats()
    
    print("   Retry policy:")
    retry = res_stats["retry"]
    print(f"     • Total attempts: {retry['total_attempts']}")
    print(f"     • Successful retries: {retry['successful_retries']}")
    print(f"     • Failed retries: {retry['failed_retries']}")
    
    print("\n   Circuit breakers:")
    for service, cb_stats in res_stats["circuit_breakers"].items():
        print(f"     • {service.upper()}: {cb_stats['current_state']}")
        print(f"       - Total requests: {cb_stats['total_requests']}")
        print(f"       - Rejected: {cb_stats['rejected_requests']}")
        print(f"       - Error rate: {cb_stats['error_rate']:.1%}")
    
    print("\n   Fallback manager:")
    fallback = res_stats["fallback"]
    print(f"     • Total fallbacks: {fallback['total_fallbacks']}")
    print(f"     • Registered services: {fallback['registered_services']}")
    
    # Simulate some activity
    print("\n4. Simulating API calls with resilience...")
    
    # Example: Using retry policy
    print("\n   a) Testing retry policy:")
    call_count = [0]
    
    async def flaky_api_call():
        call_count[0] += 1
        if call_count[0] < 2:
            print(f"      Attempt {call_count[0]}: Failed (simulated)")
            raise ConnectionError("API timeout")
        print(f"      Attempt {call_count[0]}: Success")
        return {"result": "ok"}
    
    try:
        result = await adapter.retry_policy.execute(flaky_api_call)
        print(f"      → Final result: {result}")
    except Exception as e:
        print(f"      → Failed after all retries: {e}")
    
    # Example: Using circuit breaker
    print("\n   b) Testing circuit breaker:")
    
    for i in range(3):
        try:
            async with adapter.asr_circuit_breaker:
                # Simulate ASR call
                print(f"      Call {i+1}: Processing...")
                await asyncio.sleep(0.1)
                print(f"      Call {i+1}: Success")
        except Exception as e:
            print(f"      Call {i+1}: {e}")
    
    # Example: Using fallback
    print("\n   c) Testing fallback strategies:")
    
    async def failing_asr():
        raise TimeoutError("ASR service down")
    
    result = await adapter.fallback_manager.execute('asr', failing_asr)
    print(f"      ASR fallback result: {result}")
    
    # Updated stats
    print("\n5. Updated resilience statistics:")
    res_stats = adapter.get_resilience_stats()
    
    retry = res_stats["retry"]
    print(f"   Retry attempts: {retry['total_attempts']} "
          f"(successful: {retry['successful_retries']}, "
          f"failed: {retry['failed_retries']})")
    
    fallback = res_stats["fallback"]
    print(f"   Fallback usage: {fallback['fallback_by_service']}")
    
    asr_cb = res_stats["circuit_breakers"]["asr"]
    print(f"   ASR circuit breaker: {asr_cb['current_state']} "
          f"(requests: {asr_cb['total_requests']}, "
          f"rejected: {asr_cb['rejected_requests']})")
    
    # Cleanup
    print("\n6. Shutting down...")
    await adapter.stop()
    
    print("\n" + "="*70)
    print("RESILIENCE INTEGRATION DEMO COMPLETE")
    print("="*70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Automatic retry with exponential backoff")
    print("  ✓ Circuit breaker pattern for ASR/Voiceprint")
    print("  ✓ Fallback strategies (default + cache)")
    print("  ✓ Comprehensive statistics tracking")
    print("  ✓ Production-ready error handling")


async def demo_circuit_breaker_states():
    """Demonstrate circuit breaker state transitions."""
    
    print("\n" + "="*70)
    print("CIRCUIT BREAKER STATE TRANSITIONS")
    print("="*70)
    
    adapter = AudioAdapter()
    await adapter.start()
    
    print("\n1. Initial state:")
    stats = adapter.get_resilience_stats()
    asr_cb = stats["circuit_breakers"]["asr"]
    print(f"   ASR circuit breaker: {asr_cb['current_state'].upper()}")
    
    print("\n2. Simulating failures to trigger OPEN state...")
    
    # Simulate 6 failures (threshold is 5)
    for i in range(6):
        try:
            async with adapter.asr_circuit_breaker:
                # Simulate ASR failure
                raise TimeoutError(f"ASR timeout #{i+1}")
        except Exception as e:
            print(f"   Attempt {i+1}: Failed - {str(e)[:40]}")
    
    # Check state
    stats = adapter.get_resilience_stats()
    asr_cb = stats["circuit_breakers"]["asr"]
    print(f"\n   → State after failures: {asr_cb['current_state'].upper()}")
    print(f"   → Error rate: {asr_cb['error_rate']:.1%}")
    
    print("\n3. Testing rejection during OPEN state...")
    from loom.adapters.audio.resilience import CircuitBreakerError
    
    try:
        async with adapter.asr_circuit_breaker:
            print("   This should be rejected...")
    except CircuitBreakerError as e:
        print(f"   ⊗ Request rejected: {str(e)[:60]}")
    
    print("\n4. Waiting for recovery timeout (30 seconds)...")
    print("   (In production, circuit would transition to HALF_OPEN)")
    print("   (Skipping wait for demo...)")
    
    # Manually reset for demo
    adapter.asr_circuit_breaker.reset()
    print("\n   Circuit manually reset to CLOSED")
    
    # Verify
    stats = adapter.get_resilience_stats()
    asr_cb = stats["circuit_breakers"]["asr"]
    print(f"   Final state: {asr_cb['current_state'].upper()}")
    
    await adapter.stop()
    
    print("\n" + "="*70)
    print("State transition flow:")
    print("  CLOSED → (failures exceed threshold) → OPEN")
    print("  OPEN → (timeout expires) → HALF_OPEN")
    print("  HALF_OPEN → (success threshold met) → CLOSED")
    print("  HALF_OPEN → (any failure) → OPEN")
    print("="*70)


async def main():
    """Run all demos."""
    await demo_resilience_integration()
    print("\n" + "="*70 + "\n")
    await demo_circuit_breaker_states()


if __name__ == "__main__":
    asyncio.run(main())
