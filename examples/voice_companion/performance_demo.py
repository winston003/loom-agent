"""Performance optimization demo.

Demonstrates Phase 6 (T089-T092) features:
- Cache hit rates and performance
- Concurrency limiting
- Resource cleanup
- Performance statistics

Usage:
    python examples/voice_companion/performance_demo.py
"""

import asyncio
from loom.adapters.audio import AudioAdapter, AudioAdapterConfig


async def demo_performance_optimization():
    """Demonstrate performance optimization features."""
    
    print("=" * 60)
    print("Performance Optimization Demo (Phase 6 - T089-T092)")
    print("=" * 60)
    
    # Step 1: Initialize adapter with performance components
    print("\n[Step 1] Initializing audio adapter with performance optimization...")
    config = AudioAdapterConfig(
        host="127.0.0.1",
        port=8765,
        vad_threshold=0.5,
        sample_rate=16000,
    )
    adapter = AudioAdapter(config)
    await adapter.start()
    print("✓ Adapter started with performance components")
    
    # Step 2: Create multiple sessions to test concurrency
    print("\n[Step 2] Testing concurrency limiter...")
    sessions = []
    
    for i in range(3):
        session_id = await adapter.session_manager.create_session(f"device-{i}")
        sessions.append(session_id)
        print(f"✓ Created session {i+1}: {session_id}")
    
    # Check concurrency stats
    stats = adapter.get_performance_stats()
    print("\nConcurrency stats:")
    print(f"  Global limit: {stats['concurrency']['global_limit']}")
    print(f"  Per-key limit: {stats['concurrency']['per_key_limit']}")
    print(f"  Current global: {stats['concurrency']['current_global']}")
    print(f"  Total acquired: {stats['concurrency']['total_acquired']}")
    
    # Step 3: Test cache performance
    print("\n[Step 3] Testing cache performance...")
    
    # Simulate permission checks (cache miss then hit)
    for i in range(5):
        # First check: cache miss
        result1 = await adapter.permission_cache.get(f"permission_key_{i}")
        print(f"  Check {i+1} (miss): {result1}")
        
        # Set in cache
        await adapter.permission_cache.set(f"permission_key_{i}", "allow")
        
        # Second check: cache hit
        result2 = await adapter.permission_cache.get(f"permission_key_{i}")
        print(f"  Check {i+1} (hit): {result2}")
    
    # Check cache stats
    cache_stats = adapter.permission_cache.get_stats()
    print("\nPermission cache stats:")
    print(f"  Size: {cache_stats['size']}/{cache_stats['max_size']}")
    print(f"  Hits: {cache_stats['hits']}")
    print(f"  Misses: {cache_stats['misses']}")
    print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
    
    # Step 4: Test context cache
    print("\n[Step 4] Testing context cache...")
    
    # Add conversation turns to sessions
    for session_id in sessions[:2]:
        for i in range(3):
            adapter.session_manager.add_conversation_turn(
                session_id=session_id,
                user_text=f"Test message {i+1}",
                agent_response=f"Response {i+1}",
                speaker_id="speaker_001",
            )
    
    # Get context (cache miss)
    context1 = adapter.get_conversation_context(sessions[0], compress=False)
    print(f"✓ Context 1 retrieved ({len(context1)} chars)")
    
    # Cache it
    await adapter.context_cache.set(f"context_{sessions[0]}", context1)
    
    # Get from cache (cache hit)
    cached_context = await adapter.context_cache.get(f"context_{sessions[0]}")
    print(f"✓ Context retrieved from cache ({len(cached_context)} chars)")
    
    # Step 5: Test cleanup manager
    print("\n[Step 5] Testing cleanup manager...")
    
    cleanup_stats = adapter.cleanup_manager.get_stats()
    print("Cleanup manager stats:")
    print(f"  Running: {cleanup_stats['running']}")
    print(f"  Interval: {cleanup_stats['interval']}s")
    print(f"  Handlers: {cleanup_stats['handlers']}")
    print(f"  Active tasks: {cleanup_stats['active_tasks']}")
    
    # Wait for one cleanup cycle
    print("\nWaiting for cleanup cycle (3 seconds)...")
    await asyncio.sleep(3)
    
    # Check if any expired entries were cleaned
    expired_count = await adapter.permission_cache.cleanup_expired()
    print(f"✓ Cleaned up {expired_count} expired cache entries")
    
    # Step 6: Comprehensive performance stats
    print("\n[Step 6] Comprehensive performance statistics...")
    
    final_stats = adapter.get_performance_stats()
    
    print("\n📊 Full Performance Report:")
    print("\n  Cache Statistics:")
    for cache_name, cache_data in final_stats["caches"].items():
        print(f"    {cache_name.upper()}:")
        print(f"      Size: {cache_data['size']}/{cache_data['max_size']}")
        print(f"      Hit rate: {cache_data['hit_rate']:.2%}")
        print(f"      Hits: {cache_data['hits']}, Misses: {cache_data['misses']}")
    
    print("\n  Concurrency:")
    for key, value in final_stats["concurrency"].items():
        print(f"    {key}: {value}")
    
    print("\n  Cleanup:")
    for key, value in final_stats["cleanup"].items():
        print(f"    {key}: {value}")
    
    # Step 7: Cleanup
    print("\n[Step 7] Cleaning up...")
    await adapter.stop()
    print("✓ Adapter stopped and resources cleaned")
    
    print("\n" + "=" * 60)
    print("Performance optimization demo completed! ✅")
    print("=" * 60)
    print("\nKey features demonstrated:")
    print("✓ TTL caching with hit/miss tracking (T090)")
    print("✓ Concurrency limiting (global + per-device) (T091)")
    print("✓ Automatic resource cleanup (T092)")
    print("✓ Performance statistics monitoring (T089-T092)")


if __name__ == "__main__":
    asyncio.run(demo_performance_optimization())
