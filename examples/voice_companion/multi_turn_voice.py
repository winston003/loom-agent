"""Multi-turn voice conversation demo.

Demonstrates Phase 5 (User Story 3) features:
- Multi-turn conversation tracking
- Context preservation across turns
- Automatic compression after 10 turns
- Per-speaker context isolation
- Idle timeout management

Usage:
    python examples/voice_companion/multi_turn_voice.py

Simulates a 5-turn conversation with context retention.
"""

import asyncio
from loom.adapters.audio import AudioAdapter, AudioAdapterConfig
from datetime import datetime


async def demo_multi_turn_conversation():
    """Demonstrate multi-turn conversation with context management."""
    
    print("=" * 60)
    print("Multi-turn Conversation Demo (Phase 5 - User Story 3)")
    print("=" * 60)
    
    # Step 1: Initialize audio adapter
    print("\n[Step 1] Initializing audio adapter...")
    config = AudioAdapterConfig(
        host="127.0.0.1",
        port=8765,
        vad_threshold=0.5,
        sample_rate=16000,
    )
    adapter = AudioAdapter(config)
    await adapter.start()
    print("✓ Audio adapter started")
    
    # Step 2: Create session
    print("\n[Step 2] Creating audio session...")
    session_id = await adapter.session_manager.create_session(device_id="xiaozhi-demo-001")
    print(f"✓ Session created: {session_id}")
    
    # Step 3: Simulate multi-turn conversation
    print("\n[Step 3] Simulating 5-turn conversation...")
    
    conversations = [
        {
            "user": "今天天气怎么样？",
            "agent": "今天北京天气晴朗，温度 22°C，适合外出活动。",
            "speaker_id": "speaker_001",
        },
        {
            "user": "明天呢？",  # 简化问法 - 依赖上下文理解
            "agent": "明天预计多云，温度 20°C，有小雨可能，建议带伞。",
            "speaker_id": "speaker_001",
        },
        {
            "user": "后天会下雨吗？",  # 继续天气主题
            "agent": "后天转晴，温度回升到 24°C，无降雨。",
            "speaker_id": "speaker_001",
        },
        {
            "user": "这周末去哪玩比较好？",  # 基于前面天气信息
            "agent": "根据天气预报，周六天气最好，推荐去颐和园或者故宫游览。",
            "speaker_id": "speaker_001",
        },
        {
            "user": "好的，帮我设置周六上午的闹钟",  # 延续计划
            "agent": "已设置周六上午 8:00 闹钟，祝您游玩愉快！",
            "speaker_id": "speaker_001",
        },
    ]
    
    for i, conv in enumerate(conversations, 1):
        print(f"\n--- Turn {i} ---")
        print(f"👤 User: {conv['user']}")
        
        # Add conversation turn (simulating ASR + Agent response)
        adapter.session_manager.add_conversation_turn(
            session_id=session_id,
            user_text=conv["user"],
            agent_response=conv["agent"],
            speaker_id=conv["speaker_id"],
            metadata={"turn_number": i},
        )
        
        print(f"🤖 Agent: {conv['agent']}")
        
        # Check session state
        session = adapter.session_manager.get_session(session_id)
        print(f"📊 Turn count: {session.turn_count}")
    
    # Step 4: Retrieve conversation context
    print("\n[Step 4] Retrieving conversation context...")
    
    session = adapter.session_manager.get_session(session_id)
    
    # Get full context (no compression at 5 turns)
    context = adapter.get_conversation_context(
        session_id=session_id,
        system_prompt="You are Xiaozhi, a helpful voice assistant.",
        compress=False,
    )
    
    print(f"\n📝 Full context ({len(context)} chars):")
    print(context)
    
    # Step 5: Simulate 10 more turns to trigger compression
    print("\n[Step 5] Adding 10 more turns to trigger compression...")
    
    for i in range(6, 16):
        adapter.session_manager.add_conversation_turn(
            session_id=session_id,
            user_text=f"这是第 {i} 轮测试对话",
            agent_response=f"收到第 {i} 轮消息",
            speaker_id="speaker_001",
        )
    
    session = adapter.session_manager.get_session(session_id)
    print(f"✓ Total turns: {session.turn_count}")
    print(f"✓ Compression needed: {adapter.context_manager.should_compress(session.turn_count)}")
    
    # Get compressed context
    compressed_context = adapter.get_conversation_context(
        session_id=session_id,
        system_prompt="You are Xiaozhi, a helpful voice assistant.",
        compress=True,
    )
    
    print(f"\n📦 Compressed context ({len(compressed_context)} chars):")
    print(compressed_context)
    
    # Step 6: Test idle timeout
    print("\n[Step 6] Testing idle timeout...")
    
    # Check current idle status
    is_idle = adapter.session_manager.check_idle_timeout(session_id, timeout_seconds=0)
    print(f"✓ Session idle (0s threshold): {is_idle}")
    
    # Simulate activity update
    session.last_activity_at = datetime.utcnow()
    is_idle = adapter.session_manager.check_idle_timeout(session_id, timeout_seconds=30)
    print(f"✓ Session idle (30s threshold): {is_idle}")
    
    # Step 7: Test per-speaker context isolation
    print("\n[Step 7] Testing per-speaker context isolation...")
    
    # Add turn from different speaker
    adapter.session_manager.add_conversation_turn(
        session_id=session_id,
        user_text="我是访客，请帮我查询天气",
        agent_response="好的，请问您要查询哪个城市？",
        speaker_id="speaker_002",  # Different speaker
    )
    
    # Get context for speaker_001 only
    speaker1_context = adapter.context_manager.assemble_context(
        conversation_history=session.conversation_history,
        speaker_id="speaker_001",
        compress=False,
    )
    
    # Get context for speaker_002 only
    speaker2_context = adapter.context_manager.assemble_context(
        conversation_history=session.conversation_history,
        speaker_id="speaker_002",
        compress=False,
    )
    
    print(f"✓ Speaker 1 context: {len(speaker1_context)} chars (15 turns)")
    print(f"✓ Speaker 2 context: {len(speaker2_context)} chars (1 turn)")
    
    # Step 8: Cleanup
    print("\n[Step 8] Cleaning up...")
    await adapter.session_manager.close_session(session_id)
    await adapter.stop()
    print("✓ Session closed and adapter stopped")
    
    print("\n" + "=" * 60)
    print("Multi-turn conversation demo completed! ✅")
    print("=" * 60)
    print("\nKey features demonstrated:")
    print("✓ Multi-turn conversation tracking (T076-T078)")
    print("✓ Context assembly and compression (T079-T081)")
    print("✓ Per-speaker context isolation (T082)")
    print("✓ Idle timeout management (T085)")
    print("✓ Auto-compression after 10 turns (T086)")


if __name__ == "__main__":
    asyncio.run(demo_multi_turn_conversation())
