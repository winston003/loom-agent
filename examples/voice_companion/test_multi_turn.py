"""Automated multi-turn conversation test script.

Tests Phase 5 (User Story 3) implementation:
- Context preservation across turns
- Compression accuracy and performance
- Per-speaker isolation
- Idle timeout behavior

Usage:
    python examples/voice_companion/test_multi_turn.py

Returns exit code 0 on success, 1 on failure.
"""

import asyncio
import sys
import time
from datetime import datetime, timedelta

from loom.adapters.audio import AudioAdapter, AudioAdapterConfig


class MultiTurnTestSuite:
    """Test suite for multi-turn conversation features."""
    
    def __init__(self):
        self.adapter: AudioAdapter = None
        self.session_id: str = None
        self.passed_tests = 0
        self.failed_tests = 0
    
    async def setup(self):
        """Initialize test environment."""
        print("Setting up test environment...")
        config = AudioAdapterConfig(
            host="127.0.0.1",
            port=8765,
            vad_threshold=0.5,
            sample_rate=16000,
        )
        self.adapter = AudioAdapter(config)
        await self.adapter.start()
        self.session_id = await self.adapter.session_manager.create_session("test-device")
        print(f"✓ Test environment ready (session: {self.session_id})")
    
    async def teardown(self):
        """Cleanup test environment."""
        print("\nCleaning up...")
        if self.session_id:
            await self.adapter.session_manager.close_session(self.session_id)
        if self.adapter:
            await self.adapter.stop()
        print("✓ Test environment cleaned up")
    
    def assert_true(self, condition: bool, test_name: str):
        """Assert condition is True."""
        if condition:
            print(f"✓ {test_name}")
            self.passed_tests += 1
        else:
            print(f"✗ {test_name}")
            self.failed_tests += 1
    
    def assert_equal(self, actual, expected, test_name: str):
        """Assert actual equals expected."""
        if actual == expected:
            print(f"✓ {test_name}")
            self.passed_tests += 1
        else:
            print(f"✗ {test_name} (expected: {expected}, got: {actual})")
            self.failed_tests += 1
    
    def assert_less_than(self, actual, threshold, test_name: str):
        """Assert actual is less than threshold."""
        if actual < threshold:
            print(f"✓ {test_name} ({actual} < {threshold})")
            self.passed_tests += 1
        else:
            print(f"✗ {test_name} ({actual} >= {threshold})")
            self.failed_tests += 1
    
    async def test_turn_counting(self):
        """Test T077: Turn count increment."""
        print("\n[Test 1] Turn counting...")
        
        session = self.adapter.session_manager.get_session(self.session_id)
        initial_count = session.turn_count
        
        # Add 3 turns
        for i in range(3):
            self.adapter.session_manager.add_conversation_turn(
                session_id=self.session_id,
                user_text=f"Test utterance {i+1}",
                agent_response=f"Response {i+1}",
                speaker_id="test_speaker",
            )
        
        session = self.adapter.session_manager.get_session(self.session_id)
        self.assert_equal(
            session.turn_count,
            initial_count + 3,
            "Turn count incremented correctly"
        )
    
    async def test_conversation_history(self):
        """Test T078: Conversation history storage."""
        print("\n[Test 2] Conversation history...")
        
        session = self.adapter.session_manager.get_session(self.session_id)
        history_len = len(session.conversation_history)
        
        self.assert_equal(
            history_len,
            session.turn_count,
            "History length matches turn count"
        )
        
        # Verify latest turn content
        if history_len > 0:
            latest = session.conversation_history[-1]
            self.assert_true(
                latest.user_text.startswith("Test utterance"),
                "Latest turn has correct user text"
            )
            self.assert_true(
                latest.agent_response.startswith("Response"),
                "Latest turn has correct agent response"
            )
    
    async def test_context_assembly(self):
        """Test T079: Context assembly."""
        print("\n[Test 3] Context assembly...")
        
        context = self.adapter.get_conversation_context(
            session_id=self.session_id,
            system_prompt="System prompt test",
            compress=False,
        )
        
        self.assert_true(
            "System prompt test" in context,
            "System prompt included in context"
        )
        self.assert_true(
            "Test utterance" in context,
            "User utterances included in context"
        )
        self.assert_true(
            "Response" in context,
            "Agent responses included in context"
        )
    
    async def test_compression_performance(self):
        """Test T079-T081: Compression performance (<50ms)."""
        print("\n[Test 4] Compression performance...")
        
        # Add 12 more turns to trigger compression (total: 15)
        for i in range(12):
            self.adapter.session_manager.add_conversation_turn(
                session_id=self.session_id,
                user_text=f"Compression test utterance {i+1}",
                agent_response=f"Compression test response {i+1}",
                speaker_id="test_speaker",
            )
        
        session = self.adapter.session_manager.get_session(self.session_id)
        
        # Measure compression time
        start_time = time.time()
        context = self.adapter.get_conversation_context(
            session_id=self.session_id,
            compress=True,
        )
        elapsed_ms = (time.time() - start_time) * 1000
        
        self.assert_true(
            session.turn_count >= 10,
            f"Compression threshold reached ({session.turn_count} turns)"
        )
        self.assert_less_than(
            elapsed_ms,
            50,
            "Compression time < 50ms"
        )
        self.assert_true(
            "[Earlier conversation summary]" in context,
            "Compressed summary included"
        )
        self.assert_true(
            "[Recent conversation]" in context,
            "Recent turns included"
        )
    
    async def test_per_speaker_isolation(self):
        """Test T082: Per-speaker context isolation."""
        print("\n[Test 5] Per-speaker context isolation...")
        
        # Add turns from different speakers
        self.adapter.session_manager.add_conversation_turn(
            session_id=self.session_id,
            user_text="Speaker A message",
            agent_response="Response to A",
            speaker_id="speaker_a",
        )
        self.adapter.session_manager.add_conversation_turn(
            session_id=self.session_id,
            user_text="Speaker B message",
            agent_response="Response to B",
            speaker_id="speaker_b",
        )
        
        session = self.adapter.session_manager.get_session(self.session_id)
        
        # Get context for speaker A only
        context_a = self.adapter.context_manager.assemble_context(
            conversation_history=session.conversation_history,
            speaker_id="speaker_a",
            compress=False,
        )
        
        # Get context for speaker B only
        context_b = self.adapter.context_manager.assemble_context(
            conversation_history=session.conversation_history,
            speaker_id="speaker_b",
            compress=False,
        )
        
        self.assert_true(
            "Speaker A message" in context_a and "Speaker B message" not in context_a,
            "Speaker A context isolated"
        )
        self.assert_true(
            "Speaker B message" in context_b and "Speaker A message" not in context_b,
            "Speaker B context isolated"
        )
    
    async def test_idle_timeout(self):
        """Test T085: Idle timeout detection."""
        print("\n[Test 6] Idle timeout detection...")
        
        session = self.adapter.session_manager.get_session(self.session_id)
        
        # Set last activity to 31 seconds ago
        session.last_activity_at = datetime.utcnow() - timedelta(seconds=31)
        
        is_idle = self.adapter.session_manager.check_idle_timeout(
            self.session_id,
            timeout_seconds=30
        )
        
        self.assert_true(
            is_idle,
            "Idle timeout detected (31s > 30s)"
        )
        
        # Reset activity
        session.last_activity_at = datetime.utcnow()
        is_not_idle = not self.adapter.session_manager.check_idle_timeout(
            self.session_id,
            timeout_seconds=30
        )
        
        self.assert_true(
            is_not_idle,
            "Active session not marked idle"
        )
    
    async def test_auto_compression_trigger(self):
        """Test T086: Auto-compression trigger."""
        print("\n[Test 7] Auto-compression trigger...")
        
        session = self.adapter.session_manager.get_session(self.session_id)
        
        should_compress = self.adapter.session_manager.auto_compress_if_needed(
            self.session_id
        )
        
        self.assert_true(
            should_compress == (session.turn_count >= 10),
            f"Auto-compression decision correct (turns: {session.turn_count})"
        )
    
    async def run_all_tests(self):
        """Run all test cases."""
        await self.setup()
        
        try:
            await self.test_turn_counting()
            await self.test_conversation_history()
            await self.test_context_assembly()
            await self.test_compression_performance()
            await self.test_per_speaker_isolation()
            await self.test_idle_timeout()
            await self.test_auto_compression_trigger()
        finally:
            await self.teardown()
        
        # Print summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"✓ Passed: {self.passed_tests}")
        print(f"✗ Failed: {self.failed_tests}")
        print(f"Total: {self.passed_tests + self.failed_tests}")
        
        if self.failed_tests == 0:
            print("\n✅ All tests passed!")
            return 0
        else:
            print(f"\n❌ {self.failed_tests} test(s) failed")
            return 1


async def main():
    """Run test suite."""
    suite = MultiTurnTestSuite()
    exit_code = await suite.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
