"""Demo: Two-factor confirmation for critical operations.

This example demonstrates the complete permission and confirmation flow:
1. Voiceprint verification (get speaker_id)
2. Permission check (check_tool_permission)
3. Two-factor confirmation for critical operations
4. Confirmation recognition (keywords or command repetition)

Reference: specs/002-xiaozhi-voice-adapter/tasks.md T067-T069
"""

import asyncio
from loom.adapters.audio import AudioAdapter, AudioAdapterConfig
from loom.adapters.audio.permissions import AudioPermissionManager
from loom.adapters.audio.confirmation import ConfirmationManager


async def demo_two_factor_confirmation():
    """Demonstrate two-factor confirmation workflow."""
    
    print("=" * 60)
    print("Two-Factor Confirmation Demo")
    print("=" * 60)
    
    # Step 1: Setup permission manager with owner and guest
    print("\n[Step 1] Setting up permission manager...")
    
    pm = AudioPermissionManager.from_config()
    
    # Add test users
    pm.add_owner("speaker_001")
    pm.add_guest("speaker_002")
    
    print(f"✓ Owner: speaker_001")
    print(f"✓ Guest: speaker_002")
    
    # Step 2: Setup confirmation manager
    print("\n[Step 2] Setting up confirmation manager...")
    
    cm = ConfirmationManager(default_timeout=10, max_retries=2)
    print("✓ Confirmation manager ready (10s timeout, 2 max retries)")
    
    # Step 3: Simulate permission checks
    print("\n[Step 3] Testing permission checks...")
    
    test_cases = [
        {
            "tool": "get_weather",
            "speaker": None,  # Anonymous
            "description": "Anonymous user - get_weather (low sensitivity)",
        },
        {
            "tool": "view_calendar",
            "speaker": "speaker_002",  # Guest
            "description": "Guest - view_calendar (medium sensitivity)",
        },
        {
            "tool": "unlock_door",
            "speaker": "speaker_001",  # Owner
            "description": "Owner - unlock_door (critical sensitivity)",
        },
        {
            "tool": "unlock_door",
            "speaker": "speaker_002",  # Guest
            "description": "Guest - unlock_door (critical sensitivity)",
        },
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n  Test {i}: {case['description']}")
        action = pm.check_permission(
            tool_name=case["tool"],
            speaker_id=case["speaker"],
        )
        print(f"  → Permission: {action.value}")
        
        if action.value == "ask":
            print("  → Requires two-factor confirmation")
    
    # Step 4: Simulate two-factor confirmation flow
    print("\n[Step 4] Testing two-factor confirmation...")
    
    session_id = "test_session_001"
    tool_name = "unlock_door"
    original_command = "打开门锁"
    
    # Request confirmation
    print(f"\n  User says: '{original_command}'")
    print(f"  Tool: {tool_name} (critical)")
    
    prompt = cm.request_confirmation(
        session_id=session_id,
        tool_name=tool_name,
        arguments={"location": "front_door"},
        original_command=original_command,
        speaker_id="speaker_001",
    )
    
    print(f"  System response: {prompt}")
    
    # Step 5: Test confirmation recognition
    print("\n[Step 5] Testing confirmation recognition...")
    
    confirmation_tests = [
        {
            "input": "确认",
            "expected": "confirmed",
            "description": "Keyword confirmation",
        },
        {
            "input": "打开门锁",
            "expected": "confirmed",
            "description": "Command repetition",
        },
        {
            "input": "取消",
            "expected": "rejected",
            "description": "Rejection keyword",
        },
        {
            "input": "什么",
            "expected": "pending",
            "description": "Unclear input (retry)",
        },
    ]
    
    for test in confirmation_tests:
        print(f"\n  Test: {test['description']}")
        print(f"  User input: '{test['input']}'")
        
        # Create fresh confirmation for each test
        cm.request_confirmation(
            session_id=f"session_{test['description']}",
            tool_name=tool_name,
            arguments={"location": "front_door"},
            original_command=original_command,
            speaker_id="speaker_001",
        )
        
        result = cm.check_confirmation(
            session_id=f"session_{test['description']}",
            user_input=test["input"],
        )
        
        print(f"  → State: {result.state.value}")
        print(f"  → Message: {result.message}")
        
        if result.state.value == test["expected"]:
            print("  ✓ PASS")
        else:
            print(f"  ✗ FAIL (expected {test['expected']}, got {result.state.value})")
    
    # Step 6: Test retry logic
    print("\n[Step 6] Testing retry logic...")
    
    retry_session = "retry_session"
    cm.request_confirmation(
        session_id=retry_session,
        tool_name=tool_name,
        arguments={"location": "front_door"},
        original_command=original_command,
        speaker_id="speaker_001",
    )
    
    print(f"  Initial request: {original_command}")
    
    # First unclear input
    result1 = cm.check_confirmation(retry_session, "什么")
    print(f"\n  Attempt 1: '什么'")
    print(f"  → State: {result1.state.value}")
    print(f"  → Message: {result1.message}")
    
    # Second unclear input
    result2 = cm.check_confirmation(retry_session, "嗯")
    print(f"\n  Attempt 2: '嗯'")
    print(f"  → State: {result2.state.value}")
    print(f"  → Message: {result2.message}")
    
    # Third unclear input (should cancel)
    result3 = cm.check_confirmation(retry_session, "啊")
    print(f"\n  Attempt 3: '啊'")
    print(f"  → State: {result3.state.value}")
    print(f"  → Message: {result3.message}")
    
    if result3.state.value == "cancelled":
        print("  ✓ Correctly cancelled after max retries")
    
    # Step 7: Test timeout
    print("\n[Step 7] Testing timeout...")
    
    timeout_session = "timeout_session"
    
    # Create confirmation with very short timeout for testing
    pending = cm._pending[timeout_session] = __import__("loom.adapters.audio.confirmation", fromlist=["PendingConfirmation"]).PendingConfirmation(
        tool_name=tool_name,
        arguments={"location": "front_door"},
        original_command=original_command,
        speaker_id="speaker_001",
        timeout_seconds=0,  # Immediate timeout
    )
    
    print("  Creating confirmation with 0s timeout...")
    await asyncio.sleep(0.1)  # Small delay
    
    result = cm.check_confirmation(timeout_session, "确认")
    print(f"  → State: {result.state.value}")
    print(f"  → Message: {result.message}")
    
    if result.state.value == "timeout":
        print("  ✓ Correctly detected timeout")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


async def demo_complete_workflow():
    """Demonstrate complete workflow with AudioAdapter."""
    
    print("\n\n" + "=" * 60)
    print("Complete Workflow Demo (with AudioAdapter)")
    print("=" * 60)
    
    # Create adapter
    config = AudioAdapterConfig(
        host="0.0.0.0",
        port=8765,
        voiceprint_enabled=False,  # Disable for demo
    )
    
    adapter = AudioAdapter(config)
    
    # Manually initialize managers (without starting full server)
    adapter.permission_manager = AudioPermissionManager.from_config()
    adapter.permission_manager.add_owner("speaker_001")
    adapter.confirmation_manager = ConfirmationManager()
    
    print("\n✓ AudioAdapter configured")
    
    # Simulate tool execution request
    print("\n[Scenario] Owner requests critical operation")
    
    session_id = "demo_session"
    speaker_id = "speaker_001"  # From voiceprint verification
    tool_name = "unlock_door"
    original_command = "打开门锁"
    
    print(f"  Speaker: {speaker_id} (owner)")
    print(f"  Command: '{original_command}'")
    
    # Step 1: Check permission
    action = adapter.check_tool_permission(
        tool_name=tool_name,
        speaker_id=speaker_id,
    )
    
    print(f"\n  Permission check: {action}")
    
    if action == "ask":
        # Step 2: Request confirmation
        prompt = adapter.request_confirmation(
            session_id=session_id,
            tool_name=tool_name,
            arguments={"location": "front_door"},
            original_command=original_command,
            speaker_id=speaker_id,
        )
        
        print(f"  System prompt: {prompt}")
        
        # Step 3: User confirms
        user_response = "确认"
        print(f"\n  User response: '{user_response}'")
        
        result = adapter.check_confirmation(session_id, user_response)
        
        print(f"  Confirmation state: {result['state']}")
        print(f"  Message: {result['message']}")
        
        if result["state"] == "confirmed":
            print(f"\n  ✓ Executing tool: {result['tool_name']}")
            print(f"  ✓ Arguments: {result['arguments']}")
        
    print("\n" + "=" * 60)
    print("Complete workflow demo finished!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo_two_factor_confirmation())
    asyncio.run(demo_complete_workflow())
