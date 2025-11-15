"""Voiceprint Demo: Complete registration and verification workflow.

This example demonstrates the full voiceprint feature set:
1. Register multiple users with audio samples
2. Verify speakers from audio
3. Permission-based access control
4. Two-factor confirmation for critical operations
5. Multi-user management

Reference: specs/002-xiaozhi-voice-adapter/tasks.md T073
"""

import asyncio
from loom.adapters.audio import AudioAdapter, AudioAdapterConfig
from loom.adapters.audio.permissions import AudioPermissionManager


async def demo_voiceprint_workflow():
    """Demonstrate complete voiceprint workflow."""
    
    print("=" * 70)
    print("Voiceprint Registration & Verification Demo")
    print("=" * 70)
    
    # Step 1: Setup AudioAdapter with voiceprint enabled
    print("\n[Step 1] Setting up AudioAdapter...")
    
    config = AudioAdapterConfig(
        host="0.0.0.0",
        port=8765,
        voiceprint_enabled=True,
        voiceprint_url="http://localhost:8000",  # 3DSpeaker service
    )
    
    adapter = AudioAdapter(config)
    
    # For demo, manually initialize components (without starting server)
    from loom.adapters.audio.voiceprint.storage import VoiceprintStorage
    from loom.adapters.audio.voiceprint.client import ThreeDSpeakerClient
    from loom.adapters.audio.confirmation import ConfirmationManager
    
    adapter.voiceprint_storage = VoiceprintStorage(
        encryption_key=b"demo_key_16bytes",
        base_dir="/tmp/loom_voiceprint_demo/voiceprints"
    )
    
    adapter.voiceprint_client = ThreeDSpeakerClient(
        base_url="http://localhost:8000"
    )
    
    adapter.permission_manager = AudioPermissionManager.from_config()
    adapter.confirmation_manager = ConfirmationManager()
    
    print("✓ AudioAdapter configured")
    print("✓ Voiceprint storage: /tmp/loom_voiceprint_demo/voiceprints")
    print("✓ 3DSpeaker service: http://localhost:8000")
    
    device_id = "demo_device_001"
    
    # Step 2: Register users (T073 - T074)
    print("\n[Step 2] Registering users...")
    
    # Note: In real usage, audio_samples would be actual audio data
    # For demo, we'll simulate the registration process
    
    users_to_register = [
        {
            "display_name": "Alice (Owner)",
            "role": "owner",
            "sample_count": 3,
        },
        {
            "display_name": "Bob (Guest)",
            "role": "guest",
            "sample_count": 4,
        },
        {
            "display_name": "Charlie (Guest)",
            "role": "guest",
            "sample_count": 3,
        },
    ]
    
    print("\n  Registration instructions:")
    print("  - Each user needs 3-5 audio samples (2-3 seconds each)")
    print("  - Samples should be recorded in quiet environment")
    print("  - Use consistent speaking style")
    print()
    
    registered_users = []
    
    for i, user_info in enumerate(users_to_register, 1):
        print(f"  User {i}: {user_info['display_name']}")
        print(f"    Samples needed: {user_info['sample_count']}")
        
        # Check voiceprint limit before registration
        can_add, current_count = adapter.check_voiceprint_limit(device_id)
        
        if not can_add:
            print(f"    ✗ Cannot register - limit reached ({current_count}/5)")
            continue
        
        print(f"    Current voiceprints: {current_count}/5")
        
        # Simulate registration (in real app, would call register_voiceprint with actual audio)
        speaker_id = f"speaker_{i:03d}"
        
        # Simulate saving metadata
        await adapter.voiceprint_storage.save_voiceprint(
            device_id=device_id,
            speaker_id=speaker_id,
            display_name=user_info['display_name'],
            metadata={
                "created_at": 1700000000 + i * 1000,
                "sample_count": user_info['sample_count'],
            }
        )
        
        # Add to permission manager
        if user_info['role'] == 'owner':
            adapter.permission_manager.add_owner(speaker_id)
            print(f"    ✓ Registered as OWNER")
        else:
            adapter.permission_manager.add_guest(speaker_id)
            print(f"    ✓ Registered as GUEST")
        
        registered_users.append({
            "speaker_id": speaker_id,
            **user_info
        })
        
        print(f"    Speaker ID: {speaker_id}")
        print()
    
    # Step 3: List registered voiceprints
    print("\n[Step 3] Listing registered voiceprints...")
    
    voiceprints = adapter.list_voiceprints(device_id)
    
    print(f"\n  Total voiceprints: {len(voiceprints)}")
    print("\n  " + "-" * 66)
    print(f"  {'Display Name':<20} {'Speaker ID':<15} {'Role':<10} {'Samples':<8}")
    print("  " + "-" * 66)
    
    for vp in voiceprints:
        speaker_id = vp['speaker_id']
        is_owner = adapter.permission_manager.is_owner(speaker_id)
        is_guest = adapter.permission_manager.is_guest(speaker_id)
        role = "OWNER" if is_owner else "GUEST" if is_guest else "UNKNOWN"
        
        print(
            f"  {vp['display_name']:<20} "
            f"{vp['speaker_id']:<15} "
            f"{role:<10} "
            f"{vp.get('sample_count', 0):<8}"
        )
    
    print("  " + "-" * 66)
    
    # Step 4: Simulate speaker verification and permission checks
    print("\n[Step 4] Testing permission-based access control...")
    
    test_scenarios = [
        {
            "speaker_id": "speaker_001",  # Alice (Owner)
            "operation": "view_calendar",
            "description": "Owner accessing calendar (medium sensitivity)",
        },
        {
            "speaker_id": "speaker_002",  # Bob (Guest)
            "operation": "view_calendar",
            "description": "Guest accessing calendar (medium sensitivity)",
        },
        {
            "speaker_id": None,  # Anonymous
            "operation": "view_calendar",
            "description": "Anonymous accessing calendar (medium sensitivity)",
        },
        {
            "speaker_id": "speaker_001",  # Alice (Owner)
            "operation": "unlock_door",
            "description": "Owner attempting door unlock (critical sensitivity)",
        },
        {
            "speaker_id": "speaker_002",  # Bob (Guest)
            "operation": "unlock_door",
            "description": "Guest attempting door unlock (critical sensitivity)",
        },
        {
            "speaker_id": "speaker_001",  # Alice (Owner)
            "operation": "get_weather",
            "description": "Owner checking weather (low sensitivity)",
        },
    ]
    
    print()
    for i, scenario in enumerate(test_scenarios, 1):
        speaker = scenario['speaker_id'] or "anonymous"
        
        print(f"\n  Scenario {i}: {scenario['description']}")
        print(f"    Speaker: {speaker}")
        print(f"    Operation: {scenario['operation']}")
        
        action = adapter.check_tool_permission(
            tool_name=scenario['operation'],
            speaker_id=scenario['speaker_id'],
        )
        
        print(f"    → Permission: {action.upper()}")
        
        if action == "allow":
            print("    ✓ Access granted - executing operation")
        elif action == "deny":
            print("    ✗ Access denied - insufficient permissions")
        elif action == "ask":
            print("    ⚠ Requires two-factor confirmation")
            
            # Simulate confirmation flow
            session_id = f"session_{i}"
            
            prompt = adapter.request_confirmation(
                session_id=session_id,
                tool_name=scenario['operation'],
                arguments={},
                original_command=scenario['operation'].replace('_', ' '),
                speaker_id=scenario['speaker_id'],
            )
            
            print(f"    System: {prompt}")
            
            # Simulate user confirmation
            user_response = "确认"
            print(f"    User: {user_response}")
            
            result = adapter.check_confirmation(session_id, user_response)
            
            if result['state'] == 'confirmed':
                print("    ✓ Confirmation successful - executing operation")
            else:
                print(f"    ✗ Confirmation {result['state']} - operation cancelled")
    
    # Step 5: Voiceprint statistics
    print("\n[Step 5] Voiceprint statistics...")
    
    stats = adapter.get_voiceprint_stats(device_id)
    
    print(f"\n  Total voiceprints: {stats['total_voiceprints']}")
    print(f"  Total verifications: {stats['total_verifications']}")
    
    if stats['most_active_user']:
        print(f"  Most active: {stats['most_active_user']} ({stats['most_active_count']} verifications)")
    
    # Step 6: User limit check
    print("\n[Step 6] Checking user limit...")
    
    can_add, count = adapter.check_voiceprint_limit(device_id, max_voiceprints=5)
    
    print(f"\n  Current: {count}/5 voiceprints")
    print(f"  Can add more: {'Yes' if can_add else 'No'}")
    
    if can_add:
        print(f"  → {5 - count} slot(s) available")
    else:
        print("  → Limit reached!")
        print("  → Delete a voiceprint to add a new user")
    
    # Step 7: Delete a voiceprint
    print("\n[Step 7] Deleting a voiceprint...")
    
    if len(voiceprints) > 0:
        # Delete the last guest
        to_delete = None
        for vp in reversed(voiceprints):
            if adapter.permission_manager.is_guest(vp['speaker_id']):
                to_delete = vp
                break
        
        if to_delete:
            print(f"\n  Deleting: {to_delete['display_name']} ({to_delete['speaker_id']})")
            
            deleted = await adapter.delete_voiceprint(
                device_id=device_id,
                speaker_id=to_delete['speaker_id'],
                remove_from_permissions=True,
            )
            
            if deleted:
                print("  ✓ Voiceprint deleted successfully")
                
                # Check new count
                can_add, count = adapter.check_voiceprint_limit(device_id)
                print(f"  New count: {count}/5")
                
                if can_add:
                    print(f"  ✓ Can now add {5 - count} more user(s)")
    
    print("\n" + "=" * 70)
    print("Voiceprint demo complete!")
    print("=" * 70)
    
    # Cleanup instructions
    print("\n📝 Next Steps:")
    print("  1. Start 3DSpeaker service: docker run -p 8000:8000 3dspeaker")
    print("  2. Use register_voiceprint.py to register real users with audio")
    print("  3. Test with actual audio samples from fixtures/")
    print("  4. Configure permissions in ~/.loom/audio_config.json")


if __name__ == "__main__":
    asyncio.run(demo_voiceprint_workflow())
