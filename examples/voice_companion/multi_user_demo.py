"""Demo: Multi-user voiceprint management.

This example demonstrates voiceprint management operations:
1. List all registered voiceprints
2. View voiceprint statistics
3. Check voiceprint limit
4. Delete voiceprints
5. User limit enforcement

Reference: specs/002-xiaozhi-voice-adapter/tasks.md T070-T072
"""

import asyncio
import json
from pathlib import Path
from loom.adapters.audio import AudioAdapter, AudioAdapterConfig
from loom.adapters.audio.voiceprint.storage import VoiceprintStorage


async def demo_multi_user_management():
    """Demonstrate multi-user management operations."""
    
    print("=" * 70)
    print("Multi-User Voiceprint Management Demo")
    print("=" * 70)
    
    # Setup
    device_id = "demo_device_001"
    
    # Create storage for demo
    storage = VoiceprintStorage(
        encryption_key=b"demo_key_16bytes",
        base_dir="/tmp/loom_demo/voiceprints"
    )
    
    # Step 1: Create test voiceprints
    print("\n[Step 1] Creating test voiceprints...")
    
    test_users = [
        {"speaker_id": "speaker_001", "display_name": "Alice", "verifications": 42},
        {"speaker_id": "speaker_002", "display_name": "Bob", "verifications": 15},
        {"speaker_id": "speaker_003", "display_name": "Charlie", "verifications": 8},
        {"speaker_id": "speaker_004", "display_name": "Diana", "verifications": 3},
    ]
    
    for user in test_users:
        await storage.save_voiceprint(
            device_id=device_id,
            speaker_id=user["speaker_id"],
            display_name=user["display_name"],
            metadata={
                "created_at": 1700000000,
                "sample_count": 3,
            }
        )
        
        # Simulate verifications
        for _ in range(user["verifications"]):
            await storage.update_verification_stats(device_id, user["speaker_id"])
        
        print(f"  ✓ Created: {user['display_name']} ({user['speaker_id']})")
    
    # Step 2: List voiceprints (T070)
    print("\n[Step 2] Listing all voiceprints...")
    
    adapter = AudioAdapter(AudioAdapterConfig(voiceprint_enabled=False))
    adapter.voiceprint_storage = storage
    
    voiceprints = adapter.list_voiceprints(device_id)
    
    print(f"\n  Total voiceprints: {len(voiceprints)}")
    print("\n  Registered users (sorted by activity):")
    print("  " + "-" * 66)
    print(f"  {'Display Name':<15} {'Speaker ID':<15} {'Verifications':<15} {'Role':<10}")
    print("  " + "-" * 66)
    
    for vp in voiceprints:
        print(
            f"  {vp['display_name']:<15} "
            f"{vp['speaker_id']:<15} "
            f"{vp['verification_count']:<15} "
            f"{'Active' if vp['verification_count'] > 10 else 'Inactive':<10}"
        )
    
    print("  " + "-" * 66)
    
    # Step 3: Get voiceprint statistics (T070)
    print("\n[Step 3] Voiceprint statistics...")
    
    stats = adapter.get_voiceprint_stats(device_id)
    
    print(f"\n  Total voiceprints: {stats['total_voiceprints']}")
    print(f"  Total verifications: {stats['total_verifications']}")
    print(f"  Most active user: {stats['most_active_user']} ({stats['most_active_count']} verifications)")
    print(f"  Least active user: {stats['least_active_user']} ({stats['least_active_count']} verifications)")
    
    # Step 4: Check voiceprint limit (T072)
    print("\n[Step 4] Checking voiceprint limit...")
    
    can_add, current_count = adapter.check_voiceprint_limit(device_id, max_voiceprints=5)
    
    print(f"\n  Current count: {current_count}/5")
    print(f"  Can add more: {'Yes' if can_add else 'No'}")
    
    if can_add:
        print(f"  → {5 - current_count} slot(s) available")
    else:
        print("  → Limit reached! Cannot add more voiceprints.")
    
    # Step 5: Test limit enforcement
    print("\n[Step 5] Testing limit enforcement...")
    
    # Add one more user to reach limit
    await storage.save_voiceprint(
        device_id=device_id,
        speaker_id="speaker_005",
        display_name="Eve",
        metadata={"created_at": 1700000000, "sample_count": 3}
    )
    
    print("  ✓ Added Eve (5/5)")
    
    # Try to add another (should fail in real adapter)
    can_add, current_count = adapter.check_voiceprint_limit(device_id, max_voiceprints=5)
    
    print(f"\n  After adding Eve:")
    print(f"  Current count: {current_count}/5")
    print(f"  Can add more: {'Yes' if can_add else 'No'}")
    
    if not can_add:
        print("\n  ✓ Limit enforcement working correctly!")
        print("  → Would reject new registration with error:")
        print('     "Voiceprint limit reached: 5/5 users already registered."')
    
    # Step 6: Delete voiceprint (T071)
    print("\n[Step 6] Deleting a voiceprint...")
    
    # Delete least active user
    delete_speaker = "speaker_004"  # Diana
    
    print(f"\n  Target: Diana ({delete_speaker})")
    
    deleted = await adapter.delete_voiceprint(
        device_id=device_id,
        speaker_id=delete_speaker,
        remove_from_permissions=False,  # For demo, don't modify real config
    )
    
    if deleted:
        print("  ✓ Voiceprint deleted successfully")
    
    # Verify deletion
    voiceprints_after = adapter.list_voiceprints(device_id)
    remaining_names = [vp['display_name'] for vp in voiceprints_after]
    
    print(f"\n  Remaining users: {', '.join(remaining_names)}")
    print(f"  Count: {len(voiceprints_after)}/5")
    
    # Check if can add now
    can_add, current_count = adapter.check_voiceprint_limit(device_id, max_voiceprints=5)
    
    if can_add:
        print(f"\n  ✓ Slot freed! Can now add {5 - current_count} more user(s)")
    
    # Step 7: View detailed metadata
    print("\n[Step 7] Viewing detailed metadata...")
    
    alice_metadata = storage.load_metadata(device_id, "speaker_001")
    
    if alice_metadata:
        print(f"\n  User: {alice_metadata['display_name']}")
        print(f"  Speaker ID: {alice_metadata['speaker_id']}")
        print(f"  Created: {alice_metadata.get('created_at', 'N/A')}")
        print(f"  Sample count: {alice_metadata.get('sample_count', 'N/A')}")
        print(f"  Verifications: {alice_metadata.get('verification_count', 0)}")
        
        if alice_metadata.get('last_verified_at'):
            import datetime
            last_verified = datetime.datetime.fromtimestamp(alice_metadata['last_verified_at'])
            print(f"  Last verified: {last_verified.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "=" * 70)
    print("Multi-user management demo complete!")
    print("=" * 70)


async def demo_user_limit_scenarios():
    """Demonstrate user limit scenarios."""
    
    print("\n\n" + "=" * 70)
    print("User Limit Scenarios Demo")
    print("=" * 70)
    
    device_id = "limit_test_device"
    
    # Create fresh storage
    storage = VoiceprintStorage(
        encryption_key=b"test_key_16bytes",
        base_dir="/tmp/loom_demo/limit_test"
    )
    
    adapter = AudioAdapter(AudioAdapterConfig(voiceprint_enabled=False))
    adapter.voiceprint_storage = storage
    
    # Scenario 1: Adding users up to limit
    print("\n[Scenario 1] Adding users up to limit...")
    
    for i in range(5):
        user_name = f"User_{i+1}"
        speaker_id = f"speaker_{i+1:03d}"
        
        can_add, count = adapter.check_voiceprint_limit(device_id, max_voiceprints=5)
        
        if can_add:
            await storage.save_voiceprint(
                device_id=device_id,
                speaker_id=speaker_id,
                display_name=user_name,
                metadata={"created_at": 1700000000, "sample_count": 3}
            )
            print(f"  ✓ Added {user_name} ({count + 1}/5)")
        else:
            print(f"  ✗ Cannot add {user_name} - limit reached ({count}/5)")
    
    # Scenario 2: Attempting to add beyond limit
    print("\n[Scenario 2] Attempting to add beyond limit...")
    
    can_add, count = adapter.check_voiceprint_limit(device_id, max_voiceprints=5)
    
    if not can_add:
        print(f"  Current: {count}/5 users registered")
        print("  → Attempting to add User_6...")
        print("  ✗ Registration blocked - limit reached!")
        print("\n  Error message that would be shown:")
        print('  "Voiceprint limit reached: 5/5 users already registered."')
        print('  "Please delete an existing voiceprint before adding a new one."')
    
    # Scenario 3: Delete and add workflow
    print("\n[Scenario 3] Delete-then-add workflow...")
    
    print(f"\n  Step 1: Current state: {count}/5 users")
    
    # Delete first user
    deleted = await adapter.delete_voiceprint(device_id, "speaker_001")
    
    if deleted:
        print("  Step 2: Deleted User_1")
        
        can_add, count = adapter.check_voiceprint_limit(device_id, max_voiceprints=5)
        print(f"  Step 3: New state: {count}/5 users")
        print(f"  Step 4: Can add: {can_add}")
        
        if can_add:
            await storage.save_voiceprint(
                device_id=device_id,
                speaker_id="speaker_006",
                display_name="User_6",
                metadata={"created_at": 1700000000, "sample_count": 3}
            )
            print("  ✓ Successfully added User_6 in freed slot")
    
    # Final stats
    print("\n[Final Stats]")
    stats = adapter.get_voiceprint_stats(device_id)
    print(f"  Total users: {stats['total_voiceprints']}/5")
    
    voiceprints = adapter.list_voiceprints(device_id)
    names = [vp['display_name'] for vp in voiceprints]
    print(f"  Registered: {', '.join(names)}")
    
    print("\n" + "=" * 70)
    print("User limit scenarios demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo_multi_user_management())
    asyncio.run(demo_user_limit_scenarios())
