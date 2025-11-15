#!/usr/bin/env python3
"""Voiceprint Registration Tool

Interactive CLI tool for registering users with voiceprints.

Usage:
    python register_voiceprint.py --device-id xiaozhi-001 --name Alice --role owner
    python register_voiceprint.py --device-id xiaozhi-001 --name Bob --role guest --samples 4

Features:
- Interactive audio sample collection (or from files)
- Voiceprint limit checking (max 5 users per device)
- Automatic permission configuration (owner/guest role)
- Progress feedback and error handling

Reference: specs/002-xiaozhi-voice-adapter/tasks.md T074
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loom.adapters.audio import AudioAdapter, AudioAdapterConfig
from loom.adapters.audio.permissions import AudioPermissionManager
from loom.adapters.audio.voiceprint.storage import VoiceprintStorage
from loom.adapters.audio.voiceprint.client import ThreeDSpeakerClient


def load_audio_samples_from_files(sample_paths: List[str]) -> List[bytes]:
    """Load audio samples from WAV files.
    
    Args:
        sample_paths: List of paths to WAV files
        
    Returns:
        List of audio bytes (PCM format)
    """
    samples = []
    
    for path in sample_paths:
        try:
            with open(path, 'rb') as f:
                audio_data = f.read()
                samples.append(audio_data)
            print(f"  ✓ Loaded: {path} ({len(audio_data)} bytes)")
        except Exception as e:
            print(f"  ✗ Failed to load {path}: {e}")
    
    return samples


async def register_voiceprint(
    device_id: str,
    display_name: str,
    role: str,
    sample_paths: List[str],
    voiceprint_url: str = "http://localhost:8000",
    config_path: str = None,
):
    """Register a new voiceprint.
    
    Args:
        device_id: Device ID (e.g., "xiaozhi-001")
        display_name: User display name (e.g., "Alice")
        role: User role ("owner" or "guest")
        sample_paths: List of paths to audio sample files
        voiceprint_url: 3DSpeaker service URL
        config_path: Optional custom config path
    """
    print("=" * 70)
    print("Voiceprint Registration Tool")
    print("=" * 70)
    
    print(f"\nDevice ID: {device_id}")
    print(f"User: {display_name}")
    print(f"Role: {role.upper()}")
    print(f"Samples: {len(sample_paths)}")
    print(f"3DSpeaker URL: {voiceprint_url}")
    
    # Step 1: Setup adapter
    print("\n[Step 1] Setting up AudioAdapter...")
    
    config = AudioAdapterConfig(
        voiceprint_enabled=True,
        voiceprint_url=voiceprint_url,
    )
    
    adapter = AudioAdapter(config)
    
    # Initialize components
    adapter.voiceprint_storage = VoiceprintStorage(
        encryption_key=b"production_key__",  # In production, use secure key management
        base_dir=Path.home() / ".loom" / "voiceprints"
    )
    
    adapter.voiceprint_client = ThreeDSpeakerClient(base_url=voiceprint_url)
    
    adapter.permission_manager = AudioPermissionManager.from_config(config_path)
    
    print("✓ AudioAdapter configured")
    
    # Step 2: Check voiceprint limit
    print("\n[Step 2] Checking voiceprint limit...")
    
    can_add, current_count = adapter.check_voiceprint_limit(device_id, max_voiceprints=5)
    
    print(f"  Current voiceprints: {current_count}/5")
    
    if not can_add:
        print("\n✗ ERROR: Voiceprint limit reached (5/5)")
        print("\nTo add a new user, first delete an existing voiceprint:")
        print("  1. List existing users: python list_voiceprints.py --device-id", device_id)
        print("  2. Delete a user: python delete_voiceprint.py --device-id", device_id, "--speaker-id <id>")
        return False
    
    print(f"✓ Can add user ({5 - current_count} slot(s) available)")
    
    # Step 3: Load audio samples
    print("\n[Step 3] Loading audio samples...")
    
    print(f"  Sample files: {len(sample_paths)}")
    
    if len(sample_paths) < 3:
        print("\n✗ ERROR: At least 3 audio samples required")
        print("  Provide 3-5 WAV files (2-3 seconds each, 16kHz mono PCM)")
        return False
    
    if len(sample_paths) > 5:
        print("\n✗ ERROR: Maximum 5 audio samples allowed")
        print("  Please provide only 3-5 samples")
        return False
    
    audio_samples = load_audio_samples_from_files(sample_paths)
    
    if len(audio_samples) != len(sample_paths):
        print(f"\n✗ ERROR: Failed to load some samples")
        return False
    
    print(f"✓ All {len(audio_samples)} samples loaded")
    
    # Step 4: Register voiceprint
    print("\n[Step 4] Registering voiceprint with 3DSpeaker...")
    
    try:
        speaker_id = await adapter.register_voiceprint(
            device_id=device_id,
            display_name=display_name,
            audio_samples=audio_samples,
        )
        
        print(f"✓ Registration successful!")
        print(f"  Speaker ID: {speaker_id}")
        
    except Exception as e:
        print(f"\n✗ Registration failed: {e}")
        return False
    
    # Step 5: Configure permissions
    print("\n[Step 5] Configuring permissions...")
    
    if role == "owner":
        adapter.permission_manager.add_owner(speaker_id)
        print(f"✓ Added {display_name} as OWNER")
    elif role == "guest":
        adapter.permission_manager.add_guest(speaker_id)
        print(f"✓ Added {display_name} as GUEST")
    else:
        print(f"⚠ Warning: Unknown role '{role}', skipping permission config")
    
    # Step 6: Verify registration
    print("\n[Step 6] Verifying registration...")
    
    metadata = adapter.voiceprint_storage.load_metadata(device_id, speaker_id)
    
    if metadata:
        print("✓ Voiceprint metadata saved:")
        print(f"  Display name: {metadata['display_name']}")
        print(f"  Speaker ID: {metadata['speaker_id']}")
        print(f"  Sample count: {metadata.get('sample_count', 'N/A')}")
        print(f"  Created: {metadata.get('created_at', 'N/A')}")
    
    # Show updated stats
    can_add, count = adapter.check_voiceprint_limit(device_id)
    print(f"\n  Total voiceprints: {count}/5")
    
    print("\n" + "=" * 70)
    print("✓ Registration complete!")
    print("=" * 70)
    
    return True


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Register a user voiceprint for speaker verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register owner with 3 samples
  python register_voiceprint.py --device-id xiaozhi-001 --name Alice --role owner \\
      --samples fixtures/alice_sample1.wav fixtures/alice_sample2.wav fixtures/alice_sample3.wav
  
  # Register guest with 4 samples
  python register_voiceprint.py --device-id xiaozhi-001 --name Bob --role guest \\
      --samples fixtures/bob_*.wav
  
  # Use custom 3DSpeaker service URL
  python register_voiceprint.py --device-id xiaozhi-001 --name Charlie --role guest \\
      --samples fixtures/charlie_*.wav --url http://192.168.1.100:8000
        """
    )
    
    parser.add_argument(
        '--device-id',
        required=True,
        help='Device ID (e.g., "xiaozhi-001")'
    )
    
    parser.add_argument(
        '--name',
        required=True,
        help='User display name (e.g., "Alice")'
    )
    
    parser.add_argument(
        '--role',
        choices=['owner', 'guest'],
        default='guest',
        help='User role (owner or guest, default: guest)'
    )
    
    parser.add_argument(
        '--samples',
        nargs='+',
        required=True,
        help='Audio sample files (3-5 WAV files, 16kHz mono PCM, 2-3s each)'
    )
    
    parser.add_argument(
        '--url',
        default='http://localhost:8000',
        help='3DSpeaker service URL (default: http://localhost:8000)'
    )
    
    parser.add_argument(
        '--config',
        help='Custom audio config path (default: ~/.loom/audio_config.json)'
    )
    
    args = parser.parse_args()
    
    # Run registration
    success = await register_voiceprint(
        device_id=args.device_id,
        display_name=args.name,
        role=args.role,
        sample_paths=args.samples,
        voiceprint_url=args.url,
        config_path=args.config,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
