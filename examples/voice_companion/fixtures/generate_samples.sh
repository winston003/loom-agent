#!/bin/bash
# Generate placeholder audio samples for voiceprint testing
# This creates silent WAV files in the correct format for testing

set -e

echo "Generating placeholder audio samples..."
echo "================================================"

# Create fixtures directory if it doesn't exist
mkdir -p "$(dirname "$0")"

# Function to generate silent WAV file
generate_silent_wav() {
    local filename=$1
    local duration=$2
    
    # Generate 16kHz mono silent audio using ffmpeg
    ffmpeg -f lavfi -i anullsrc=r=16000:cl=mono -t "$duration" -acodec pcm_s16le -y "$filename" 2>/dev/null
    
    echo "✓ Created: $filename (${duration}s, 16kHz mono PCM)"
}

# Generate samples for User A (Alice - 3 samples)
echo ""
echo "Generating User A (Alice) samples..."
generate_silent_wav "user_A_sample1.wav" 2.5
generate_silent_wav "user_A_sample2.wav" 2.5
generate_silent_wav "user_A_sample3.wav" 2.5

# Generate samples for User B (Bob - 4 samples)
echo ""
echo "Generating User B (Bob) samples..."
generate_silent_wav "user_B_sample1.wav" 2.5
generate_silent_wav "user_B_sample2.wav" 2.5
generate_silent_wav "user_B_sample3.wav" 2.5
generate_silent_wav "user_B_sample4.wav" 2.5

# Generate samples for User C (Charlie - 3 samples)
echo ""
echo "Generating User C (Charlie) samples..."
generate_silent_wav "user_C_sample1.wav" 2.5
generate_silent_wav "user_C_sample2.wav" 2.5
generate_silent_wav "user_C_sample3.wav" 2.5

echo ""
echo "================================================"
echo "✓ All placeholder samples generated!"
echo ""
echo "⚠️  NOTE: These are SILENT placeholder files for testing"
echo "   For real voiceprint testing, record actual speech:"
echo ""
echo "   1. Use ffmpeg: ffmpeg -f avfoundation -i ':0' -ar 16000 -ac 1 -t 3 user_A_sample1.wav"
echo "   2. Use sox: sox -d -r 16000 -c 1 user_A_sample1.wav trim 0 3"
echo "   3. See fixtures/README.md for more recording options"
echo ""
echo "Sample count:"
ls -1 user_*.wav 2>/dev/null | wc -l | xargs echo "  Total samples:"
echo ""
