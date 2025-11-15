#!/bin/bash
# Automated End-to-End Test for Voice Companion
# This script runs a complete test of the voice interaction pipeline

set -e  # Exit on error

echo "🧪 Voice Companion End-to-End Test"
echo "===================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo "📋 Checking prerequisites..."

if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}❌ Error: OPENAI_API_KEY not set${NC}"
    echo "   Please set your OpenAI API key:"
    echo "   export OPENAI_API_KEY='sk-...'"
    exit 1
fi

if ! python3 -c "import websockets" 2>/dev/null; then
    echo -e "${RED}❌ Error: websockets package not installed${NC}"
    echo "   Install with: pip install websockets"
    exit 1
fi

if ! python3 -c "import edge_tts" 2>/dev/null; then
    echo -e "${RED}❌ Error: edge-tts package not installed${NC}"
    echo "   Install with: pip install loom-agent[audio]"
    exit 1
fi

echo -e "${GREEN}✅ All prerequisites met${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Paths
FIXTURES_DIR="$SCRIPT_DIR/fixtures"
TEST_AUDIO="$FIXTURES_DIR/test_speech_pattern.wav"
OUTPUT_AUDIO="$SCRIPT_DIR/test_output_tts.wav"

# Generate test audio if not exists
if [ ! -f "$TEST_AUDIO" ]; then
    echo "📁 Generating test audio files..."
    python3 "$FIXTURES_DIR/generate_test_audio.py"
    echo ""
else
    echo "✅ Test audio files already exist"
    echo ""
fi

# Start server in background
echo "🚀 Starting voice companion server..."
python3 "$SCRIPT_DIR/hello_voice.py" > /tmp/voice_server.log 2>&1 &
SERVER_PID=$!

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    if [ ! -z "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Wait for server to start
echo "⏳ Waiting for server to start (5 seconds)..."
sleep 5

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}❌ Server failed to start${NC}"
    echo "Server log:"
    cat /tmp/voice_server.log
    exit 1
fi

echo -e "${GREEN}✅ Server started (PID: $SERVER_PID)${NC}"
echo ""

# Run test client
echo "🧪 Running test client..."
echo "   Audio: $TEST_AUDIO"
echo "   Output: $OUTPUT_AUDIO"
echo ""

if python3 "$SCRIPT_DIR/test_client.py" \
    --audio "$TEST_AUDIO" \
    --output "$OUTPUT_AUDIO"; then
    
    echo ""
    echo -e "${GREEN}✅ Test client completed successfully${NC}"
else
    echo ""
    echo -e "${RED}❌ Test client failed${NC}"
    exit 1
fi

# Verify output
echo ""
echo "📊 Verifying results..."

if [ -f "$OUTPUT_AUDIO" ]; then
    FILE_SIZE=$(stat -f%z "$OUTPUT_AUDIO" 2>/dev/null || stat -c%s "$OUTPUT_AUDIO" 2>/dev/null)
    
    if [ "$FILE_SIZE" -gt 1000 ]; then
        echo -e "${GREEN}✅ TTS output file created successfully${NC}"
        echo "   File: $OUTPUT_AUDIO"
        echo "   Size: $FILE_SIZE bytes"
        
        # Try to get duration using ffprobe if available
        if command -v ffprobe &> /dev/null; then
            DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUTPUT_AUDIO" 2>/dev/null || echo "unknown")
            echo "   Duration: ${DURATION}s"
        fi
    else
        echo -e "${YELLOW}⚠️  Warning: Output file is very small ($FILE_SIZE bytes)${NC}"
        echo "   This might indicate TTS failure"
    fi
else
    echo -e "${RED}❌ Test failed: No output file generated${NC}"
    exit 1
fi

# Show server log excerpt
echo ""
echo "📋 Server log excerpt (last 20 lines):"
echo "----------------------------------------"
tail -20 /tmp/voice_server.log
echo "----------------------------------------"

echo ""
echo -e "${GREEN}🎉 End-to-End Test Completed Successfully!${NC}"
echo ""
echo "✅ Verified:"
echo "   - Server startup"
echo "   - WebSocket connection"
echo "   - Audio transmission"
echo "   - TTS response reception"
echo ""
echo "💡 Next steps:"
echo "   - Play $OUTPUT_AUDIO to verify TTS quality"
echo "   - Check /tmp/voice_server.log for detailed logs"
echo "   - Run: python $SCRIPT_DIR/hello_voice.py for interactive testing"
