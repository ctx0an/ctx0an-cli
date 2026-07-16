#!/bin/bash
# install.sh - Installer for Codex Agent (ctx0an) on Termux, Linux, and macOS.

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Codex Agent Installer ===${NC}"

# 1. Detect environment
IS_TERMUX=false
if [ -d "/data/data/com.termux" ] || [ -n "$TERMUX_VERSION" ]; then
    IS_TERMUX=true
fi

# 2. Check and install system dependencies
if [ "$IS_TERMUX" = true ]; then
    echo -e "${BLUE}[1/4] Detecting Termux. Installing dependencies via pkg...${NC}"
    pkg update -y || true
    pkg install -y python python-cryptography nodejs-lts git
else
    echo -e "${BLUE}[1/4] Detecting Linux/macOS. Checking command dependencies...${NC}"
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python 3 is not installed. Please install it first.${NC}"
        exit 1
    fi
fi

# 3. Install Python SDK requirements
echo -e "${BLUE}[2/4] Installing Python packages...${NC}"
if [ "$IS_TERMUX" = true ]; then
    pip install google-genai rich --extra-index-url https://termux-user-repository.github.io/pypi/
else
    python3 -m pip install --upgrade google-genai rich || pip3 install google-genai rich
fi

# 4. Copy executable script
echo -e "${BLUE}[3/4] Installing codex script to PATH...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_FILE="$SCRIPT_DIR/ctx0an.py"

if [ ! -f "$SOURCE_FILE" ]; then
    echo -e "${RED}Error: ctx0an.py not found in $SCRIPT_DIR.${NC}"
    exit 1
fi

if [ "$IS_TERMUX" = true ]; then
    DEST_DIR="$PREFIX/bin"
    DEST_FILE="$DEST_DIR/codex"
    cp "$SOURCE_FILE" "$DEST_FILE"
    chmod +x "$DEST_FILE"
    echo -e "${GREEN}Installed to $DEST_FILE${NC}"
else
    # Try installing to local bin or fallback to global bin (needs sudo)
    LOCAL_BIN="$HOME/.local/bin"
    if [ -d "$LOCAL_BIN" ] && [[ ":$PATH:" == *":$LOCAL_BIN:"* ]]; then
        DEST_FILE="$LOCAL_BIN/codex"
        cp "$SOURCE_FILE" "$DEST_FILE"
        chmod +x "$DEST_FILE"
        echo -e "${GREEN}Installed to $DEST_FILE${NC}"
    else
        DEST_FILE="/usr/local/bin/codex"
        echo -e "${YELLOW}Installing to $DEST_FILE (may require sudo privileges)...${NC}"
        sudo cp "$SOURCE_FILE" "$DEST_FILE"
        sudo chmod +x "$DEST_FILE"
        echo -e "${GREEN}Installed to $DEST_FILE${NC}"
    fi
fi

# 5. Initialize config directory
echo -e "${BLUE}[4/4] Setting up Codex directory structure...${NC}"
mkdir -p "$HOME/.codex/sessions"

# Create a default template config if missing
CONFIG_FILE="$HOME/.codex/mcp_config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    cat <<EOT > "$CONFIG_FILE"
{
  "mcpServers": {
    "example-sqlite": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-sqlite",
        "--db-path",
        "$HOME/.codex/sqlite.db"
      ]
    }
  }
}
EOT
    echo -e "${GREEN}Created default configuration template in $CONFIG_FILE${NC}"
fi

echo -e "\n${GREEN}=== Codex Installed Successfully! ===${NC}"
echo -e "${YELLOW}To get started, follow these steps:${NC}"
echo -e "1. Get a Gemini API key from: ${BLUE}https://aistudio.google.com/apikey${NC}"
echo -e "2. Set the key in your terminal session:"
echo -e "   ${GREEN}export GEMINI_API_KEY='your-api-key-here'${NC}"
echo -e "   (Add this line to your ~/.bashrc or ~/.zshrc to persist it)"
echo -e "3. Start the interactive console dashboard:"
echo -e "   ${GREEN}codex${NC}"
echo -e "   Or start the web dashboard:"
echo -e "   ${GREEN}codex --gui${NC}\n"
