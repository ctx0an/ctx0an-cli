#!/bin/bash
# install.sh - Installer for Ctx0an Agent (ctx0an) on Termux, Linux, and macOS.

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Ctx0an Agent Installer ===${NC}"

# 1. Detect environment
IS_TERMUX=false
if [ -d "/data/data/com.termux" ] || [ -n "$TERMUX_VERSION" ]; then
    IS_TERMUX=true
fi

# 2. Check and install system dependencies
if [ "$IS_TERMUX" = true ]; then
    echo -e "${BLUE}[1/4] Detecting Termux. Checking package dependencies...${NC}"
    packages=(python python-cryptography nodejs-lts git rust)
    to_install=()
    for pkg in "${packages[@]}"; do
        if ! dpkg -s "$pkg" &>/dev/null; then
            to_install+=("$pkg")
        fi
    done
    if [ ${#to_install[@]} -ne 0 ]; then
        echo -e "${YELLOW}Installing missing packages: ${to_install[*]}${NC}"
        pkg update -y || true
        pkg install -y "${to_install[@]}"
    else
        echo -e "${GREEN}All system packages are already installed.${NC}"
    fi
    
    # Fix AttributeError: module 'os' has no attribute 'link' on Termux Python 3.14+
    SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
    if [ -n "$SITE_PACKAGES" ]; then
        SITE_CUSTOMIZE="$SITE_PACKAGES/sitecustomize.py"
        if [ ! -f "$SITE_CUSTOMIZE" ] || ! grep -q "os.link" "$SITE_CUSTOMIZE"; then
            echo -e "${YELLOW}Applying os.link hotfix for Termux Python compatibility...${NC}"
            mkdir -p "$SITE_PACKAGES"
            echo -e "\nimport os\nif not hasattr(os, 'link'):\n    os.link = lambda src, dst, *args, **kwargs: None" >> "$SITE_CUSTOMIZE"
        fi
    fi
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
    if python3 -c "from google import genai; import rich" &>/dev/null; then
        echo -e "${GREEN}Python packages google-genai and rich are already installed.${NC}"
    else
        pip install google-genai rich --extra-index-url https://termux-user-repository.github.io/pypi/
    fi
else
    if python3 -c "from google import genai; import rich" &>/dev/null; then
        echo -e "${GREEN}Python packages google-genai and rich are already installed.${NC}"
    else
        python3 -m pip install --upgrade google-genai rich || pip3 install google-genai rich
    fi
fi

# 4. Copy executable script
echo -e "${BLUE}[3/4] Installing ctx0an script to PATH...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_FILE="$SCRIPT_DIR/ctx0an.py"

if [ ! -f "$SOURCE_FILE" ]; then
    echo -e "${RED}Error: ctx0an.py not found in $SCRIPT_DIR.${NC}"
    exit 1
fi

if [ "$IS_TERMUX" = true ]; then
    DEST_DIR="$PREFIX/bin"
    DEST_FILE="$DEST_DIR/ctx0an"
    if [ -f "$DEST_FILE" ] && cmp -s "$SOURCE_FILE" "$DEST_FILE"; then
        echo -e "${GREEN}Script is already up-to-date at $DEST_FILE${NC}"
    else
        cp "$SOURCE_FILE" "$DEST_FILE"
        chmod +x "$DEST_FILE"
        echo -e "${GREEN}Installed/Updated script at $DEST_FILE${NC}"
    fi
else
    # Try installing to local bin or fallback to global bin (needs sudo)
    LOCAL_BIN="$HOME/.local/bin"
    if [ -d "$LOCAL_BIN" ] && [[ ":$PATH:" == *":$LOCAL_BIN:"* ]]; then
        DEST_FILE="$LOCAL_BIN/ctx0an"
        if [ -f "$DEST_FILE" ] && cmp -s "$SOURCE_FILE" "$DEST_FILE"; then
            echo -e "${GREEN}Script is already up-to-date at $DEST_FILE${NC}"
        else
            cp "$SOURCE_FILE" "$DEST_FILE"
            chmod +x "$DEST_FILE"
            echo -e "${GREEN}Installed/Updated script at $DEST_FILE${NC}"
        fi
    else
        DEST_FILE="/usr/local/bin/ctx0an"
        if [ -f "$DEST_FILE" ] && cmp -s "$SOURCE_FILE" "$DEST_FILE"; then
            echo -e "${GREEN}Script is already up-to-date at $DEST_FILE${NC}"
        else
            echo -e "${YELLOW}Installing/Updating $DEST_FILE (may require sudo privileges)...${NC}"
            sudo cp "$SOURCE_FILE" "$DEST_FILE"
            sudo chmod +x "$DEST_FILE"
            echo -e "${GREEN}Installed/Updated script at $DEST_FILE${NC}"
        fi
    fi
fi

# 5. Initialize config directory
echo -e "${BLUE}[4/4] Setting up Ctx0an directory structure...${NC}"
mkdir -p "$HOME/.ctx0an/sessions"

# Create a default template config if missing
CONFIG_FILE="$HOME/.ctx0an/mcp_config.json"
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
        "$HOME/.ctx0an/sqlite.db"
      ]
    }
  }
}
EOT
    echo -e "${GREEN}Created default configuration template in $CONFIG_FILE${NC}"
fi

# 6. Optional API Key configuration
if [ -z "$GEMINI_API_KEY" ]; then
    echo -e -n "\n${YELLOW}Would you like to configure your GEMINI_API_KEY now? [y/N]: ${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo -e -n "Enter your Gemini API Key: "
        read -r api_key
        if [ -n "$api_key" ]; then
            # Determine shell config file
            SHELL_CONFIG=""
            if [ -f "$HOME/.bashrc" ]; then
                SHELL_CONFIG="$HOME/.bashrc"
            elif [ -f "$HOME/.zshrc" ]; then
                SHELL_CONFIG="$HOME/.zshrc"
            else
                SHELL_CONFIG="$HOME/.bashrc"
                touch "$SHELL_CONFIG"
            fi
            
            # Append if not already present
            if ! grep -q "export GEMINI_API_KEY=" "$SHELL_CONFIG"; then
                # Ensure it ends with newline before appending
                [ -n "$(tail -c1 "$SHELL_CONFIG" 2>/dev/null)" ] && echo "" >> "$SHELL_CONFIG"
                echo "export GEMINI_API_KEY='$api_key'" >> "$SHELL_CONFIG"
                echo -e "${GREEN}Saved API key to $SHELL_CONFIG${NC}"
                export GEMINI_API_KEY="$api_key"
            else
                echo -e "${YELLOW}GEMINI_API_KEY is already defined in $SHELL_CONFIG. Skipping auto-write.${NC}"
            fi
        fi
    fi
fi

echo -e "\n${GREEN}=== Ctx0an Installed Successfully! ===${NC}"
echo -e "${YELLOW}To get started, follow these steps:${NC}"
echo -e "1. Get a Gemini API key from: ${BLUE}https://aistudio.google.com/apikey${NC}"
echo -e "2. Set the key in your terminal session:"
echo -e "   ${GREEN}export GEMINI_API_KEY='your-api-key-here'${NC}"
echo -e "   (Add this line to your ~/.bashrc or ~/.zshrc to persist it)"
echo -e "3. Start the interactive console dashboard:"
echo -e "   ${GREEN}ctx0an${NC}"
echo -e "   Or start the web dashboard:"
echo -e "   ${GREEN}ctx0an --gui${NC}\n"
