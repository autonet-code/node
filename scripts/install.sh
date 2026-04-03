#!/usr/bin/env bash
# Autonet installer for Linux and macOS
# Downloads and installs the autonet package into an isolated venv.
#
# Usage:
#   curl -sSL https://github.com/autonet-code/node/releases/latest/download/install.sh | bash
#   # or:
#   ./install.sh                  # install from PyPI
#   ./install.sh ./autonet.whl    # install from a local wheel
set -euo pipefail

INSTALL_DIR="${AUTONET_INSTALL_DIR:-$HOME/.autonet}"
VENV_DIR="$INSTALL_DIR/venv"
BIN_LINK="/usr/local/bin/atn"
MIN_PYTHON="3.11"

info()  { printf '\033[1;34m=> %s\033[0m\n' "$*"; }
error() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --- Find Python >= 3.11 ---
find_python() {
    for candidate in python3.13 python3.12 python3.11 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            version=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
            major="${version%%.*}"
            minor="${version#*.}"
            if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
                echo "$candidate"
                return
            fi
        fi
    done
    error "Python >= $MIN_PYTHON is required but not found. Install it first."
}

PYTHON=$(find_python)
info "Using $PYTHON ($($PYTHON --version))"

# --- Create venv ---
info "Creating virtual environment at $VENV_DIR"
mkdir -p "$INSTALL_DIR"
"$PYTHON" -m venv "$VENV_DIR"

# --- Install ---
if [ "${1:-}" != "" ] && [ -f "$1" ]; then
    info "Installing from local file: $1"
    "$VENV_DIR/bin/pip" install --upgrade "$1"
else
    info "Installing autonet from PyPI"
    "$VENV_DIR/bin/pip" install --upgrade autonet
fi

# --- Symlink ---
ATN_BIN="$VENV_DIR/bin/atn"
if [ -f "$ATN_BIN" ]; then
    if [ -w "$(dirname "$BIN_LINK")" ]; then
        ln -sf "$ATN_BIN" "$BIN_LINK"
        info "Linked: $BIN_LINK -> $ATN_BIN"
    else
        info "Adding $VENV_DIR/bin to your PATH (no write access to /usr/local/bin)"
        SHELL_RC=""
        if [ -f "$HOME/.zshrc" ]; then
            SHELL_RC="$HOME/.zshrc"
        elif [ -f "$HOME/.bashrc" ]; then
            SHELL_RC="$HOME/.bashrc"
        fi
        if [ -n "$SHELL_RC" ]; then
            if ! grep -q "$VENV_DIR/bin" "$SHELL_RC" 2>/dev/null; then
                echo "export PATH=\"$VENV_DIR/bin:\$PATH\"" >> "$SHELL_RC"
                info "Added PATH entry to $SHELL_RC — restart your shell or run: source $SHELL_RC"
            fi
        else
            info "Add this to your shell profile: export PATH=\"$VENV_DIR/bin:\$PATH\""
        fi
    fi
fi

info "Installation complete!"
info "Run 'atn' to start the agent framework."
info "Optional extras: pip install autonet[voice] autonet[node] autonet[p2p]"
