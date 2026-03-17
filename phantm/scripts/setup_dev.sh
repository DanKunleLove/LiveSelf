#!/bin/bash
# ============================================
# LiveSelf Dev Environment Setup
# Run this once on a fresh machine or Colab
# ============================================

set -e  # Stop on any error

echo "===== LiveSelf Dev Setup ====="
echo ""

# --- Check Python version ---
PYTHON_VERSION=$(python3 --version 2>/dev/null || python --version 2>/dev/null)
echo "Python: $PYTHON_VERSION"
echo "WARNING: You need Python 3.10 or 3.11. If you have 3.13, install 3.11 separately."
echo ""

# --- Check Node version ---
NODE_VERSION=$(node --version 2>/dev/null || echo "NOT INSTALLED")
echo "Node.js: $NODE_VERSION"
echo ""

# --- Check Git ---
GIT_VERSION=$(git --version 2>/dev/null || echo "NOT INSTALLED")
echo "Git: $GIT_VERSION"
echo ""

# --- Backend setup ---
echo "===== Setting up Backend ====="
cd "$(dirname "$0")/../backend"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3.11 -m venv .venv 2>/dev/null || python3 -m venv .venv 2>/dev/null || python -m venv .venv
fi

# Activate and install
echo "Installing backend dependencies..."
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null
pip install -r requirements.txt
echo "Backend setup complete."
echo ""

# --- Engine setup (skip on local, run on Colab/RunPod) ---
echo "===== Engine Setup ====="
echo "NOTE: Engine dependencies are heavy (GPU required)."
echo "For local dev, skip this. For Colab/RunPod, run:"
echo "  cd engine && pip install -r requirements.txt"
echo ""

# --- Frontend setup ---
echo "===== Setting up Frontend ====="
cd "$(dirname "$0")/../frontend"
if [ -f "package.json" ]; then
    echo "Installing frontend dependencies..."
    npm install
    echo "Frontend setup complete."
else
    echo "Frontend not scaffolded yet (Phase 2). Skipping."
fi
echo ""

# --- Copy env file ---
cd "$(dirname "$0")/.."
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "IMPORTANT: Edit .env and fill in your API keys."
else
    echo ".env already exists."
fi

echo ""
echo "===== Setup Complete ====="
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys"
echo "2. Start backend: cd backend && source .venv/bin/activate && uvicorn app.main:app --reload"
echo "3. For AI pipeline: use Google Colab or RunPod (GPU required)"
