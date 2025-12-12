#!/bin/bash
# Quick setup script for VSAX development

set -e

echo "🚀 Setting up VSAX development environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "🐍 Creating virtual environment..."
uv venv

echo "📚 Installing VSAX in development mode..."
source .venv/bin/activate
uv pip install -e ".[dev,docs]"

echo "✅ Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Then you can:"
echo "  - Run tests: pytest"
echo "  - Check types: mypy vsax"
echo "  - Lint code: ruff check vsax tests"
echo "  - Serve docs: mkdocs serve"
