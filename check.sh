#!/bin/bash
# Pre-commit checks - run the same checks as CI/CD locally
# Usage: ./check.sh

set -e  # Exit on any error

echo "Running pre-commit checks..."
echo ""

echo "1. Running ruff linter..."
uv run ruff check vsax tests
echo "✓ Ruff passed"
echo ""

echo "2. Running mypy type checker..."
uv run mypy vsax --no-site-packages
echo "✓ Mypy passed"
echo ""

echo "3. Running pytest..."
uv run pytest --cov=vsax --cov-report=term-missing -q
echo "✓ Tests passed"
echo ""

echo "================================"
echo "✓ All checks passed!"
echo "Safe to commit and push."
echo "================================"
