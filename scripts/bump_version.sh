#!/bin/bash
# Version bumping and release script for VSAX

set -e

# Check if version argument is provided
if [ -z "$1" ]; then
    echo "Usage: ./scripts/bump_version.sh <version>"
    echo "Example: ./scripts/bump_version.sh 1.3.0"
    exit 1
fi

NEW_VERSION=$1

# Validate version format (basic check)
if ! [[ $NEW_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must be in format X.Y.Z (e.g., 1.3.0)"
    exit 1
fi

echo "Bumping version to $NEW_VERSION..."
echo ""

# Update version in pyproject.toml
echo "Updating pyproject.toml..."
sed -i "s/^version = .*/version = \"$NEW_VERSION\"/" pyproject.toml

# Update version in vsax/__init__.py
echo "Updating vsax/__init__.py..."
sed -i "s/__version__ = .*/__version__ = \"$NEW_VERSION\"/" vsax/__init__.py

# Update version in tests/test_infrastructure.py
echo "Updating tests/test_infrastructure.py..."
sed -i "s/assert vsax.__version__ == .*/assert vsax.__version__ == \"$NEW_VERSION\"/" tests/test_infrastructure.py

# Update version in README.md citation
echo "Updating README.md citation..."
sed -i "s/version = {.*}/version = {$NEW_VERSION}/" README.md

# Update version in docs/index.md citation
echo "Updating docs/index.md citation..."
sed -i "s/version = {.*}/version = {$NEW_VERSION}/" docs/index.md

echo ""
echo "✓ Updated version to $NEW_VERSION in all files"
echo ""
echo "Files updated:"
echo "  - pyproject.toml"
echo "  - vsax/__init__.py"
echo "  - tests/test_infrastructure.py"
echo "  - README.md (citation)"
echo "  - docs/index.md (citation)"
echo ""
echo "Next steps:"
echo "1. Review changes: git diff"
echo "2. Run tests: uv run pytest"
echo "3. Commit: git add -A && git commit -m 'Bump version to $NEW_VERSION'"
echo "4. Tag: git tag v$NEW_VERSION"
echo "5. Push: git push origin main --tags"
echo ""
echo "GitHub Actions will automatically:"
echo "  - Run tests"
echo "  - Build package"
echo "  - Publish to PyPI (with trusted publisher)"
echo "  - Create GitHub Release with changelog"
