#!/bin/bash
# Script to build and publish VSAX to PyPI

set -e  # Exit on any error

echo "Cleaning previous builds..."
rm -rf dist build vsax.egg-info

echo ""
echo "Building package..."
uv build

echo ""
echo "Publishing to PyPI..."
if [ -z "$UV_PUBLISH_TOKEN" ]; then
    echo "ERROR: UV_PUBLISH_TOKEN is not set!"
    echo "Please set it with: export UV_PUBLISH_TOKEN=pypi-YOUR_TOKEN_HERE"
    exit 1
fi

uv publish

echo ""
echo "Successfully published to PyPI!"
echo "Users can now install with: pip install vsax"
