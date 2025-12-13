# Script to build and publish VSAX to PyPI
# Usage: .\publish.ps1

$ErrorActionPreference = "Stop"

Write-Host "Cleaning previous builds..."
Remove-Item -Path dist, build, vsax.egg-info -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "`nBuilding package..."
uv build

Write-Host "`nPublishing to PyPI..."

# Check if environment variable is set
if (-not $env:UV_PUBLISH_TOKEN) {
    Write-Host "ERROR: UV_PUBLISH_TOKEN not set!" -ForegroundColor Red
    Write-Host 'Please set it with: $env:UV_PUBLISH_TOKEN="pypi-YOUR_TOKEN_HERE"'
    exit 1
}

uv publish

Write-Host "`nSuccessfully published to PyPI!" -ForegroundColor Green
Write-Host "Users can now install with: pip install vsax"
