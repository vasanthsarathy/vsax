# Pre-commit checks - run the same checks as CI/CD locally
# Usage: .\check.ps1

$ErrorActionPreference = "Stop"

Write-Host "Running pre-commit checks..." -ForegroundColor Cyan
Write-Host ""

Write-Host "1. Running ruff linter..." -ForegroundColor Yellow
uv run ruff check vsax tests
Write-Host "✓ Ruff passed" -ForegroundColor Green
Write-Host ""

Write-Host "2. Running mypy type checker..." -ForegroundColor Yellow
uv run mypy vsax --no-site-packages
Write-Host "✓ Mypy passed" -ForegroundColor Green
Write-Host ""

Write-Host "3. Running pytest..." -ForegroundColor Yellow
uv run pytest --cov=vsax --cov-report=term-missing -q
Write-Host "✓ Tests passed" -ForegroundColor Green
Write-Host ""

Write-Host "================================" -ForegroundColor Cyan
Write-Host "✓ All checks passed!" -ForegroundColor Green
Write-Host "Safe to commit and push." -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
