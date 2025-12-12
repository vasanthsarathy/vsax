# Quick setup script for VSAX development (Windows PowerShell)

Write-Host "🚀 Setting up VSAX development environment..." -ForegroundColor Green

# Check if uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "📦 Installing uv..." -ForegroundColor Yellow
    irm https://astral.sh/uv/install.ps1 | iex
}

Write-Host "🐍 Creating virtual environment..." -ForegroundColor Green
uv venv

Write-Host "📚 Installing VSAX in development mode..." -ForegroundColor Green
.\.venv\Scripts\Activate.ps1
uv pip install -e ".[dev,docs]"

Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the virtual environment, run:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Then you can:" -ForegroundColor Cyan
Write-Host "  - Run tests: pytest" -ForegroundColor White
Write-Host "  - Check types: mypy vsax" -ForegroundColor White
Write-Host "  - Lint code: ruff check vsax tests" -ForegroundColor White
Write-Host "  - Serve docs: mkdocs serve" -ForegroundColor White
