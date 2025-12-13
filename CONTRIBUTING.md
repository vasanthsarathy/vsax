# Contributing to VSAX

Thank you for your interest in contributing to VSAX! This document provides guidelines and instructions for contributing.

## Development Setup

### Using uv (Recommended)

1. Install [uv](https://github.com/astral-sh/uv):
   ```bash
   # Unix/macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Fork the repository

3. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/vsax.git
   cd vsax
   ```

4. Create a virtual environment and install in development mode:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e ".[dev,docs]"
   ```

### Using pip

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/vsax.git
   cd vsax
   ```

3. Create a virtual environment and install in development mode:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e ".[dev,docs]"
   ```

## Development Workflow

### Quick Start (Recommended)

**Before every commit**, run the pre-commit check script to catch issues locally:

```bash
# Windows
.\check.ps1

# Unix/macOS
./check.sh
```

This runs the same checks as CI/CD:
- ✅ Ruff linting
- ✅ Mypy type checking
- ✅ Pytest with coverage

If all checks pass, you're safe to commit and push!

### Detailed Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the coding standards below

3. **Run pre-commit checks** (catches CI/CD issues locally):
   ```bash
   # Windows
   .\check.ps1

   # Unix/macOS
   ./check.sh
   ```

4. If checks pass, commit your changes:
   ```bash
   git add -A
   git commit -m "feat: add new feature"
   ```

5. Push to your fork and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

### Manual Checks (if not using check script)

If you prefer to run checks individually:

```bash
# Linting
ruff check vsax tests
ruff format vsax tests

# Type checking
mypy vsax --no-site-packages

# Tests with coverage
pytest --cov=vsax --cov-report=term-missing
```

### Optional: Automatic Pre-Commit Hook

To automatically run checks before every commit:

```bash
# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
./check.sh
EOF

# Make it executable
chmod +x .git/hooks/pre-commit
```

Now checks run automatically on `git commit`!

## Coding Standards

- **Python Version**: Support Python 3.9+
- **Type Hints**: All functions must have type annotations
- **Docstrings**: Use Google-style docstrings for all public APIs
- **Line Length**: Maximum 100 characters
- **Testing**: Maintain ≥80% code coverage
- **Linting**: Code must pass `ruff` checks
- **Type Checking**: Code must pass `mypy` checks

## Testing

- Write tests for all new features
- Ensure tests pass on all supported Python versions (3.9, 3.10, 3.11)
- Run the full test suite:
  ```bash
  pytest --cov=vsax --cov-report=term-missing
  ```

## Documentation

- Update documentation for any API changes
- Add docstrings to all public functions and classes
- Update the changelog (CHANGELOG.md)
- Build docs locally to verify:
  ```bash
  mkdocs serve
  ```

## Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions or modifications
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

## Pull Request Process

1. Update README.md and documentation as needed
2. Update CHANGELOG.md with your changes
3. Ensure all tests pass and coverage is maintained
4. Request review from maintainers
5. Address any feedback
6. Once approved, your PR will be merged

## Questions?

Feel free to open an issue for any questions or concerns.
