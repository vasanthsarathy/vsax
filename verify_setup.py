#!/usr/bin/env python3
"""Verification script for VSAX Iteration 1 setup."""

import sys
from pathlib import Path


def check_file_exists(path: Path, description: str) -> bool:
    """Check if a file exists."""
    if path.exists():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ MISSING: {description}: {path}")
        return False


def check_directory_exists(path: Path, description: str) -> bool:
    """Check if a directory exists."""
    if path.is_dir():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ MISSING: {description}: {path}")
        return False


def main():
    """Run verification checks."""
    print("=" * 70)
    print("VSAX Iteration 1 Setup Verification")
    print("=" * 70)
    print()

    root = Path(__file__).parent
    all_checks_passed = True

    # Check core files
    print("📦 Core Configuration Files:")
    all_checks_passed &= check_file_exists(root / "pyproject.toml", "Package configuration")
    all_checks_passed &= check_file_exists(root / ".gitignore", "Git ignore file")
    all_checks_passed &= check_file_exists(root / "LICENSE", "License file")
    all_checks_passed &= check_file_exists(root / "README.md", "README")
    print()

    # Check documentation
    print("📚 Documentation:")
    all_checks_passed &= check_file_exists(root / "CONTRIBUTING.md", "Contributing guide")
    all_checks_passed &= check_file_exists(root / "CHANGELOG.md", "Changelog")
    all_checks_passed &= check_file_exists(root / "mkdocs.yml", "MkDocs config")
    all_checks_passed &= check_file_exists(root / "docs" / "index.md", "Docs homepage")
    all_checks_passed &= check_file_exists(root / "docs" / "getting-started.md", "Getting started")
    all_checks_passed &= check_file_exists(root / "docs" / "api" / "index.md", "API reference")
    print()

    # Check package structure
    print("🐍 Python Package Structure:")
    all_checks_passed &= check_file_exists(root / "vsax" / "__init__.py", "Main package init")
    all_checks_passed &= check_file_exists(root / "vsax" / "py.typed", "Type marker")
    all_checks_passed &= check_file_exists(root / "vsax" / "core" / "__init__.py", "Core module init")
    all_checks_passed &= check_file_exists(root / "vsax" / "core" / "base.py", "Abstract base classes")
    all_checks_passed &= check_file_exists(root / "vsax" / "core" / "model.py", "VSAModel")
    print()

    print("📁 Module Placeholders:")
    modules = ["representations", "ops", "sampling", "encoders", "similarity", "io", "utils"]
    for module in modules:
        all_checks_passed &= check_directory_exists(root / "vsax" / module, f"{module} module")
    print()

    # Check tests
    print("🧪 Test Infrastructure:")
    all_checks_passed &= check_file_exists(root / "tests" / "__init__.py", "Tests package")
    all_checks_passed &= check_file_exists(root / "tests" / "conftest.py", "Pytest config")
    all_checks_passed &= check_file_exists(root / "tests" / "test_infrastructure.py", "Infrastructure tests")
    all_checks_passed &= check_directory_exists(root / "tests" / "core", "Core tests directory")
    print()

    # Check CI/CD
    print("⚙️ CI/CD:")
    all_checks_passed &= check_file_exists(root / ".github" / "workflows" / "ci.yml", "CI workflow")
    all_checks_passed &= check_file_exists(root / ".github" / "workflows" / "publish.yml", "Publish workflow")
    print()

    # Check setup scripts
    print("🚀 Setup Scripts:")
    all_checks_passed &= check_file_exists(root / "setup.sh", "Unix setup script")
    all_checks_passed &= check_file_exists(root / "setup.ps1", "Windows setup script")
    print()

    # Try to import the package
    print("🔍 Package Import Test:")
    try:
        sys.path.insert(0, str(root))
        import vsax
        from vsax.core import AbstractHypervector, AbstractOpSet, VSAModel

        print(f"✓ vsax package imported successfully")
        print(f"✓ vsax.__version__ = {vsax.__version__}")
        print(f"✓ AbstractHypervector imported")
        print(f"✓ AbstractOpSet imported")
        print(f"✓ VSAModel imported")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        all_checks_passed = False
    print()

    # Final summary
    print("=" * 70)
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED - Iteration 1 setup is complete!")
    else:
        print("❌ SOME CHECKS FAILED - Please review the issues above")
    print("=" * 70)
    print()

    # Next steps
    if all_checks_passed:
        print("Next Steps:")
        print("1. Install dependencies:")
        print("   uv venv && source .venv/bin/activate  # Unix/macOS")
        print("   uv venv && .venv\\Scripts\\activate    # Windows")
        print("   uv pip install -e \".[dev,docs]\"")
        print()
        print("2. Run tests:")
        print("   pytest")
        print()
        print("3. Check code quality:")
        print("   mypy vsax")
        print("   ruff check vsax tests")
        print()
        print("4. Build documentation:")
        print("   mkdocs serve")
        print()

    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
