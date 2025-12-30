# VSAX Release Scripts

## Version Bumping and Release Workflow

### Quick Start

To release a new version of VSAX:

```bash
# 1. Bump the version
./scripts/bump_version.sh 1.3.0

# 2. Review changes
git diff

# 3. Run tests locally (optional but recommended)
uv run pytest
uv run ruff check vsax tests
uv run mypy vsax

# 4. Commit and tag
git add -A
git commit -m "Bump version to 1.3.0"
git tag v1.3.0

# 5. Push (triggers automated workflow)
git push origin main --tags
```

That's it! GitHub Actions will automatically:
- ✅ Run all tests, linting, and type checking
- ✅ Build the distribution packages
- ✅ Publish to PyPI (via trusted publisher)
- ✅ Create a GitHub Release with changelog

---

## Scripts

### `bump_version.sh`

Automatically updates version numbers across all project files.

**Usage:**
```bash
./scripts/bump_version.sh <version>
```

**Example:**
```bash
./scripts/bump_version.sh 1.3.0
```

**Files Updated:**
- `pyproject.toml` - Package version
- `vsax/__init__.py` - `__version__` attribute
- `tests/test_infrastructure.py` - Version test assertion
- `README.md` - Citation version
- `docs/index.md` - Citation version

---

## GitHub Actions Workflow

### Publish to PyPI (`.github/workflows/publish.yml`)

**Trigger:** Push a version tag (e.g., `v1.3.0`)

**Jobs:**

1. **Test** - Runs full test suite with coverage, linting, and type checking
2. **Build** - Builds source distribution and wheel
3. **Publish to PyPI** - Publishes to PyPI using trusted publisher (OIDC)
4. **Create Release** - Auto-creates GitHub Release with CHANGELOG

**Requirements:**

- **PyPI Trusted Publisher** must be configured:
  - Go to: https://pypi.org/manage/account/publishing/
  - Add publisher for `vasanthsarathy/vsax`
  - Workflow: `publish.yml`
  - Environment: (leave blank)

- **CHANGELOG.md** must have a section for the version:
  ```markdown
  ## [1.3.0] - 2025-01-15

  ### Added
  - New feature X

  ### Fixed
  - Bug Y
  ```

---

## Trusted Publisher Setup on PyPI

### Step 1: Initial Setup (First Time Only)

If this is your first release, you'll need to create the PyPI project manually:

1. Build the package: `python -m build`
2. Upload manually: `twine upload dist/*` (requires PyPI API token)

### Step 2: Configure Trusted Publisher

Once the project exists on PyPI:

1. Go to https://pypi.org/manage/project/vsax/settings/publishing/
2. Click "Add a new publisher"
3. Fill in:
   - **Owner:** `vasanthsarathy`
   - **Repository:** `vsax`
   - **Workflow:** `publish.yml`
   - **Environment:** (leave blank)
4. Save

### Step 3: Verify

After setup, all future releases will use the trusted publisher automatically!

---

## Version Numbering

VSAX follows [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH** (e.g., `1.3.0`)
  - **MAJOR**: Breaking changes
  - **MINOR**: New features (backward compatible)
  - **PATCH**: Bug fixes (backward compatible)

### Examples:

- `1.2.1` → `1.2.2`: Bug fix release
- `1.2.1` → `1.3.0`: New feature release
- `1.2.1` → `2.0.0`: Breaking change release

---

## Troubleshooting

### "Trusted publisher failed"

**Issue:** PyPI returns 403 Forbidden

**Solution:**
- Verify trusted publisher is configured correctly on PyPI
- Check that repository owner, name, and workflow match exactly
- Ensure you're pushing a tag (not just committing)

### "Tests failed in CI"

**Issue:** GitHub Actions fails during test job

**Solution:**
- Run tests locally: `uv run pytest --cov=vsax`
- Fix failing tests before pushing tag
- If tests pass locally but fail in CI, check for environment differences

### "Changelog not found"

**Issue:** GitHub Release has no body

**Solution:**
- Ensure `CHANGELOG.md` has a section for your version:
  ```markdown
  ## [1.3.0] - YYYY-MM-DD
  ```
- Or update the `[Unreleased]` section before tagging

---

## CI/CD Pipeline Overview

```
Tag Push (v1.3.0)
    ↓
Run Tests (pytest, ruff, mypy)
    ↓
Build Package (sdist + wheel)
    ↓
Publish to PyPI (trusted publisher)
    ↓
Create GitHub Release (with CHANGELOG)
    ↓
✅ Version 1.3.0 live on PyPI!
```

---

## Manual Release (Fallback)

If automated release fails, you can release manually:

```bash
# Build
python -m build

# Check
twine check dist/*

# Upload (requires API token)
twine upload dist/*

# Create GitHub Release manually
gh release create v1.3.0 --notes "Release notes here"
```

But the automated workflow is strongly preferred! 🚀
