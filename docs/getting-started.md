# Getting Started

## Installation

VSAX requires Python 3.9 or later.

### From PyPI (Coming Soon)

```bash
pip install vsax
```

### From Source

#### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver created by Astral (the makers of ruff). It's significantly faster than pip and handles virtual environments seamlessly.

**Install uv:**

```bash
# Unix/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Install VSAX:**

```bash
git clone https://github.com/yourusername/vsax.git
cd vsax

# Create virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

#### Using pip

```bash
git clone https://github.com/yourusername/vsax.git
cd vsax

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -e .
```

### Development Installation

To install with development dependencies:

**Using uv:**
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev,docs]"
```

**Using pip:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev,docs]"
```

## Verifying Installation

```bash
# Check that vsax is installed
python -c "import vsax; print(vsax.__version__)"

# Run tests
pytest
```

## Basic Usage

*Coming in Iteration 2+*

The library will support three VSA models:

1. **FHRR** - Fourier Holographic Reduced Representation
2. **MAP** - Multiply-Add-Permute
3. **Binary VSA** - Binary hypervectors with XOR binding

## Next Steps

- Explore the [API Reference](api/index.md)
- Check out example notebooks (coming in Iteration 4)
- Read the [design specification](https://github.com/yourusername/vsax/blob/main/docs/design-spec.md)
