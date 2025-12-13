# Getting Started

## Installation

VSAX requires Python 3.9 or later.

### From PyPI (Recommended)

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

VSAX supports three VSA models:

1. **FHRR** - Fourier Holographic Reduced Representation (complex hypervectors)
2. **MAP** - Multiply-Add-Permute (real hypervectors)
3. **Binary VSA** - Binary hypervectors with XOR binding

### Quick Start

```python
from vsax import create_fhrr_model, VSAMemory, DictEncoder
from vsax.similarity import cosine_similarity

# Create model with factory function
model = create_fhrr_model(dim=512)

# Create memory for symbols
memory = VSAMemory(model)
memory.add_many(["dog", "cat", "run", "jump"])

# Encode structured data
encoder = DictEncoder(model, memory)
sentence = encoder.encode({"subject": "dog", "action": "run"})

# Similarity search
similarity = cosine_similarity(memory["dog"], memory["cat"])
print(f"Similarity: {similarity:.3f}")
```

## Next Steps

- Explore the [API Reference](api/index.md)
- Check out [example notebooks](../examples/)
- Read the [design specification](design-spec.md)
- See the [User Guide](guide/models.md) for detailed tutorials
