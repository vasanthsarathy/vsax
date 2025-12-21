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
git clone https://github.com/vasanthsarathy/vsax.git
cd vsax

# Create virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

#### Using pip

```bash
git clone https://github.com/vasanthsarathy/vsax.git
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
memory.add_many(["subject", "action", "dog", "cat", "run", "jump"])

# Encode structured data
encoder = DictEncoder(model, memory)
sentence = encoder.encode({"subject": "dog", "action": "run"})

# Similarity search
similarity = cosine_similarity(memory["dog"], memory["cat"])
print(f"Similarity: {similarity:.3f}")
```

### Advanced Features

**Resonator Networks** (v0.7.0+) - Factorize composite hypervectors:

```python
from vsax import CleanupMemory, Resonator

# Create codebooks
letters = CleanupMemory(["alpha", "beta"], memory)
numbers = CleanupMemory(["one", "two"], memory)

# Create resonator
resonator = Resonator([letters, numbers], model.opset)

# Factorize composite
composite = model.opset.bind(memory["alpha"].vec, memory["one"].vec)
factors = resonator.factorize(composite)  # ["alpha", "one"]
```

**I/O & Persistence** (v0.6.0+) - Save and load basis vectors:

```python
from vsax import save_basis, load_basis

# Save basis to JSON
save_basis(memory, "my_basis.json")

# Load basis from JSON
new_memory = VSAMemory(model)
load_basis(new_memory, "my_basis.json")
```

**Clifford Operators** (v1.1.0+) - Exact transformations for reasoning:

```python
from vsax.operators import CliffordOperator, OperatorKind
import jax

# Create spatial operator
LEFT_OF = CliffordOperator.random(
    dim=512,
    kind=OperatorKind.SPATIAL,
    name="LEFT_OF",
    key=jax.random.PRNGKey(100)
)

# Encode spatial relation: "cup LEFT_OF plate"
scene = model.opset.bundle(
    memory["cup"].vec,
    LEFT_OF.apply(memory["plate"]).vec
)

# Query with exact inversion
RIGHT_OF = LEFT_OF.inverse()
answer = RIGHT_OF.apply(model.rep_cls(scene))
# Similarity to "cup" will be > 0.7 (vs 0.3-0.6 with bundling)
```

## Next Steps

- Explore the [API Reference](api/index.md)
- Try the [Tutorials](tutorials/index.md) with real datasets
  - [Tutorial 10: Clifford Operators](tutorials/10_clifford_operators.md) - Exact transformations (NEW in v1.1.0)
- Read the [User Guide](guide/models.md) for detailed information
  - [Operators Guide](guide/operators.md) - Using Clifford operators (NEW)
- Read the [design specification](design-spec.md)
- Learn about [Resonator Networks](guide/resonator.md)
- Understand [Persistence](guide/persistence.md)
