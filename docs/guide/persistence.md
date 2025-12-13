# Persistence: Saving and Loading Basis Vectors

VSAX provides simple JSON-based persistence for saving and loading basis vectors. This enables you to:

- **Preserve semantic spaces** across sessions
- **Share vocabularies** between projects
- **Version control** your basis vectors
- **Reproduce experiments** with exact same vectors

## Quick Start

```python
from vsax import create_fhrr_model, VSAMemory, save_basis, load_basis

# Create and populate memory
model = create_fhrr_model(dim=512)
memory = VSAMemory(model)
memory.add_many(["dog", "cat", "animal", "pet"])

# Save to JSON
save_basis(memory, "animals.json")

# Later: Load into new memory
new_memory = VSAMemory(model)
load_basis(new_memory, "animals.json")

# Vectors are preserved exactly
assert "dog" in new_memory
```

## Saving Basis Vectors

### Basic Usage

```python
from pathlib import Path
from vsax import save_basis

# Save with Path object
save_basis(memory, Path("my_basis.json"))

# Or with string path
save_basis(memory, "my_basis.json")
```

### What Gets Saved?

The JSON file contains:

1. **Metadata**: Dimension, representation type, vector count
2. **Vectors**: All named vectors in the memory

Example JSON structure for FHRR (complex) vectors:

```json
{
  "metadata": {
    "dim": 512,
    "rep_type": "complex",
    "num_vectors": 3
  },
  "vectors": {
    "dog": {
      "real": [0.12, -0.34, ...],
      "imag": [0.56, 0.78, ...]
    },
    "cat": {
      "real": [-0.45, 0.23, ...],
      "imag": [0.11, -0.67, ...]
    }
  }
}
```

### All Three Models Supported

**FHRR (Complex Vectors)**:
- Stored as separate real and imaginary parts
- JSON keys: `"real"` and `"imag"`

**MAP (Real Vectors)**:
- Stored as simple float arrays
- Direct list representation

**Binary (Bipolar Vectors)**:
- Stored as integer arrays (-1, +1 or 0, 1)
- Compact representation

```python
# Each model saves differently
fhrr_model = create_fhrr_model(dim=512)
map_model = create_map_model(dim=512)
binary_model = create_binary_model(dim=10000, bipolar=True)

memory_fhrr = VSAMemory(fhrr_model)
memory_map = VSAMemory(map_model)
memory_binary = VSAMemory(binary_model)

# All use same API
save_basis(memory_fhrr, "fhrr.json")
save_basis(memory_map, "map.json")
save_basis(memory_binary, "binary.json")
```

## Loading Basis Vectors

### Basic Usage

```python
from vsax import load_basis

# Create empty memory with correct model
model = create_fhrr_model(dim=512)
memory = VSAMemory(model)

# Load from file
load_basis(memory, "my_basis.json")

# Memory is now populated
print(f"Loaded {len(memory._vectors)} vectors")
```

### Requirements

1. **Empty Memory**: Memory must be empty before loading
2. **Matching Dimension**: File dimension must match memory's model dimension
3. **Matching Type**: File rep_type must match memory's model type

### Error Handling

```python
# Dimension mismatch
model_128 = create_fhrr_model(dim=128)
model_256 = create_fhrr_model(dim=256)

memory_128 = VSAMemory(model_128)
memory_128.add("test")
save_basis(memory_128, "test.json")

memory_256 = VSAMemory(model_256)
try:
    load_basis(memory_256, "test.json")  # ❌ Dimension mismatch!
except ValueError as e:
    print(f"Error: {e}")

# Representation type mismatch
fhrr_memory = VSAMemory(create_fhrr_model(dim=128))
fhrr_memory.add("test")
save_basis(fhrr_memory, "test.json")

map_memory = VSAMemory(create_map_model(dim=128))
try:
    load_basis(map_memory, "test.json")  # ❌ Type mismatch!
except ValueError as e:
    print(f"Error: {e}")

# Non-empty memory
memory = VSAMemory(create_fhrr_model(dim=128))
memory.add("existing")
try:
    load_basis(memory, "test.json")  # ❌ Memory not empty!
except ValueError as e:
    print(f"Error: {e}")
```

## Common Use Cases

### 1. Persistent Semantic Spaces

Build a knowledge base once, reuse it across sessions:

```python
# Session 1: Build semantic space
model = create_fhrr_model(dim=1024)
memory = VSAMemory(model)

# Add domain vocabulary
memory.add_many([
    "entity1", "entity2", "relation1", "relation2",
    "attribute1", "attribute2", ...
])

# Create complex structures
entity_with_attr = model.opset.bind(
    memory["entity1"].vec,
    memory["attribute1"].vec
)

# Save for later
save_basis(memory, "knowledge_base.json")

# Session 2: Load and use
memory_new = VSAMemory(model)
load_basis(memory_new, "knowledge_base.json")

# All symbols available immediately
entity = memory_new["entity1"]
```

### 2. Sharing Vocabularies

Share exact basis vectors between projects or team members:

```python
# Project A: Create shared vocabulary
model = create_map_model(dim=512)
memory = VSAMemory(model)
memory.add_many(["term1", "term2", "term3", ...])
save_basis(memory, "shared_vocab.json")

# Project B: Use same vocabulary
model_b = create_map_model(dim=512)  # Same dim!
memory_b = VSAMemory(model_b)
load_basis(memory_b, "shared_vocab.json")

# Projects now use identical basis
```

### 3. Reproducible Research

Version control your basis vectors for reproducible experiments:

```bash
# Save basis with experiment
git add experiment_basis.json
git commit -m "Add basis for experiment 1"

# Others can reproduce exact results
git clone repo
python experiment.py  # Loads experiment_basis.json
```

### 4. Incremental Development

Save progress and resume later:

```python
# Day 1: Initial setup
memory = VSAMemory(create_fhrr_model(dim=512))
memory.add_many(["concept1", "concept2", ...])
save_basis(memory, "progress.json")

# Day 2: Resume and extend
memory = VSAMemory(create_fhrr_model(dim=512))
load_basis(memory, "progress.json")
memory.add_many(["concept3", "concept4", ...])  # Add more
save_basis(memory, "progress.json")  # Overwrite
```

## Best Practices

### File Organization

```
project/
├── basis/
│   ├── entities.json      # Entity vectors
│   ├── relations.json     # Relation vectors
│   └── attributes.json    # Attribute vectors
├── experiments/
│   ├── exp1_basis.json
│   └── exp2_basis.json
└── shared/
    └── common_vocab.json
```

### Naming Conventions

```python
# Descriptive filenames
save_basis(memory, "medical_terms_512d_fhrr.json")
save_basis(memory, "colors_256d_map.json")
save_basis(memory, "code_symbols_10k_binary.json")
```

### Version Control

```python
# Include dimension and date in filename
from datetime import datetime

date_str = datetime.now().strftime("%Y%m%d")
filename = f"basis_{model.dim}d_{date_str}.json"
save_basis(memory, filename)
```

### Testing

```python
import jax.numpy as jnp

# Always verify round-trip
save_basis(memory_original, "test.json")
load_basis(memory_loaded, "test.json")

for name in memory_original._vectors:
    vec1 = memory_original[name].vec
    vec2 = memory_loaded[name].vec
    assert jnp.allclose(vec1, vec2, atol=1e-6)
```

## Performance Considerations

### File Size

- **FHRR**: 2× vector dimension (real + imag parts)
- **MAP**: 1× vector dimension (real values)
- **Binary**: 1× vector dimension (integers)

Approximate sizes for 100 vectors:

| Model | Dim | File Size |
|-------|-----|-----------|
| FHRR | 512 | ~500 KB |
| MAP | 512 | ~250 KB |
| Binary | 10,000 | ~2 MB |

### Load Time

Loading is fast (typically < 100ms for typical sizes):

```python
import time

start = time.time()
load_basis(memory, "large_basis.json")
elapsed = time.time() - start
print(f"Loaded in {elapsed*1000:.1f}ms")
```

### Large Vocabularies

For very large vocabularies (1000s of vectors):

```python
# Consider splitting into multiple files
save_basis(entities_memory, "entities.json")
save_basis(relations_memory, "relations.json")
save_basis(attributes_memory, "attributes.json")

# Load only what you need
memory = VSAMemory(model)
load_basis(memory, "entities.json")  # Load just entities
```

## Troubleshooting

**File not found?**
```python
from pathlib import Path

path = Path("my_basis.json")
if not path.exists():
    print(f"File not found: {path.absolute()}")
```

**Wrong dimension?**
```python
# Check file metadata first
import json
with open("basis.json") as f:
    data = json.load(f)
    print(f"File dimension: {data['metadata']['dim']}")
    print(f"File type: {data['metadata']['rep_type']}")
```

**Corrupted JSON?**
```python
try:
    load_basis(memory, "basis.json")
except json.JSONDecodeError:
    print("JSON file is corrupted")
```

## See Also

- [API Reference: I/O](../api/io/index.md) - Complete API documentation
- [Examples: persistence.py](../../examples/persistence.py) - Full working example
- [VSAMemory Guide](memory.md) - Memory management
