# VSAMemory: Symbol Table Management

VSAMemory provides a dictionary-style interface for creating and managing named hypervectors (basis symbols). It acts as a symbol table that automatically samples and stores hypervectors for symbolic concepts.

## Overview

VSAMemory simplifies working with VSA models by:

- **Automatic sampling**: Creates hypervectors on-demand when you add symbols
- **Dictionary-style access**: Use familiar `memory["symbol"]` syntax
- **Reproducibility**: Optional PRNG key for deterministic sampling
- **Model-agnostic**: Works with FHRR, MAP, and Binary models

## Basic Usage

### Creating Memory

```python
from vsax import create_fhrr_model, VSAMemory

# Create a model
model = create_fhrr_model(dim=512)

# Create memory (with optional key for reproducibility)
memory = VSAMemory(model, key=jax.random.PRNGKey(42))
```

### Adding Symbols

```python
# Add a single symbol
dog = memory.add("dog")

# Add multiple symbols
memory.add_many(["cat", "bird", "fish"])

# Adding duplicate returns the same hypervector
dog2 = memory.add("dog")  # Same as dog
assert jnp.array_equal(dog.vec, dog2.vec)
```

### Accessing Symbols

```python
# Dictionary-style access
dog = memory["dog"]
cat = memory["cat"]

# Check if symbol exists
if "dog" in memory:
    print("Dog is in memory")

# Get all symbol names
symbols = memory.keys()  # ['dog', 'cat', 'bird', 'fish']

# Number of symbols
count = len(memory)  # 4
```

### Using Symbols

```python
# Get the underlying vector
dog_vec = memory["dog"].vec

# Bind two concepts
dog_is_animal = model.opset.bind(memory["dog"].vec, memory["animal"].vec)

# Bundle multiple concepts
pets = model.opset.bundle(
    memory["dog"].vec,
    memory["cat"].vec,
    memory["bird"].vec
)
```

### Clearing Memory

```python
# Remove all symbols
memory.clear()
assert len(memory) == 0
```

## Complete Example: Role-Filler Binding

```python
import jax
from vsax import create_fhrr_model, VSAMemory

# Create FHRR model and memory
model = create_fhrr_model(dim=512)
memory = VSAMemory(model, key=jax.random.PRNGKey(42))

# Add roles and fillers
memory.add_many(["subject", "predicate", "object"])
memory.add_many(["dog", "chases", "cat"])

# Create sentence: "dog chases cat"
subject_dog = model.opset.bind(
    memory["subject"].vec,
    memory["dog"].vec
)

predicate_chases = model.opset.bind(
    memory["predicate"].vec,
    memory["chases"].vec
)

object_cat = model.opset.bind(
    memory["object"].vec,
    memory["cat"].vec
)

# Bundle into sentence representation
sentence = model.opset.bundle(subject_dog, predicate_chases, object_cat)
```

## Reproducibility

Use a PRNG key for deterministic symbol generation:

```python
import jax

key = jax.random.PRNGKey(42)

# Two memories with same key produce identical symbols
memory1 = VSAMemory(create_fhrr_model(dim=512), key=key)
memory2 = VSAMemory(create_fhrr_model(dim=512), key=key)

dog1 = memory1.add("dog")
dog2 = memory2.add("dog")

assert jnp.array_equal(dog1.vec, dog2.vec)  # Identical
```

## Working with Different Models

VSAMemory works identically across all model types:

### FHRR Model

```python
fhrr = create_fhrr_model(dim=512)
memory = VSAMemory(fhrr)
dog = memory.add("dog")
# dog.vec is complex-valued
```

### MAP Model

```python
map_model = create_map_model(dim=512)
memory = VSAMemory(map_model)
feature = memory.add("feature")
# feature.vec is real-valued
```

### Binary Model

```python
binary = create_binary_model(dim=10000, bipolar=True)
memory = VSAMemory(binary)
concept = memory.add("concept")
# concept.vec is bipolar {-1, +1}
```

## API Reference

### VSAMemory Class

```python
class VSAMemory:
    def __init__(self, model: VSAModel, key: Optional[jax.Array] = None):
        """Initialize VSAMemory with a model."""

    def add(self, name: str) -> AbstractHypervector:
        """Add a symbol and return its hypervector."""

    def add_many(self, names: Iterable[str]) -> List[AbstractHypervector]:
        """Add multiple symbols at once."""

    def get(self, name: str) -> AbstractHypervector:
        """Get a hypervector by name (raises KeyError if missing)."""

    def __getitem__(self, name: str) -> AbstractHypervector:
        """Dictionary-style access: memory["dog"]"""

    def __contains__(self, name: str) -> bool:
        """Check if symbol exists: "dog" in memory"""

    def keys(self) -> List[str]:
        """Get all symbol names."""

    def clear(self) -> None:
        """Remove all symbols."""

    def __len__(self) -> int:
        """Number of stored symbols."""
```

## Best Practices

1. **Use factory functions**: Create models with `create_fhrr_model()`, `create_map_model()`, or `create_binary_model()`
2. **Add symbols upfront**: Add all symbols at once with `add_many()` for consistency
3. **Use keys for reproducibility**: Pass a PRNG key when reproducibility matters
4. **Access vectors explicitly**: Use `.vec` to get the underlying array for operations

## Next Steps

- [Factory Functions](factory.md) - Easy model creation
- [Operations Guide](operations.md) - Binding and bundling operations
- [Examples](../examples/fhrr.md) - Complete working examples
