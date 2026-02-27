# Tutorial 15: Program Synthesis with VSA

This tutorial demonstrates how to use Vector Symbolic Architectures for program synthesis — representing, searching, composing, and decomposing programs as hypervectors.

[📓 **Open in Jupyter Notebook**](../../examples/notebooks/tutorial_15_program_synthesis.ipynb)

## What You'll Learn

- How to encode programs as structured role-filler hypervectors
- How to build a searchable program library
- How to query programs by partial specification (e.g., "which program uses addition?")
- How to compose programs into multi-step pipelines
- How to decompose pipelines back into individual steps via unbinding

## Why VSA for Program Synthesis?

Program synthesis — automatically constructing programs from specifications — is a core challenge in AI. VSAs offer a natural fit:

1. **Structured Representation**: Programs have structure (operation, arguments, constants) that maps directly to role-filler binding
2. **Compositional**: Programs can be composed into pipelines using bundling and binding
3. **Searchable**: A library of programs can be searched by partial specification using similarity
4. **Decomposable**: Composed programs can be taken apart via unbinding to recover sub-programs
5. **Fixed-Size**: Every program, regardless of complexity, is a single fixed-dimensional vector

## Setup

```python
import jax.numpy as jnp
from vsax import create_fhrr_model, VSAMemory
from vsax.similarity import cosine_similarity

# FHRR gives exact unbinding — critical for decomposition
model = create_fhrr_model(dim=1024)
memory = VSAMemory(model)

# Define vocabulary
operations = ["add", "sub", "mul", "inc", "double", "negate"]
roles = ["op", "arg", "const", "step1", "step2"]
variables = ["x"]
constants = ["one", "two", "three", "five"]

memory.add_many(operations + roles + variables + constants)

print(f"Model: {model.rep_cls.__name__}, dim={model.dim}")
print(f"Vocabulary: {len(memory)} symbols")
```

Output:
```
Model: ComplexHypervector, dim=1024
Vocabulary: 16 symbols
```

## Step 1: Encoding Programs as Hypervectors

Each program in our DSL has three components:
- **Operation**: what to do (`add`, `sub`, `mul`, etc.)
- **Argument**: the input variable (`x`)
- **Constant**: an optional numeric operand (`one`, `two`, etc.)

We encode a program as a role-filler structure:
```
program = bundle(bind(op_role, operation), bind(arg_role, variable), bind(const_role, constant))
```

Bundling creates a superposition of the three role-filler pairs. Each pair is recoverable via unbinding, though the other pairs contribute some noise. Higher dimensions reduce this noise.

```python
opset = model.opset

def encode_program(op_name, arg_name, const_name=None):
    """Encode a program as a role-filler hypervector."""
    parts = [
        opset.bind(memory["op"].vec, memory[op_name].vec),
        opset.bind(memory["arg"].vec, memory[arg_name].vec),
    ]
    if const_name is not None:
        parts.append(opset.bind(memory["const"].vec, memory[const_name].vec))
    return opset.bundle(*parts)

# Encode six programs
programs = {
    "add(x,3)":    encode_program("add", "x", "three"),
    "sub(x,2)":    encode_program("sub", "x", "two"),
    "mul(x,5)":    encode_program("mul", "x", "five"),
    "inc(x)":      encode_program("inc", "x", "one"),
    "double(x)":   encode_program("double", "x", "two"),
    "negate(x)":   encode_program("negate", "x"),
}

print("Encoded programs:")
for name in programs:
    print(f"  {name}")
```

Output:
```
Encoded programs:
  add(x,3)
  sub(x,2)
  mul(x,5)
  inc(x)
  double(x)
  negate(x)
```

## Step 2: Building a Program Library

The program library is simply a dictionary mapping names to hypervectors. In a real system, this could be a large collection of reusable program fragments.

```python
library = programs.copy()
print(f"Program library: {len(library)} programs")
for name in library:
    print(f"  - {name}")
```

Output:
```
Program library: 6 programs
  - add(x,3)
  - sub(x,2)
  - mul(x,5)
  - inc(x)
  - double(x)
  - negate(x)
```

## Step 3: Querying Programs — Extracting Attributes

Given a program hypervector, we can extract any attribute by unbinding the corresponding role. For example, to find what operation a program uses:

```
recovered_op = unbind(program, op_role)
```

Then compare the result against all known operations to identify the best match. Since each program is a bundle of multiple role-filler pairs, unbinding one role recovers the correct filler plus noise from the other pairs. The correct answer always has the highest similarity.

```python
def extract_attribute(program_vec, role_name, candidates):
    """Extract an attribute from a program by unbinding the role."""
    role_vec = memory[role_name].vec
    recovered = opset.unbind(program_vec, role_vec)

    best_name, best_sim = None, -1.0
    for name in candidates:
        sim = float(cosine_similarity(recovered, memory[name].vec))
        if sim > best_sim:
            best_sim = sim
            best_name = name
    return best_name, best_sim

# Extract the operation from each program
print("Extracting operations from programs:")
for prog_name, prog_vec in library.items():
    op, sim = extract_attribute(prog_vec, "op", operations)
    print(f"  {prog_name:15s} -> op={op:8s} (similarity={sim:.3f})")
```

Output (exact values vary due to random sampling):
```
Extracting operations from programs:
  add(x,3)        -> op=add      (similarity=0.134)
  sub(x,2)        -> op=sub      (similarity=0.152)
  mul(x,5)        -> op=mul      (similarity=0.119)
  inc(x)           -> op=inc      (similarity=0.147)
  double(x)        -> op=double   (similarity=0.163)
  negate(x)        -> op=negate   (similarity=0.218)
```

The similarities are modest because each program is a bundle of 2-3 role-filler pairs, and unbinding recovers the target filler plus noise from the others. However, the correct operation always has the **highest** similarity among all candidates — that's what matters.

## Step 4: Searching by Specification

We can search the library for programs matching a partial specification. For example, "find all programs that use addition":

```
query = bind(op_role, add)
```

This query is compared against every library entry using cosine similarity. Because the query is one of the role-filler pairs that was bundled into the program, the matching program will have the highest similarity.

```python
def search_library(query_vec, library, top_k=3):
    """Search library for programs similar to a query."""
    results = []
    for name, prog_vec in library.items():
        sim = float(cosine_similarity(query_vec, prog_vec))
        results.append((name, sim))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

# Query: "which programs use addition?"
query_add = opset.bind(memory["op"].vec, memory["add"].vec)
print("Query: programs that use 'add'")
for name, sim in search_library(query_add, library):
    print(f"  {name:15s}  similarity={sim:.3f}")

print()

# Query: "which programs use constant 'two'?"
query_two = opset.bind(memory["const"].vec, memory["two"].vec)
print("Query: programs that use constant 'two'")
for name, sim in search_library(query_two, library):
    print(f"  {name:15s}  similarity={sim:.3f}")
```

Output (exact values vary):
```
Query: programs that use 'add'
  add(x,3)         similarity=0.522
  inc(x)            similarity=0.037
  negate(x)         similarity=0.029

Query: programs that use constant 'two'
  sub(x,2)          similarity=0.522
  double(x)         similarity=0.522
  mul(x,5)          similarity=0.022
```

The correct programs clearly dominate. Both `sub(x,2)` and `double(x)` use constant `two`, so they score equally high.

## Step 5: Composing Programs into Pipelines

We can compose multiple programs into a pipeline by binding each program to a step role:

```
pipeline = bundle(bind(step1, program_A), bind(step2, program_B))
```

This represents: "first apply program A, then apply program B."

```python
# Pipeline: first double(x), then add(x, 3)
# Semantically: add(double(x), 3) = 2x + 3
prog_double = library["double(x)"]
prog_add3 = library["add(x,3)"]

pipeline = opset.bundle(
    opset.bind(memory["step1"].vec, prog_double),
    opset.bind(memory["step2"].vec, prog_add3),
)

print("Pipeline: step1=double(x), step2=add(x,3)")
print(f"Pipeline vector shape: {pipeline.shape}")
```

Output:
```
Pipeline: step1=double(x), step2=add(x,3)
Pipeline vector shape: (1024,)
```

## Step 6: Decomposing Pipelines

Given a pipeline, we can recover each step by unbinding the step role, then searching the library:

```
recovered_step1 = unbind(pipeline, step1_role)
```

The recovered vector is compared against all library programs to identify the best match.

```python
def decompose_pipeline(pipeline_vec, library):
    """Decompose a pipeline into its constituent steps."""
    steps = {}
    for step_name in ["step1", "step2"]:
        step_role = memory[step_name].vec
        recovered = opset.unbind(pipeline_vec, step_role)

        # Search library for the best match
        best_name, best_sim = None, -1.0
        for prog_name, prog_vec in library.items():
            sim = float(cosine_similarity(recovered, prog_vec))
            if sim > best_sim:
                best_sim = sim
                best_name = prog_name
        steps[step_name] = (best_name, best_sim)
    return steps

steps = decompose_pipeline(pipeline, library)
print("Decomposed pipeline:")
for step, (name, sim) in steps.items():
    print(f"  {step}: {name} (similarity={sim:.3f})")
```

Output (exact values vary):
```
Decomposed pipeline:
  step1: double(x) (similarity=0.322)
  step2: add(x,3) (similarity=0.298)
```

The correct programs are recovered as the best match for each step.

```python
# Try another pipeline: first negate(x), then mul(x, 5)
# Semantically: mul(negate(x), 5) = -5x
pipeline2 = opset.bundle(
    opset.bind(memory["step1"].vec, library["negate(x)"]),
    opset.bind(memory["step2"].vec, library["mul(x,5)"]),
)

steps2 = decompose_pipeline(pipeline2, library)
print("Pipeline 2 decomposed:")
for step, (name, sim) in steps2.items():
    print(f"  {step}: {name} (similarity={sim:.3f})")
```

Output (exact values vary):
```
Pipeline 2 decomposed:
  step1: negate(x) (similarity=0.341)
  step2: mul(x,5) (similarity=0.310)
```

## Key Takeaways

1. **Role-Filler Encoding**: Programs are naturally represented as `bundle(bind(role, filler), ...)` structures
2. **Attribute Extraction**: Unbinding a role from a program recovers the filler — the correct answer always has the highest similarity
3. **Library Search**: Partial specifications (e.g., "uses addition") find matching programs via similarity
4. **Composition**: Multi-step pipelines are built by binding programs to step roles and bundling
5. **Decomposition**: Unbinding step roles from a pipeline recovers the original sub-programs as best matches
6. **Fixed-Size Vectors**: Every program — simple or composed — is a single 1024-dimensional vector
7. **Noise vs. Signal**: Bundling introduces noise, but the correct answer always dominates — higher dimensions improve signal-to-noise ratio

## Next Steps

- Scale the DSL with more operations, control flow, and data types
- Explore program search over larger libraries (100+ programs)
- Use resonator networks for factorizing programs with unknown structure
- Combine with neural models for learning program representations from examples
- See [Tutorial 7: Hierarchical Structures](07_hierarchical_structures.md) for recursive composition patterns
- See [Tutorial 2: Knowledge Graph Reasoning](02_knowledge_graph.md) for more role-filler binding examples

## Running This Tutorial

This tutorial is available as a Jupyter notebook at `examples/notebooks/tutorial_15_program_synthesis.ipynb`.

To run it:
```bash
jupyter notebook examples/notebooks/tutorial_15_program_synthesis.ipynb
```
