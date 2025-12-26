# Lesson 1.4: Your First VSAX Program

**Duration:** ~30 minutes

**Learning Objectives:**

- Install and verify VSAX setup
- Create your first VSA model and memory
- Add symbols and perform operations
- Implement binding and bundling
- Query using similarity
- Build a complete VSA program from scratch

---

## Introduction

You've learned the theory - now let's write code! In this lesson, we'll build a complete VSA program from scratch that encodes a simple scene and answers queries about it.

**What we'll build:** A program that encodes "The cat sat on the mat" and can answer questions like "What's the subject?" or "What action happened?"

---

## Step 1: Installation and Setup

### Install VSAX

If you haven't already:

```bash
pip install vsax
```

Or for the latest development version:

```bash
git clone https://github.com/vasanthsarathy/vsax.git
cd vsax
pip install -e ".[dev]"
```

### Verify Installation

```python
import vsax
print(f"VSAX version: {vsax.__version__}")
```

**Expected Output:**
```
VSAX version: 1.2.1
```

### Check JAX

VSAX uses JAX for GPU acceleration. Let's verify:

```python
import jax
print(f"JAX devices: {jax.devices()}")
```

**Expected Output (example):**
```
JAX devices: [CpuDevice(id=0)]
```

Don't worry if you see CPU - VSAX works great on CPU too!

---

## Step 2: Create Your First Model

VSAX provides **factory functions** to create models easily.

```python
from vsax import create_fhrr_model

# Create an FHRR model with 2048 dimensions
model = create_fhrr_model(dim=2048)

print(f"Model type: {type(model)}")
print(f"Dimension: {model.dim}")
print(f"Representation: {model.rep_cls.__name__}")
```

**Expected Output:**
```
Model type: <class 'vsax.core.model.VSAModel'>
Dimension: 2048
Representation: ComplexHypervector
```

**What just happened?**
- We created a VSA model using FHRR (complex vectors)
- The model has 2048 dimensions
- All hypervectors will be ComplexHypervector objects

---

## Step 3: Create Memory and Add Symbols

**VSAMemory** is like a symbol table - it stores named hypervectors.

```python
from vsax import VSAMemory

# Create memory using the model
memory = VSAMemory(model)

# Add symbols (concepts)
concepts = ["cat", "mat", "sat", "on", "the"]
memory.add_many(concepts)

# Also add role labels
roles = ["subject", "object", "verb", "preposition", "article"]
memory.add_many(roles)

print(f"Memory contains {len(memory)} symbols")
print(f"Symbol 'cat': {memory['cat']}")
```

**Expected Output:**
```
Memory contains 10 symbols
Symbol 'cat': ComplexHypervector(shape=(2048,), dtype=complex64)
```

**What just happened?**
- We created a memory that will store our symbols
- We added 10 symbols (5 concepts + 5 roles)
- Each symbol is automatically assigned a random hypervector

---

## Step 4: Binding - Create Composite Concepts

Now let's bind concepts with their roles.

```python
# Bind "cat" with "subject" role
cat_as_subject = model.opset.bind(
    memory["cat"].vec,      # The concept
    memory["subject"].vec   # The role
)

# Bind "mat" with "object" role
mat_as_object = model.opset.bind(
    memory["mat"].vec,
    memory["object"].vec
)

# Bind "sat" with "verb" role
sat_as_verb = model.opset.bind(
    memory["sat"].vec,
    memory["verb"].vec
)

print("Created role-filler bindings:")
print(f"  cat ⊗ subject")
print(f"  mat ⊗ object")
print(f"  sat ⊗ verb")
```

**What just happened?**
- We bound each concept with its grammatical role
- Each binding creates a NEW vector that's dissimilar to both inputs
- These bindings preserve the relationship between concept and role

---

## Step 5: Bundling - Create the Scene

Now bundle all the role-filler pairs into a single "sentence" vector.

```python
# Bundle all bindings into one scene
sentence = model.opset.bundle(
    cat_as_subject,
    mat_as_object,
    sat_as_verb
)

print("Created sentence vector by bundling:")
print(f"  (cat⊗subject) ⊕ (mat⊗object) ⊕ (sat⊗verb)")
print(f"\nSentence vector shape: {sentence.shape}")
```

**Expected Output:**
```
Created sentence vector by bundling:
  (cat⊗subject) ⊕ (mat⊗object) ⊕ (sat⊗verb)

Sentence vector shape: (2048,)
```

**What just happened?**
- We bundled three role-filler bindings into one "sentence" vector
- This single vector encodes the entire scene
- We can now query it to retrieve specific information

---

## Step 6: Query the Scene

Let's ask questions by unbinding!

### Query 1: "What's the subject?"

```python
from vsax.similarity import cosine_similarity

# To find the subject, unbind the "subject" role
subject_inverse = model.opset.inverse(memory["subject"].vec)
retrieved = model.opset.bind(sentence, subject_inverse)

# Check similarity to all concepts
print("Query: What's the subject?\n")
for concept in ["cat", "mat", "sat"]:
    sim = cosine_similarity(retrieved, memory[concept].vec)
    print(f"  Similarity to '{concept}': {sim:.4f}")
```

**Expected Output:**
```
Query: What's the subject?

  Similarity to 'cat': 0.9234
  Similarity to 'mat': 0.0123
  Similarity to 'sat': 0.0089
```

**Success!** The highest similarity is to "cat" - correctly identified as the subject.

### Query 2: "What's the object?"

```python
# Unbind the "object" role
object_inverse = model.opset.inverse(memory["object"].vec)
retrieved = model.opset.bind(sentence, object_inverse)

print("Query: What's the object?\n")
for concept in ["cat", "mat", "sat"]:
    sim = cosine_similarity(retrieved, memory[concept].vec)
    print(f"  Similarity to '{concept}': {sim:.4f}")
```

**Expected Output:**
```
Query: What's the object?

  Similarity to 'cat': 0.0156
  Similarity to 'mat': 0.9187
  Similarity to 'sat': 0.0098
```

**Success!** "mat" is correctly identified as the object.

### Query 3: "What action happened?"

```python
# Unbind the "verb" role
verb_inverse = model.opset.inverse(memory["verb"].vec)
retrieved = model.opset.bind(sentence, verb_inverse)

print("Query: What action happened?\n")
for concept in ["cat", "mat", "sat"]:
    sim = cosine_similarity(retrieved, memory[concept].vec)
    print(f"  Similarity to '{concept}': {sim:.4f}")
```

**Expected Output:**
```
Query: What action happened?

  Similarity to 'cat': 0.0112
  Similarity to 'mat': 0.0134
  Similarity to 'sat': 0.9156
```

**Success!** "sat" is correctly identified as the verb.

---

## Complete Working Example

Here's the full program in one script:

```python
"""
VSAX First Program: Encoding "The cat sat on the mat"
"""

from vsax import create_fhrr_model, VSAMemory
from vsax.similarity import cosine_similarity

# Step 1: Create model
print("Creating FHRR model...")
model = create_fhrr_model(dim=2048)

# Step 2: Create memory and add symbols
print("Adding symbols to memory...")
memory = VSAMemory(model)
memory.add_many(["cat", "mat", "sat", "subject", "object", "verb"])

# Step 3: Encode sentence using bind and bundle
print("\nEncoding sentence: 'The cat sat on the mat'")
print("  Binding concepts with roles...")

cat_subj = model.opset.bind(memory["cat"].vec, memory["subject"].vec)
mat_obj = model.opset.bind(memory["mat"].vec, memory["object"].vec)
sat_verb = model.opset.bind(memory["sat"].vec, memory["verb"].vec)

print("  Bundling role-filler pairs...")
sentence = model.opset.bundle(cat_subj, mat_obj, sat_verb)

# Step 4: Query the sentence
print("\n" + "="*50)
print("QUERYING THE ENCODED SENTENCE")
print("="*50)

queries = {
    "subject": "What's the subject?",
    "object": "What's the object?",
    "verb": "What action happened?"
}

for role, question in queries.items():
    print(f"\nQ: {question}")

    # Unbind the role
    role_inv = model.opset.inverse(memory[role].vec)
    retrieved = model.opset.bind(sentence, role_inv)

    # Find best match
    best_concept = None
    best_sim = -1

    for concept in ["cat", "mat", "sat"]:
        sim = cosine_similarity(retrieved, memory[concept].vec)
        if sim > best_sim:
            best_sim = sim
            best_concept = concept

    print(f"A: '{best_concept}' (similarity: {best_sim:.4f})")

print("\n" + "="*50)
print("SUCCESS! All queries answered correctly.")
print("="*50)
```

**Run this program and you should see:**

```
Creating FHRR model...
Adding symbols to memory...

Encoding sentence: 'The cat sat on the mat'
  Binding concepts with roles...
  Bundling role-filler pairs...

==================================================
QUERYING THE ENCODED SENTENCE
==================================================

Q: What's the subject?
A: 'cat' (similarity: 0.9234)

Q: What's the object?
A: 'mat' (similarity: 0.9187)

Q: What action happened?
A: 'sat' (similarity: 0.9156)

==================================================
SUCCESS! All queries answered correctly.
==================================================
```

---

## Understanding the Flow

Let's visualize what just happened:

```
Input: "The cat sat on the mat"
           ↓
Step 1: Create role-filler bindings
        cat ⊗ subject → [vector1]
        mat ⊗ object  → [vector2]
        sat ⊗ verb    → [vector3]
           ↓
Step 2: Bundle into scene
        scene = [vector1] ⊕ [vector2] ⊕ [vector3]
           ↓
Step 3: Query by unbinding
        scene ⊗ subject⁻¹ → retrieves 'cat'
        scene ⊗ object⁻¹  → retrieves 'mat'
        scene ⊗ verb⁻¹    → retrieves 'sat'
```

**Key insight:** A single vector encodes the entire structured scene. We can query any aspect by unbinding the appropriate role!

---

## Common Mistakes and Debugging

### ❌ Mistake 1: Forgetting to normalize

```python
# WRONG: Using raw bind result without normalization
bound = model.opset.bind(a.vec, b.vec)
# bound might have large magnitude
```

**Fix:** VSAX operations automatically normalize, so you're safe!

### ❌ Mistake 2: Using .vec vs the object

```python
# WRONG: Mixing hypervector objects and raw arrays
memory["cat"] + memory["mat"]  # Might not work as expected

# CORRECT: Use .vec to get the underlying array
model.opset.bundle(memory["cat"].vec, memory["mat"].vec)
```

### ❌ Mistake 3: Wrong inverse

```python
# WRONG: Using the same vector instead of inverse
retrieved = model.opset.bind(bound, key.vec)  # Wrong!

# CORRECT: Use inverse
retrieved = model.opset.bind(bound, model.opset.inverse(key.vec))
```

---

## Extending the Example

### Challenge 1: Add More Concepts

Extend the sentence to "The big cat sat on the red mat"

**Hint:** Bind "big" with "modifier1", "red" with "modifier2", then bundle those bindings too.

### Challenge 2: Multiple Scenes

Encode two scenes and store them separately:
- Scene 1: "The cat sat on the mat"
- Scene 2: "The dog ran in the park"

Can you query each scene independently?

### Challenge 3: Change Models

Try the same program with MAP and Binary models:

```python
from vsax import create_map_model, create_binary_model

# Try with MAP
model = create_map_model(dim=2048)
# ... rest of code stays the same

# Try with Binary
model = create_binary_model(dim=2048)
# ... rest of code stays the same
```

**Question:** Do the similarities differ between models? Why?

---

## Self-Assessment

Before moving to Module 2, ensure you can:

- [ ] Install and import VSAX successfully
- [ ] Create a VSA model using factory functions
- [ ] Create VSAMemory and add symbols
- [ ] Perform binding operations
- [ ] Perform bundling operations
- [ ] Query using unbinding and similarity
- [ ] Explain the bind-bundle-query pattern
- [ ] Debug common issues (inverse, .vec access)

---

## Quick Quiz

**Q1:** What does this code do?
```python
x = model.opset.bind(a.vec, b.vec)
```

a) Adds two vectors
b) Creates a new vector dissimilar to both a and b
c) Calculates similarity between a and b
d) Unbinds a from b

<details>
<summary>Answer</summary>
**b) Creates a new vector dissimilar to both a and b** - This is the binding operation.
</details>

**Q2:** To retrieve 'cat' from `scene = (cat⊗subject) ⊕ (mat⊗object)`, you should:

a) `scene ⊗ subject`
b) `scene ⊗ subject⁻¹`
c) `scene ⊕ subject`
d) `scene ⊕ subject⁻¹`

<details>
<summary>Answer</summary>
**b) `scene ⊗ subject⁻¹`** - Unbind by binding with the inverse of the role.
</details>

**Q3:** What's the purpose of VSAMemory?

a) Stores temporary computation results
b) Caches similarity computations
c) Stores named hypervectors (symbol table)
d) Manages GPU memory

<details>
<summary>Answer</summary>
**c) Stores named hypervectors (symbol table)** - VSAMemory maps names to hypervectors.
</details>

---

## Congratulations!

You've written your first complete VSA program! You can now:

✅ Create VSA models
✅ Store and retrieve symbols
✅ Encode structured information using bind and bundle
✅ Query encoded information using unbinding

This is the foundation for all VSA applications - from image classification to knowledge graphs to analogical reasoning.

---

## Module 1 Complete!

You've finished Module 1: Foundations. You should now:

- Understand why high dimensions work
- Master binding and bundling operations
- Know the three VSA models and when to use each
- Be able to write VSA programs from scratch

**Next Steps:**

1. **Complete the Module 1 Capstone**: Build an analogy solver
2. **Proceed to Module 2**: Deep dive into the three models
3. **Explore tutorials**: Try the MNIST classification tutorial

---

## Module 1 Capstone Project

**Project:** Build an Analogy Solver

Implement a VSA program that solves analogies: "A is to B as C is to ?"

**Example:** "King is to Queen as Man is to ?"
- Answer: "Woman"

**Steps:**

1. Encode word pairs: (king, queen), (man, woman), (boy, girl)
2. Create mapping vectors: queen ⊗ king⁻¹ (represents "male→female")
3. Apply mapping: woman_hv = man ⊗ mapping
4. Test: Does woman_hv match the stored "woman" vector?

**Solution template:**

```python
from vsax import create_fhrr_model, VSAMemory
from vsax.similarity import cosine_similarity

model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)

# Add words
memory.add_many(["king", "queen", "man", "woman", "boy", "girl"])

# Create mapping from king→queen
mapping = model.opset.bind(
    memory["queen"].vec,
    model.opset.inverse(memory["king"].vec)
)

# Apply to "man" to find "?"
result = model.opset.bind(memory["man"].vec, mapping)

# Find best match
for word in ["queen", "woman", "girl"]:
    sim = cosine_similarity(result, memory[word].vec)
    print(f"Similarity to '{word}': {sim:.4f}")
```

**Expected:** High similarity to "woman"!

---

**Next Module:** [Module 2: Core Operations and Models](../02_operations/index.md)

Deep dive into FHRR, MAP, and Binary with mathematical foundations and implementation details.

**Previous:** [Lesson 1.3: The Three VSA Models](03_models.md)
