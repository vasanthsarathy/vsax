# VSAX Tutorial Roadmap

This document outlines planned tutorials and examples for VSAX, organized by category and priority.

---

## Completed Tutorials ✅

### Tutorial 1: MNIST Digit Classification
**Status**: ✅ Complete
**Level**: Beginner
**Topics**: Image encoding, prototype learning, similarity-based classification
**Files**:
- `examples/notebooks/tutorial_01_mnist_classification.ipynb`
- `docs/tutorials/01_mnist_classification.md`

Learn how to use VSA for image classification with the classic MNIST dataset. Compare different VSA models (FHRR, MAP, Binary) and achieve 95%+ accuracy using simple prototype matching.

### Tutorial 2: Knowledge Graph Reasoning
**Status**: ✅ Complete
**Level**: Intermediate
**Topics**: Graph encoding, factorization, multi-hop reasoning
**Files**:
- `examples/notebooks/tutorial_02_knowledge_graph.ipynb`
- `docs/tutorials/02_knowledge_graph.md`

Build and query a knowledge graph using VSA. Encode relational facts (triples), perform queries using unbinding, use resonator networks to decode compositional structures, and perform multi-hop reasoning for property inheritance.

### Tutorial 3: Kanerva's "Dollar of Mexico" - Analogical Reasoning
**Status**: ✅ Complete
**Level**: Advanced
**Topics**: Holistic encoding, mapping vectors, prototypes, analogical reasoning
**Files**:
- `examples/notebooks/tutorial_03_kanerva_analogies.ipynb`
- `docs/tutorials/03_kanerva_analogies.md`

Implement the classic examples from Pentti Kanerva's foundational paper on hyperdimensional computing. Learn to encode structured records holistically, compute mapping vectors from examples, perform analogical queries like "What's the dollar of Mexico?", solve IQ-test analogies, and chain mappings transitively.

### Tutorial 4: Word Analogies & Random Indexing
**Status**: ✅ Complete
**Level**: Intermediate
**Topics**: Word embeddings, semantic similarity, Random Indexing, word analogies
**Files**:
- `examples/notebooks/tutorial_04_word_analogies.ipynb`
- `docs/tutorials/04_word_analogies.md`

Build word embeddings using Random Indexing (Kanerva et al. 2000) and perform classic word analogies like "king - man + woman = queen". Learn how context co-occurrence shapes meaning, perform semantic similarity search, compare VSA models for NLP tasks, and understand vector composition for analogical reasoning.

### Tutorial 5: Understanding VSA Models - Comparative Analysis
**Status**: ✅ Complete
**Level**: Intermediate
**Topics**: Model comparison, FHRR vs MAP vs Binary, capacity analysis, noise tolerance
**Files**:
- `examples/notebooks/tutorial_05_model_comparison.ipynb`
- `docs/tutorials/05_model_comparison.md`

Compare all three VSA models (FHRR, MAP, Binary) across classification accuracy, noise robustness, capacity analysis, and speed benchmarks. Learn when to use each model, understand the trade-offs between accuracy, speed, and memory, and get a practical decision guide for choosing the right model for your task.

### Tutorial 6: VSA for Edge Computing - Lightweight Alternative to Neural Networks
**Status**: ✅ Complete
**Level**: Intermediate
**Topics**: Edge computing, VSA vs neural networks, efficiency, deployment, resource constraints
**Files**:
- `examples/notebooks/tutorial_06_edge_computing.ipynb`
- `docs/tutorials/06_edge_computing.md`

Compare VSA with neural networks on Fashion-MNIST classification. Demonstrates VSA's advantages for edge computing: 4-10x faster training, comparable model size, and similar accuracy without gradient descent. Shows when to choose VSA over neural networks for resource-constrained environments (IoT, wearables, embedded systems).

---

## Planned Tutorials - Next Priority 🎯

### Tutorial 7: Hierarchical Structures - Trees & Nested Composition ⭐⭐
**Status**: 📋 Planned
**Level**: Advanced
**Topics**: Recursive binding, parse trees, deep composition, resonator factorization
**Priority**: **NEXT**

**Learning Goals**:
- Encode hierarchical structures (parse trees, nested lists)
- Recursive role-filler binding
- Decode deep structures with resonators
- Handle variable-depth trees

**Key Features Demonstrated**:
- Deep compositional power of VSA
- Resonator networks for multi-level factorization
- Recursive encoding patterns
- Tree traversal and search

**Examples**:
1. **Arithmetic Expressions**: `(2 + 3) * (4 - 1)` → tree → evaluate
2. **Parse Trees**: Sentence syntax trees
3. **Nested Data**: JSON-like structures
4. **Family Trees**: Hierarchical relationships

**Key Challenge**: Factorizing deeply nested structures with resonators

**Reference Papers**:
- Plate, T. A. (1995). "Holographic Reduced Representations"
- Frady et al. (2020). "Resonator networks" (for factorization)

**Why This Tutorial**:
- Shows advanced VSA capabilities
- Unique to VSAX (resonator networks)
- Gap in existing tutorials
- Impressive demonstrations

---

## Natural Language Processing Tutorials 🔤

### Tutorial 7: Sentence Encoding & Semantic Similarity
**Status**: 💡 Idea
**Level**: Intermediate
**Topics**: Sentence encoding, bag-of-words vs compositional, similarity metrics
**Priority**: Medium

**Learning Goals**:
- Encode sentences compositionally
- Role-filler binding for syntax (subject-verb-object)
- Sentence similarity and paraphrase detection
- Compare different encoding strategies

**Examples**:
- "The dog chased the cat" vs "The cat was chased by the dog" (paraphrase)
- "The man ate the pizza" vs "The pizza ate the man" (role importance)
- Similarity search in sentence database

**Key Features**: SequenceEncoder, DictEncoder composition

---

## Symbolic Reasoning & Logic Tutorials 🎲

### Tutorial 8: Solving Logic Puzzles with VSA
**Status**: 💡 Idea
**Level**: Advanced
**Topics**: Constraint satisfaction, search, symbolic reasoning
**Priority**: Medium

**Learning Goals**:
- Encode constraints as hypervectors
- Search through solution space
- Verify solutions with similarity
- Handle backtracking and pruning

**Example Puzzles**:
1. **Sudoku**: Encode board state, constraints, search for solution
2. **N-Queens**: Place N queens on chessboard
3. **Logic Grid Puzzles**: "Einstein's Riddle"
4. **Graph Coloring**: Color graph nodes with constraints

**Key Features**: Resonator networks for constraint satisfaction, search

**Why This Tutorial**: Fun, impressive, shows reasoning capabilities

---

### Tutorial 9: Planning & Problem Solving with VSA
**Status**: 💡 Idea
**Level**: Advanced
**Topics**: State space search, action sequences, goal-directed reasoning
**Priority**: Low

**Examples**:
- Blocks world planning
- Route finding with obstacles
- Game playing (tic-tac-toe, simple board games)
- Robotic task planning

---

## Advanced VSA Concepts Tutorials 📊

### Tutorial 10: Noise, Capacity & Dimensionality Analysis
**Status**: 💡 Idea
**Level**: Intermediate
**Topics**: Information capacity, noise tolerance, dimension effects
**Priority**: High

**Learning Goals**:
- How many items can you bundle before losing information?
- Effect of dimensionality on accuracy and robustness
- Noise analysis and recovery thresholds
- Optimal dimension selection

**Experiments**:
1. **Bundling Capacity**: Bundle 10, 100, 1000 vectors - measure retrieval accuracy
2. **Noise Tolerance**: Add random noise - find recovery threshold
3. **Dimension Sweep**: Same task with dim=128, 512, 1024, 10000
4. **Interference Analysis**: How similar can items be before collision?

**Deliverables**:
- Capacity plots (items vs accuracy)
- Noise tolerance curves
- Dimension selection guide
- Theoretical vs empirical comparison

**Why This Tutorial**: Very educational, helps understand VSA limits, practical for tuning

---

### Tutorial 11: Understanding Binding Operations
**Status**: 💡 Idea
**Level**: Beginner/Intermediate
**Topics**: XOR vs Convolution vs Multiplication, mathematical properties
**Priority**: Medium

**Learning Goals**:
- Compare binding operations: XOR (Binary), Convolution (FHRR), Multiplication (MAP)
- Mathematical properties: commutative, associative, distributive
- When each binding makes sense
- Geometric intuition

**Interactive Demonstrations**:
- Visualize binding in 2D/3D (projection from high-D)
- Show distributivity over bundling
- Demonstrate exact vs approximate unbinding

---

### Tutorial 12: Similarity Metrics Explained
**Status**: 💡 Idea
**Level**: Beginner
**Topics**: Cosine vs Dot vs Hamming similarity
**Priority**: Medium

**Learning Goals**:
- When to use cosine vs dot vs Hamming
- Geometric interpretation
- Normalized vs unnormalized
- Distance vs similarity

**Examples**:
- Same vectors, different metrics - different results
- Choosing metric for your data type
- Performance implications

---

## Time Series & Sequences Tutorials ⏱️

### Tutorial 13: Sequence Prediction & Pattern Matching
**Status**: 💡 Idea
**Level**: Intermediate
**Topics**: Time series forecasting, sequence recognition, anomaly detection
**Priority**: Medium

**Applications**:
- Stock price prediction
- Sensor data anomaly detection
- Activity recognition from sequences
- Pattern matching in signals

**Key Features**: Temporal binding, SequenceEncoder, streaming data

---

### Tutorial 14: Gesture & Activity Recognition
**Status**: 💡 Idea
**Level**: Intermediate
**Topics**: Sensor sequences, activity patterns, real-time recognition
**Priority**: Low

**Dataset**: Accelerometer/gyroscope data for gestures

**Applications**:
- Smartphone gesture recognition
- Wearable activity tracking
- Sign language recognition

---

## Domain-Specific Applications 🧬

### Tutorial 15: DNA/Protein Sequence Analysis
**Status**: 💡 Idea
**Level**: Advanced
**Topics**: Bioinformatics, k-mer encoding, sequence alignment
**Priority**: Medium

**Following hdlib's Success**: Build on proven bioinformatics applications

**Learning Goals**:
- Encode DNA/protein sequences
- k-mer based encoding
- Sequence similarity and alignment
- Motif discovery

**Dataset**: Genomic sequences, protein databases

---

### Tutorial 16: Molecular Fingerprints (Chemistry)
**Status**: 💡 Idea
**Level**: Advanced
**Topics**: Chemical similarity, SMILES encoding, property prediction
**Priority**: Low

**Applications**:
- Drug discovery
- Chemical similarity search
- Property prediction (toxicity, solubility)

**Key Features**: GraphEncoder for molecular graphs

---

### Tutorial 17: Recommendation System
**Status**: 💡 Idea
**Level**: Intermediate
**Topics**: Collaborative filtering, user-item encoding, similarity search
**Priority**: Medium

**Learning Goals**:
- Encode user preferences
- Encode item features
- Similarity-based recommendations
- Handle cold start problem

**Dataset**: MovieLens, book ratings, product reviews

---

## Computer Vision Tutorials 🖼️

### Tutorial 18: Visual Analogies
**Status**: 💡 Idea
**Level**: Intermediate
**Topics**: Geometric transformations, shape relationships, visual reasoning
**Priority**: Low

**Examples**:
- "Square is to cube as circle is to ?"
- "Rotate 90° clockwise" as mapping
- Size, color, shape transformations

---

### Tutorial 19: Object Composition & Scene Understanding
**Status**: 💡 Idea
**Level**: Advanced
**Topics**: Multi-attribute binding, spatial relationships, compositional vision
**Priority**: Low

**Examples**:
- "Red ball on blue table"
- "Large dog next to small cat"
- Scene graphs

---

## Practical/Meta Tutorials 🔧

### Tutorial 20: Building Custom Encoders
**Status**: 💡 Idea
**Level**: Intermediate
**Topics**: Extending AbstractEncoder, design patterns, best practices
**Priority**: High

**Learning Goals**:
- Design your own encoder
- When to use fit() vs encode()
- Handling special data types
- Testing and validation

**Examples**:
- DateEncoder (already in examples)
- ColorEncoder (already in examples)
- AudioEncoder (spectrograms)
- ImagePatchEncoder

**Template Provided**: `examples/custom_encoder_template.py`

---

### Tutorial 21: Performance Tuning & GPU Optimization
**Status**: 💡 Idea
**Level**: Advanced
**Topics**: Batch operations, JIT, memory management, benchmarking
**Priority**: Medium

**Learning Goals**:
- Use vmap for batch operations
- JIT compilation for speed
- GPU vs CPU benchmarking
- Memory-efficient encoding

**Key Features**: JAX integration, device utilities (already in VSAX)

---

### Tutorial 22: Debugging VSA Applications
**Status**: 💡 Idea
**Level**: Intermediate
**Topics**: Common failure modes, diagnosis, visualization
**Priority**: Medium

**Common Issues**:
- Low similarity scores - why?
- Bundling too many items
- Wrong encoder for data type
- Dimension too small/large

**Debugging Tools**:
- Similarity inspection
- Noise analysis
- Visualization techniques
- Unit testing VSA code

---

## Educational Deep-Dives 🎓

### Tutorial 23: VSA Theory - Capacity Analysis
**Status**: 💡 Idea
**Level**: Advanced
**Topics**: Information theory, capacity bounds, theoretical limits
**Priority**: Low

**Learning Goals**:
- Information-theoretic perspective
- Theoretical capacity bounds
- Empirical vs theoretical
- Trade-offs and limits

**Reference Papers**:
- Kanerva's theoretical work
- Information-theoretic analyses of VSA

---

## Quick Examples (Not Full Tutorials)

Beyond full tutorials, short focused examples:

### Code Examples to Add

1. **examples/word_analogies.py** - Quick word analogy demo
2. **examples/noise_tolerance.py** - Bundle until breakdown
3. **examples/dimension_sweep.py** - Accuracy vs dimension analysis
4. **examples/model_comparison.py** - FHRR vs MAP vs Binary side-by-side
5. **examples/custom_encoder_template.py** - Starter template for users
6. **examples/hybrid_neural_vsa.py** - VSA + small neural network
7. **examples/streaming_sequences.py** - Online/incremental encoding
8. **examples/visualization.py** - Plot hypervector spaces (PCA/t-SNE)
9. **examples/feature_extraction.py** - VSA for feature engineering
10. **examples/compositional_vision.py** - Multi-attribute objects

### Utility Scripts

11. **examples/utils/benchmark.py** - Standard benchmarking suite
12. **examples/utils/dimension_selection.py** - Helper for choosing dimension
13. **examples/utils/visualize_space.py** - Visualization utilities

---

## Interactive Ideas 🎮

### Jupyter Widgets
- Interactive sliders for dimension, noise, bundle count
- Real-time similarity updates
- Live visualization of operations

### Google Colab Notebooks
- Click-to-run versions of all tutorials
- No local setup required
- Pre-loaded with datasets

### VSA Playground (Streamlit App)
- Web interface for experimentation
- Upload your data
- Try different encoders
- Visualize results

### Video Tutorials
- Screencasts explaining concepts
- Live coding sessions
- Concept animations

---

## Prioritization Matrix

### High Priority (Do Soon)
1. ⭐⭐⭐ **Word Analogies & Random Indexing** - Classic VSA, impressive
2. ⭐⭐⭐ **Model Comparison** - Educational, practical
3. ⭐⭐⭐ **Noise & Capacity Analysis** - Understanding limits
4. ⭐⭐ **Hierarchical Structures** - Advanced composition, unique
5. ⭐⭐ **Custom Encoder Guide** - Extensibility, practical

### Medium Priority (Good to Have)
6. ⭐⭐ **Logic Puzzles** - Fun, impressive
7. ⭐⭐ **Sentence Encoding** - NLP extension
8. ⭐⭐ **Similarity Metrics Explained** - Beginner-friendly
9. ⭐⭐ **Debugging Guide** - Very practical
10. ⭐ **Sequence Prediction** - Time series applications

### Lower Priority (Future)
11. ⭐ **DNA Sequence Analysis** - Domain-specific
12. ⭐ **Visual Analogies** - Different domain
13. ⭐ **Recommendation System** - Practical ML
14. ⭐ **Performance Tuning** - Advanced optimization
15. ⭐ **VSA Theory** - Theoretical deep-dive

---

## Tutorial Development Workflow

For each tutorial:

### Phase 1: Planning
1. Define learning objectives
2. Choose dataset/domain
3. Outline key concepts
4. Identify VSAX features to showcase

### Phase 2: Implementation
1. Create Jupyter notebook (`.ipynb`)
   - Interactive, runnable code
   - Visualizations and plots
   - Clear explanations
   - Expected outputs
2. Create documentation version (`.md`)
   - Same content as notebook
   - Formatted for docs site
   - Includes expected outputs
   - Links to notebook

### Phase 3: Integration
1. Add to `docs/tutorials/index.md`
2. Update `README.md`
3. Update `mkdocs.yml` navigation
4. Add any new dependencies
5. Create example scripts if needed

### Phase 4: Quality
1. Test notebook executes without errors
2. Verify outputs are correct
3. Check documentation formatting
4. Spell check and grammar
5. Cross-reference with other tutorials

---

## Tutorial Template Structure

Each tutorial should follow this structure:

```markdown
# Tutorial N: [Title]

[Brief introduction and motivation]

## What You'll Learn

- Bullet point 1
- Bullet point 2
- Bullet point 3

## Why [This Topic]?

[Explain importance and applications]

## Setup

[Code for imports and basic setup]

## Part 1: [Concept 1]

[Explanation and code]

## Part 2: [Concept 2]

[Explanation and code]

...

## Key Takeaways

1. Summary point 1
2. Summary point 2
3. Summary point 3

## Next Steps

- Extension 1
- Extension 2
- Related tutorials

## Running This Tutorial

[Instructions for running notebook]

## References

[Papers, docs, related work]
```

---

## Success Metrics

For each tutorial, we aim for:

- ✅ **Clear learning objectives** - Users know what they'll learn
- ✅ **Runnable code** - All code executes without errors
- ✅ **Expected outputs** - Users can verify correctness
- ✅ **Explanations** - Concepts explained, not just code
- ✅ **Visualizations** - Plots/figures where helpful
- ✅ **VSAX features** - Showcases library capabilities
- ✅ **Real datasets** - Uses actual data, not toy examples
- ✅ **References** - Links to papers and further reading

---

## Contributing

Have an idea for a tutorial? See [CONTRIBUTING.md](../../CONTRIBUTING.md)!

We especially welcome tutorials on:
- Domain-specific applications (biology, chemistry, robotics)
- Novel VSA techniques
- Hybrid VSA + ML approaches
- Performance optimization
- Real-world use cases

---

*Last updated: 2025-01-16*
