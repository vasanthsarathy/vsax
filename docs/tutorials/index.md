# VSAX Tutorials

Hands-on tutorials demonstrating VSAX features with real datasets and practical examples.

## Available Tutorials

### Tutorial 1: MNIST Digit Classification
**Level**: Beginner
**Topics**: Image encoding, prototype learning, similarity-based classification
**Dataset**: MNIST digits (sklearn)

Learn how to use VSA for image classification with the classic MNIST dataset. Compare different VSA models (FHRR, MAP, Binary) and achieve 95%+ accuracy using simple prototype matching.

[📖 Read Tutorial](01_mnist_classification.md) | [📓 Open Notebook](../../examples/notebooks/tutorial_01_mnist_classification.ipynb)

---

### Tutorial 2: Knowledge Graph Reasoning
**Level**: Intermediate
**Topics**: Graph encoding, factorization, multi-hop reasoning
**Dataset**: Custom animal taxonomy

Build and query a knowledge graph using VSA. Encode relational facts (triples), perform queries using unbinding, use resonator networks to decode compositional structures, and perform multi-hop reasoning for property inheritance.

[📖 Read Tutorial](02_knowledge_graph.md) | [📓 Open Notebook](../../examples/notebooks/tutorial_02_knowledge_graph.ipynb)

---

### Tutorial 3: Analogical Reasoning - Kanerva's "Dollar of Mexico"
**Level**: Advanced
**Topics**: Holistic encoding, mapping vectors, prototypes, analogical reasoning
**Dataset**: Countries with structured attributes

Implement the classic examples from Pentti Kanerva's foundational paper on hyperdimensional computing. Learn to encode structured records holistically, compute mapping vectors from examples, perform analogical queries like "What's the dollar of Mexico?", solve IQ-test analogies, and chain mappings transitively.

[📖 Read Tutorial](03_kanerva_analogies.md) | [📓 Open Notebook](../../examples/notebooks/tutorial_03_kanerva_analogies.ipynb)

---

### Tutorial 4: Word Analogies & Random Indexing
**Level**: Intermediate
**Topics**: Word embeddings, semantic similarity, Random Indexing, word analogies
**Dataset**: Custom text corpus with semantic relationships

Build word embeddings using Random Indexing (Kanerva et al. 2000) and perform classic word analogies like "king - man + woman = queen". Learn how context co-occurrence shapes meaning, perform semantic similarity search, compare VSA models for NLP tasks, and understand vector composition for analogical reasoning.

[📖 Read Tutorial](04_word_analogies.md) | [📓 Open Notebook](../../examples/notebooks/tutorial_04_word_analogies.ipynb)

---

### Tutorial 5: Understanding VSA Models - Comparative Analysis
**Level**: Intermediate
**Topics**: Model comparison, FHRR vs MAP vs Binary, performance benchmarking, trade-offs
**Dataset**: Iris classification dataset

Compare all three VSA models (FHRR, MAP, Binary) across classification accuracy, noise robustness, capacity analysis, and speed benchmarks. Learn when to use each model, understand the trade-offs between accuracy, speed, and memory, and get a practical decision guide for choosing the right model for your task.

[📖 Read Tutorial](05_model_comparison.md) | [📓 Open Notebook](../../examples/notebooks/tutorial_05_model_comparison.ipynb)

---

### Tutorial 6: VSA for Edge Computing - Lightweight Alternative to Neural Networks
**Level**: Intermediate
**Topics**: Edge computing, VSA vs neural networks, efficiency, deployment, resource constraints
**Dataset**: Fashion-MNIST

Compare VSA with neural networks on model size, training time, inference speed, and accuracy. Discover VSA's advantages for edge computing: 4-10x faster training, similar model size, and comparable accuracy without gradient descent. Perfect for IoT, wearables, and embedded systems where resources are limited.

[📖 Read Tutorial](06_edge_computing.md) | [📓 Open Notebook](../../examples/notebooks/tutorial_06_edge_computing.ipynb)

---

### Tutorial 7: Hierarchical Structures - Trees & Nested Composition
**Level**: Advanced
**Topics**: Recursive binding, tree encoding, parse trees, compositional semantics, resonator networks
**Examples**: Arithmetic expressions, nested lists, syntax trees, family trees

Encode hierarchical structures through recursive role-filler binding. Learn to represent trees holistically in single vectors, decode nested structures with exact unbinding, and use resonator networks for robust factorization. Demonstrates VSA's powerful compositional capabilities for representing syntax trees, nested data, and genealogy.

[📖 Read Tutorial](07_hierarchical_structures.md) | [📓 Open Notebook](../../examples/notebooks/tutorial_07_hierarchical_structures.ipynb)

---

### Tutorial 8: Multi-Modal Concept Grounding with MNIST
**Level**: Advanced
**Topics**: Multi-modal fusion, heterogeneous binding, cross-modal queries, online learning
**Dataset**: MNIST digits + arithmetic facts

Demonstrate VSA's powerful multi-modal capabilities by fusing vision (MNIST images), symbolic atoms, and arithmetic relationships into rich concept representations. Learn to encode heterogeneous data (images, symbols, operations) in the same space, perform cross-modal queries ("What is 1+2?" → retrieve visual prototype of 3), and add new knowledge online without retraining. Shows VSA's unique advantage: concepts defined by multiple modalities and their relationships.

[📖 Read Tutorial](08_multimodal_grounding.md) | [📓 Open Notebook](../../examples/notebooks/tutorial_08_multimodal_grounding.ipynb)

---

## Tutorial Format

Each tutorial is available in two formats:

1. **Jupyter Notebook** (`.ipynb`) - Interactive, runnable code with visualizations
   - Located in `examples/notebooks/`
   - Can be run locally or in Google Colab
   - Includes plots and interactive exploration

2. **Documentation** (`.md`) - Readable reference with complete code
   - Embedded in this documentation site
   - Easy to copy-paste code snippets
   - Includes all outputs and explanations

## Running Tutorials

### Prerequisites

```bash
# Install VSAX
pip install vsax

# Install tutorial dependencies
pip install scikit-learn matplotlib seaborn jupyter
```

### Option 1: Run Jupyter Notebooks Locally

```bash
# Clone the repository
git clone https://github.com/vasanthsarathy/vsax.git
cd vsax

# Install dependencies
pip install -e ".[dev]"
pip install jupyter scikit-learn matplotlib seaborn

# Launch Jupyter
jupyter notebook examples/notebooks/
```

### Option 2: Read in Documentation

Simply navigate to the tutorial pages in this documentation and copy the code snippets directly.

## Tutorial Structure

Each tutorial follows this structure:

1. **Introduction** - What you'll learn and why it matters
2. **Setup** - Imports and data loading
3. **Step-by-step Implementation** - Detailed walkthrough with code
4. **Evaluation** - Results and performance analysis
5. **Comparison** - Different approaches or models
6. **Key Takeaways** - Summary and lessons learned
7. **Next Steps** - Extensions and related tutorials

## Feedback and Contributions

Found an issue or have a suggestion for a new tutorial? Please [open an issue](https://github.com/vasanthsarathy/vsax/issues) on GitHub.

Want to contribute a tutorial? See our [Contributing Guide](https://github.com/vasanthsarathy/vsax/blob/main/CONTRIBUTING.md).
