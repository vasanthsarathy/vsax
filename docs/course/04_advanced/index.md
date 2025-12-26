# Module 4: Advanced Techniques

**Duration:** ~4 hours | **Difficulty:** Advanced

## Overview

Module 4 introduces cutting-edge VSA techniques that enable sophisticated reasoning, spatial intelligence, and multi-modal AI systems. You'll learn how to build systems that combine continuous spatial representations, hierarchical structures, and heterogeneous data fusion.

## What You'll Learn

- **Clifford Operators:** Exact, invertible transformations for directional relations
- **Spatial Semantic Pointers:** Encode continuous space without discretization
- **Hierarchical Structures:** Recursive role-filler binding for trees and nested data
- **Resonator Networks:** Convergent factorization for decoding complex bindings
- **Multi-Modal Integration:** Fuse vision, language, and symbols in unified VSA space
- **Neural-Symbolic AI:** Combine neural networks with symbolic VSA reasoning (HD-Glue)

## Lessons

### [4.1: Clifford Operators](01_operators.md)
**Time:** 45 minutes

Learn how operators enable exact, invertible transformations for spatial and semantic relations where standard binding fails.

**Key concepts:** Operators vs hypervectors, phase-based transformations, exact inversion (>0.999 similarity)

---

### [4.2: Spatial Semantic Pointers](02_ssp.md)
**Time:** 50 minutes

Encode continuous spatial coordinates using Fractional Power Encoding for smooth, queryable spatial representations.

**Key concepts:** SSP = \(X^x \otimes Y^y\), continuous spatial encoding, "what/where" queries

---

### [4.3: Hierarchical Structures & Resonators](03_hierarchical.md)
**Time:** 60 minutes

Encode tree structures with recursive binding and decode using resonator networks for convergent factorization.

**Key concepts:** Recursive role-filler binding, resonator algorithm, parsing trees and nested data

---

### [4.4: Multi-Modal & Neural-Symbolic Integration](04_multimodal.md)
**Time:** 55 minutes

Fuse heterogeneous data (vision + language + symbols) and combine neural networks with VSA symbolically.

**Key concepts:** Cross-modal grounding, HD-Glue, ensemble learning, neuro-symbolic AI

---

## Hands-On Exercises

**Exercise 1:** [Spatial Reasoning System](exercises/01_spatial_reasoning.py)
- Combine SSP + Clifford Operators for complete spatial reasoning
- Build 2D scene encoder with locations and directional relations

**Exercise 2:** [Tree Decoder](exercises/02_tree_decoder.py)
- Encode/decode expression trees, JSON, family trees
- Use resonators for factorization of complex bindings

**Capstone:** [Multi-Modal Reasoning System](exercises/capstone_multimodal_reasoning.py)
- Integrate ALL Module 4 techniques
- Build unified knowledge base with spatial, relational, hierarchical, and attribute representations

---

## Prerequisites

✅ Module 1: Foundational concepts (binding, bundling, three models)
✅ Module 2: FHRR operations (exact unbinding needed for operators)
✅ Module 3: Encoders (FPE, DictEncoder used extensively)

---

## Learning Outcomes

After Module 4, you will be able to:

- [ ] Create Clifford Operators for exact relational encoding
- [ ] Use SSP for continuous 2D/3D spatial reasoning
- [ ] Encode hierarchical trees with recursive binding
- [ ] Decode complex bindings using resonator networks
- [ ] Fuse multiple modalities in unified VSA space
- [ ] Integrate neural embeddings with symbolic VSA

---

**Previous:** [Module 3: Encoders & Applications](../03_encoders/index.md)
**Next:** [Lesson 4.1: Clifford Operators](01_operators.md) or [Module 5: Research & Extensions](../05_research/index.md)
