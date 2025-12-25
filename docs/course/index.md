# VSAX Course: From Foundations to Research

Welcome to the VSAX comprehensive course! This progressive learning path will take you from zero VSA knowledge to advanced research capabilities.

## What You'll Learn

This course teaches **Vector Symbolic Architectures** (VSAs), also known as Hyperdimensional Computing (HDC), through a unique combination of mathematical foundations and hands-on implementation.

By the end of this course, you will:

- ✅ Understand why high-dimensional vectors enable symbolic computation
- ✅ Master the three VSA models (FHRR, MAP, Binary) and when to use each
- ✅ Build encoders for any data type (images, graphs, sequences, continuous spaces)
- ✅ Implement advanced techniques (operators, resonators, spatial encoding)
- ✅ Debug common VSA issues and optimize performance
- ✅ Design research extensions and contribute to VSAX

## Course Structure

The course consists of **5 modules**, **20 lessons**, and takes approximately **12-20 hours** to complete.

### Module Breakdown

| Module | Topics | Level | Duration |
|--------|--------|-------|----------|
| **[Module 1](01_foundations/index.md)** | Foundations | Beginner | 3-4 hours |
| **[Module 2](02_operations/index.md)** | Core Operations | Beginner-Intermediate | 4-5 hours |
| **[Module 3](03_encoders/index.md)** | Encoders & Applications | Intermediate | 6-8 hours |
| **[Module 4](04_advanced/index.md)** | Advanced Techniques | Advanced | 6-8 hours |
| **[Module 5](05_research/index.md)** | Research & Extensions | Research | 3-4 hours |

### Detailed Module Overview

#### Module 1: Foundations
**Why high dimensions work | Binding & bundling | Three VSA models | First program**

Start here if you're new to VSA. Learn the mathematical intuitions behind hyperdimensional computing and write your first VSAX program.

[Start Module 1 →](01_foundations/index.md)

#### Module 2: Core Operations
**FHRR mathematics | MAP & Binary operations | Similarity metrics | Model selection**

Deep dive into the three VSA models. Understand the mathematical foundations, implementation details, and decision frameworks for choosing the right model.

[Start Module 2 →](02_operations/index.md)

#### Module 3: Encoders & Applications
**Scalar encoding | Dictionaries & sets | Image classification | Knowledge graphs | Analogies**

Learn encoding strategies for different data types and build real-world applications: classifiers, knowledge bases, and analogical reasoning systems.

[Start Module 3 →](03_encoders/index.md)

#### Module 4: Advanced Techniques
**Clifford operators | Spatial semantic pointers | Hierarchical structures | Multi-modal fusion**

Master advanced VSA techniques for complex reasoning: exact transformations, continuous spatial encoding, tree structures, and neural-symbolic integration.

[Start Module 4 →](04_advanced/index.md)

#### Module 5: Research & Extensions
**Vector function architecture | Custom encoders | Research frontiers**

Prepare for VSA research. Learn VFA, design custom encoders, explore open problems, and contribute to VSAX development.

[Start Module 5 →](05_research/index.md)

---

## Learning Paths

Choose a learning path based on your background and goals:

### Path 1: Full Course (Beginners)
**Recommended for:** Complete beginners to VSA

**Path:** Module 1 → Module 2 → Module 3 → Module 4 → Module 5

**Duration:** ~20 hours

**You'll learn:** Complete VSA foundations, all three models, encoding strategies, advanced techniques, and research preparation

[Start with Module 1 →](01_foundations/index.md)

---

### Path 2: Application-Focused (ML Engineers)
**Recommended for:** ML practitioners who want to build VSA applications

**Path:** Module 1 (skim 1.1-1.2) → Module 3 → Module 4 (selected topics)

**Duration:** ~10 hours

**You'll learn:** Practical encoding strategies, image classification, knowledge graphs, multi-modal systems

**Skip:** Deep mathematical foundations, model comparison details

[Start with Module 1.3 →](01_foundations/03_models.md)

---

### Path 3: Research-Focused (PhD Students)
**Recommended for:** Researchers exploring VSA for their work

**Path:** Module 1 → Module 2 → Module 4 → Module 5

**Duration:** ~15 hours

**You'll learn:** Mathematical foundations, model selection, advanced techniques (operators, SSP, VFA), research frontiers

**Skip:** Basic application tutorials (can revisit as needed)

[Start with Module 1 →](01_foundations/index.md)

---

### Path 4: Quick Start (Developers)
**Recommended for:** Developers who need to use VSAX immediately

**Path:** [Getting Started](../getting-started.md) → Module 3 → [Tutorials](../tutorials/index.md) (as needed)

**Duration:** ~6 hours

**You'll learn:** How to use VSAX API, encoding strategies, specific application recipes

**Skip:** Mathematical foundations (can revisit later)

[Start with Getting Started →](../getting-started.md)

---

## Prerequisites

### Required
- **Python proficiency**: Comfortable with Python syntax, functions, classes
- **NumPy basics**: Understand arrays, array operations, broadcasting
- **Basic linear algebra**: Vectors, dot products, norms

### Recommended (Not Required)
- **JAX familiarity**: Helps but not necessary (we'll teach JAX patterns)
- **Machine learning basics**: Helpful for understanding applications
- **Complex numbers**: Useful for FHRR model (we'll review when needed)

### Installation

Before starting, ensure VSAX is installed:

```bash
pip install vsax
```

Or for development:

```bash
git clone https://github.com/vasanthsarathy/vsax.git
cd vsax
pip install -e ".[dev]"
```

---

## How to Use This Course

### Progressive Learning
This course is designed for **progressive learning**. Each lesson builds on previous concepts. We recommend:

1. **Follow in order** within modules
2. **Complete exercises** before moving forward
3. **Check self-assessments** to ensure understanding
4. **Build capstone projects** at module end

### Exercises and Assessments

Each lesson includes:

- **Self-Assessment Checklist**: "I can..." statements to verify understanding
- **Quick Quiz**: 3-5 conceptual questions
- **Hands-On Exercise**: Coding problems with solutions

Each module includes:

- **Capstone Project**: Larger application combining module concepts

### Code Examples

All code examples are:

- ✅ Copy-paste ready (runnable as-is)
- ✅ Tested with latest VSAX version
- ✅ Commented with explanations
- ✅ Available in `/exercises/` directories

### Getting Help

**Stuck on a concept?**

- Check [Troubleshooting](troubleshooting.md) for common issues
- Review previous lessons for foundations
- Try the exercises (learning by doing!)
- Ask questions on [GitHub Discussions](https://github.com/vasanthsarathy/vsax/discussions)

**Found an error?**

- Open an issue on [GitHub](https://github.com/vasanthsarathy/vsax/issues)
- Or submit a pull request with fixes!

---

## Course vs Tutorials vs User Guide

**Confused about where to start?** Here's how the documentation is organized:

| Section | Purpose | When to Use |
|---------|---------|-------------|
| **Course** (you are here) | Progressive learning with theory + practice | Learning VSA from scratch |
| **[Tutorials](../tutorials/index.md)** | Cookbook recipes for specific tasks | Need to solve a specific problem |
| **[User Guide](../guide/models.md)** | Feature reference documentation | Looking up API details |
| **[Getting Started](../getting-started.md)** | Quick introduction | Want to try VSAX right now |

---

## Time Commitment

### Full Course
- **Self-paced**: 2-4 weeks at 1 hour/day
- **Intensive**: 1 week full-time
- **Casual**: 4-8 weeks at 2-3 hours/week

### Individual Modules
- **Module 1**: One weekend (3-4 hours)
- **Module 2**: One weekend (4-5 hours)
- **Module 3**: Two weekends (6-8 hours)
- **Module 4**: Two weekends (6-8 hours)
- **Module 5**: One weekend (3-4 hours)

---

## What Makes This Course Unique?

### 1. Theory + Practice Together
Every mathematical concept is immediately followed by JAX/VSAX code. You'll understand both the "why" and the "how".

### 2. Multiple Learning Paths
Not everyone learns the same way. Choose a path that matches your background and goals.

### 3. Hands-On Throughout
No passive reading. Every lesson has exercises. Learning VSA requires building intuitions through coding.

### 4. Reuses Best Content
60% of content links to existing excellent tutorials and guides. 40% is new foundational material filling critical gaps.

### 5. Self-Paced with Scaffolding
Self-assessments help you know when you're ready to proceed. No instructor required.

### 6. Research-Ready
Module 5 prepares you for advanced VSA research and contributing to VSAX.

---

## Ready to Start?

### Beginners: Start with Module 1
Learn why high-dimensional vectors enable symbolic computation.

[Module 1: Foundations →](01_foundations/index.md)

### Experienced ML practitioners: Jump to Module 3
Learn encoding strategies and build applications.

[Module 3: Encoders & Applications →](03_encoders/index.md)

### Researchers: Start with Module 1, Focus on 2, 4, 5
Deep mathematical foundations and advanced techniques.

[Module 1: Foundations →](01_foundations/index.md)

---

## Feedback and Contributions

This course is a living document. Help us improve it!

- **Found a typo?** Submit a PR
- **Have a suggestion?** Open an issue
- **Built something cool?** Share in Discussions
- **Want to contribute a lesson?** We welcome contributions!

**Repository**: [github.com/vasanthsarathy/vsax](https://github.com/vasanthsarathy/vsax)

---

## Course Completion

When you complete all 5 modules and capstone projects:

- [ ] You can explain why high dimensions enable symbolic computation
- [ ] You can choose the appropriate VSA model for any task
- [ ] You can build encoders for custom data types
- [ ] You can debug common VSA issues (low similarity, capacity limits)
- [ ] You can implement advanced techniques (operators, resonators, SSP)
- [ ] You can propose research extensions to VSAX

**Congratulations!** You're now a VSA expert ready to build advanced cognitive architectures.

---

**Ready?** Let's begin your VSA journey.

[Start Module 1: Foundations →](01_foundations/index.md)
