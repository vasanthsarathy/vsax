# VSAX Academic Paper

This directory contains the academic paper introducing VSAX: "VSAX: A GPU-Accelerated Vector Symbolic Algebra Library for JAX"

## Files

- `vsax_paper.tex` - Main LaTeX source file
- `vsax_paper.pdf` - Compiled PDF (after compilation)

## Compiling

### On Windows (easiest)

Simply run the build script:

```cmd
build.bat
```

This will:
- Compile the paper using latexmk
- Put all auxiliary files in `build/` directory
- Copy the final PDF to `vsax_paper.pdf`

To clean up:
```cmd
clean.bat
```

### Manual Compilation

#### Using latexmk (recommended)

```bash
# Compile with build directory
latexmk -pdf -output-directory=build vsax_paper.tex

# Copy PDF to main directory
cp build/vsax_paper.pdf .
```

#### Using pdflatex

```bash
pdflatex vsax_paper.tex
pdflatex vsax_paper.tex  # Run twice for references
```

### Using Overleaf

Upload `vsax_paper.tex` to [Overleaf](https://www.overleaf.com) for online compilation.

## Submission

This paper is suitable for:

1. **arXiv** - Submit to cs.AI, cs.LG, or cs.SE categories
2. **Conferences** - NeurIPS, ICML, ICLR (workshop track or main)
3. **Journals** - ACM Transactions on Intelligent Systems, Neural Computation
4. **Workshops** - VSA/HDC workshops at major ML conferences

## Citation

Once published on arXiv, others can cite this paper as:

```bibtex
@article{sarathy2025vsax,
  title={VSAX: A GPU-Accelerated Vector Symbolic Algebra Library for JAX},
  author={Sarathy, Vasanth},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

The paper content follows academic fair use. The VSAX library itself is MIT licensed.
