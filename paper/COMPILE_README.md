# MLOSS Paper Compilation Guide

## Quick Start

**To compile the MLOSS paper with references:**

```bash
build.bat mloss
```

This will:
- Run the proper compilation sequence (pdflatex → bibtex → pdflatex × 3)
- Generate `vsax_mloss.pdf` with all references properly rendered
- Put auxiliary files in the `build/` directory

## Other Build Options

```bash
# Build just the main VSAX paper
build.bat main

# Build both papers
build.bat all
```

## Fixing References Issue

If references weren't showing up before, it's because LaTeX + BibTeX requires **multiple compilation passes**:

1. First `pdflatex` pass - processes the document, finds citations
2. `bibtex` - generates the bibliography from .bib file
3. Second `pdflatex` pass - includes the bibliography
4. Third `pdflatex` pass - resolves cross-references

The `build.bat` script handles all of this automatically.

## Verifying the Paper

After running `build.bat mloss`, check:

1. ✓ Open `vsax_mloss.pdf`
2. ✓ References section appears at the end
3. ✓ Citations like [1], [2] are properly linked to references
4. ✓ Main content is ≤4 pages (excluding references)
5. ✓ Total is 5-6 pages including references

## Changes Made in Revision

The MLOSS paper was significantly revised to meet JMLR requirements:

### Length Reduction (~60% cut)
**Before:** ~8-10 pages
**After:** ~4 pages main content + 1-2 pages references

Removed:
- Verbose technical sections
- Extensive workflow examples
- Redundant feature descriptions
- Detailed mathematical explanations

### Structural Changes
Following the format of successful MLOSS papers (GraphNeuralNetworks.jl, WEFE, skglm):

- **Introduction** (~1 page): Problem → Solution → Installation → Minimal example
- **Package Design** (~2 pages): Core features, advanced capabilities, architecture, performance
- **Comparison** (~0.5 pages): Table comparing VSAX vs Torchhd/hdlib/PyBHV
- **Software Engineering** (~0.5 pages): Testing, documentation, community
- **Conclusion** (~0.25 pages): Summary + brief future work

### Content Focus
- Shifted from research paper to **software description**
- Emphasized practical usage over theory
- Highlighted unique features vs. competitors
- Focused on software quality (94% coverage, 618 tests, 11 tutorials)
- Made comparison table more prominent (Section 3)

## Expected Page Layout

- **Page 1**: Title, Abstract, Introduction with installation
- **Pages 2-3**: Package design, features, performance benchmarks
- **Page 4**: Comparison table, software engineering, conclusion
- **Pages 5-6**: References

## JMLR MLOSS Requirements

✓ **Page limit**: 4 pages max for main description (references can extend beyond)
✓ **Installation**: Clear installation instructions included
✓ **Comparison**: Feature comparison with existing tools
✓ **Documentation**: Mentioned (11 tutorials, 9 guides)
✓ **Testing**: Covered (618 tests, 94% coverage)
✓ **Examples**: Code examples showing usage
✓ **Availability**: PyPI (`pip install vsax`) and GitHub
✓ **License**: MIT license specified

## Troubleshooting

### "Cannot find jmlr2e.sty"
Download from JMLR: https://jmlr.org/format/jmlr2e.sty
Place in the `paper/` directory

### Build fails with compilation errors
Check `build/vsax_mloss.log` for detailed error messages

### References still not showing
Make sure `vsax_mloss.bib` exists in the `paper/` directory and run `build.bat mloss` again

## Final Submission Checklist

Before submitting to JMLR MLOSS:

- [ ] Paper compiles without errors (`build.bat mloss`)
- [ ] All references appear correctly
- [ ] Main content is ≤4 pages
- [ ] Comparison table is present (Table 1)
- [ ] Installation instructions included
- [ ] GitHub URL is correct: https://github.com/vasanthsarathy/vsax
- [ ] PyPI package name is correct: `pip install vsax`
- [ ] Statistics are accurate:
  - [ ] 94% test coverage
  - [ ] 618 tests
  - [ ] 11 tutorials
  - [ ] 9 user guides
- [ ] License specified (MIT)
