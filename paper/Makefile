# Makefile for VSAX paper

PAPER = vsax_paper
TEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean view

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex
	$(TEX) $(PAPER).tex
	$(TEX) $(PAPER).tex
	@echo "Paper compiled successfully!"

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz
	rm -f *.fdb_latexmk *.fls *.lof *.lot
	@echo "Cleaned auxiliary files"

distclean: clean
	rm -f $(PAPER).pdf
	@echo "Cleaned all generated files"

view: $(PAPER).pdf
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(PAPER).pdf; \
	elif command -v open > /dev/null; then \
		open $(PAPER).pdf; \
	elif command -v start > /dev/null; then \
		start $(PAPER).pdf; \
	else \
		echo "Please open $(PAPER).pdf manually"; \
	fi
