LATEX = xelatex -interaction=nonstopmode -halt-on-error -file-line-error
BIBTEX = bibtex

.PHONY: all clean
all:
	mkdir -p build
	$(LATEX) -output-directory=build poster.tex
	BIBINPUTS=.: BSTINPUTS=.: $(BIBTEX) build/poster
	$(LATEX) -output-directory=build poster.tex
	$(LATEX) -output-directory=build poster.tex
	cp build/poster.pdf Poster_EUVIP_2026_Hauseux_A0.pdf

clean:
	rm -rf build
