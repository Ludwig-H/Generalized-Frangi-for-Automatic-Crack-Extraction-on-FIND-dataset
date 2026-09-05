LATEX = xelatex -interaction=nonstopmode -halt-on-error -file-line-error
FIGURES = frangi-hessian frangi-terms foundation-hierarchy

.PHONY: all figures clean
all: figures
	mkdir -p build
	$(LATEX) -output-directory=build poster.tex
	$(LATEX) -output-directory=build poster.tex
	cp build/poster.pdf Poster_EUVIP_2026_Hauseux_A0.pdf

figures:
	mkdir -p build
	@for figure in $(FIGURES); do \
	  $(LATEX) -output-directory=build figures/$$figure.tex > build/$$figure.stdout || \
	    { cat build/$$figure.stdout; exit 1; }; \
	  cp build/$$figure.pdf figures/$$figure.pdf; \
	done

clean:
	rm -rf build
