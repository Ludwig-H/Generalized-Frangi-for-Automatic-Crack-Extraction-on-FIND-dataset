LATEX = lualatex -interaction=nonstopmode -halt-on-error -file-line-error
PDF = Cerema_2026-09-10_Hauseux_SAM_Hierarchie.pdf

.PHONY: all clean
all:
	mkdir -p build
	TEXINPUTS=.:theme//: $(LATEX) -output-directory=build main.tex
	BIBINPUTS=.: BSTINPUTS=.: bibtex build/main
	TEXINPUTS=.:theme//: $(LATEX) -output-directory=build main.tex
	TEXINPUTS=.:theme//: $(LATEX) -output-directory=build main.tex
	cp build/main.pdf $(PDF)

clean:
	rm -rf build
