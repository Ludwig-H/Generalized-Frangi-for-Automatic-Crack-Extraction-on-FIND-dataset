LATEXCMD = lualatex
LATEXFLAGS = -file-line-error -halt-on-error -interaction=nonstopmode

.PHONY: all figures clean

all: main.pdf

figures:
	python tools/make_figures.py

main.pdf: main.tex figures
	TEXINPUTS=.:theme//: $(LATEXCMD) $(LATEXFLAGS) main.tex
	TEXINPUTS=.:theme//: $(LATEXCMD) $(LATEXFLAGS) main.tex

clean:
	rm -f main.aux main.log main.nav main.out main.snm main.toc main.fls main.fdb_latexmk
