PYTHON ?= python3
PDF := Presentation_EUVIP_2026_Hauseux.pdf
export TEXINPUTS := .:theme//:$(TEXINPUTS)

.PHONY: all bootstrap check clean
all: bootstrap
	mkdir -p build
	latexmk -lualatex -pdflualatex='lualatex -no-shell-escape %O %S' -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
	cp build/main.pdf $(PDF)

bootstrap:
	$(PYTHON) bootstrap_assets.py

check: all
	$(PYTHON) verify.py

clean:
	rm -rf build
