# Build the JOSS paper.
#
#   make pdf     JOSS's own build (openjournals/inara, via Docker) -> paper.pdf
#   make html    quick preview with local pandoc, no JOSS layout -> paper.html
#   make words   body word count (JOSS asks for 750-1750)
#   make clean
#
# The inara image is published for linux/amd64 only; on Apple Silicon
# Docker runs it under emulation. Override with DOCKER_PLATFORM= if needed.

DOCKER          ?= docker
DOCKER_PLATFORM ?= linux/amd64
INARA_IMAGE     ?= openjournals/inara:latest
PANDOC          ?= pandoc

.PHONY: pdf html words clean

pdf: paper.pdf

paper.pdf: paper.md paper.bib
	$(DOCKER) run --rm --platform $(DOCKER_PLATFORM) \
	  -v "$(CURDIR)":/data -u $$(id -u):$$(id -g) \
	  --env JOURNAL=joss $(INARA_IMAGE) -o pdf -p paper.md

html: paper.html

paper.html: paper.md paper.bib
	$(PANDOC) paper.md --citeproc --standalone --metadata link-citations=true -o $@

words:
	@sed -n '/^# Summary/,/^# References/p' paper.md | grep -v '^#' | wc -w

clean:
	rm -f paper.pdf paper.html paper.jats paper.crossref paper.preprint.tex
	rm -rf jats
