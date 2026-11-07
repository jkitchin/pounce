# JOSS paper

This directory holds the POUNCE paper for the Journal of Open Source
Software: `paper.md` and `paper.bib`.

## Do not merge this branch into `main`

The paper lives on the `joss-paper` branch only. **Do not open a pull
request from `joss-paper` to `main`.** JOSS does not want the paper
merged into the software's main branch. At submission, point JOSS at
this repository and give `joss-paper` as the branch that holds
`paper.md`.

## Building

```sh
make pdf     # JOSS's own build (openjournals/inara, via Docker) -> paper.pdf
make html    # quick preview with local pandoc -> paper.html
make words   # body word count; JOSS asks for 750-1750
make clean
```

`make pdf` needs Docker. The inara image is linux/amd64 only, so on Apple
Silicon it runs under emulation. The build outputs are ignored by git.
