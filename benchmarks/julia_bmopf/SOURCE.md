# BMOPFTools teaching feeder

- Author: Frederik Geth.
- Source: <https://github.com/frederikgeth/BMOPFTools.jl>.
- Pinned commit: `542a48ba7f4c95e50b8acdc790a5323a4ef24930` (package 0.1.0).
- Input: `test/data/case_development/feeder.json` at that commit. Four buses,
  three four-wire lines, and two unbalanced three-phase constant-power loads.
- Attribution/citation: **BMOPFTools fixtures — Frederik Geth**, BMOPFTools.jl,
  case-development teaching feeder, revision above.
- Fixture license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
  granted in upstream `test/data/license.md`; upstream ships the license text.
  Package code is BSD-3-Clause. The harness reads the installed pinned fixture;
  it does not redistribute data or copy upstream implementation code.
- Changes: active and reactive load powers are scaled together by the stated
  scenario factor. The input fixture itself is unchanged. Parameter providers
  insert that factor in the JuMP expressions; fresh controls scale input data.
- This is an original synthetic teaching example, not a calibrated utility
  feeder. Results validate this adapter path, not utility-scale performance.
