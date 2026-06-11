# Git hooks

Tracked hooks for this repo. They are **not active until you opt in** —
`core.hooksPath` is local git config and does not travel with a clone.

Enable once per clone:

```sh
git config core.hooksPath .githooks
```

## `pre-commit`

Runs `cargo fmt --all -- --check`, the same gate CI enforces, and rejects
the commit if any file is not rustfmt-clean. This keeps formatting drift
from reaching CI (it had red-failed `main` twice before this hook existed).

Fix a rejection with:

```sh
cargo fmt --all
git add -u
```

Bypass for a single commit with `git commit --no-verify` (discouraged).
