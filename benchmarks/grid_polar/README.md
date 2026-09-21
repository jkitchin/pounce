# `grid_polar` — the polar control for the ECF comparison

This directory holds results only. Its `.nl` are emitted by
`benchmarks/grid_ecf/ecf_nl_export.py --form polar`, and everything about the
suite — what it is, why it exists separately from `benchmarks/grid/`, the
conventions it is built under, and how to run it — is documented in
[`../grid_ecf/README.md`](../grid_ecf/README.md).

Do not regenerate this set on its own. The comparison is only valid when both
halves come from the same generator invocation with the same conventions:

```bash
make -C benchmarks grid-ecf-generate    # writes BOTH .nl sets
```
