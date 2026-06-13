# POUNCE-QP vs Clarabel vs POUNCE-NLP — Maros-Meszaros QP benchmark

138 problems; 138 with ground-truth optima (DOC 97/6, BPMPD reference). A solve is **correct** when `|obj-opt| <= 1e-05 + 0.0001·max(|obj|,|opt|)`.

### pounce QP-IPM (solver_selection=qp-ipm)
- Solved (own status): **137/138**
- Correct vs ground truth: **137/138**
- Solved-but-wrong (status OK, obj off): **0**
- Median rel-err on correct solves: 1.4e-09
### Clarabel
- Solved (own status): **133/138**
- Correct vs ground truth: **124/138**
- Solved-but-wrong (status OK, obj off): **9**
- Median rel-err on correct solves: 2.3e-09
- Wrong objectives: YAO(re=5.4e-01), UBH1(re=5.6e-04), LISWET7(re=9.2e-01), LISWET10(re=4.9e-01), LISWET1(re=2.7e-01), LISWET8(re=9.2e-01), LISWET11(re=2.6e-01), LISWET12(re=8.6e-01), LISWET9(re=7.2e-01)
### pounce NLP (solver_selection=nlp)
- Solved (own status): **137/138**
- Correct vs ground truth: **129/138**
- Solved-but-wrong (status OK, obj off): **8**
- Median rel-err on correct solves: 1.1e-08
- Wrong objectives: YAO(re=7.7e-03), LISWET7(re=2.2e-01), LISWET10(re=2.1e-01), LISWET1(re=2.5e-01), LISWET8(re=8.9e-02), LISWET11(re=6.0e-02), LISWET12(re=3.7e-02), LISWET9(re=3.3e-02)
### Speed (geomean over 123 all-three-correct problems)
- pounce QP-IPM : 0.041s
- Clarabel      : 0.008s
- pounce NLP    : 0.061s
- QP-IPM vs Clarabel: 5.22×  (Clarabel faster)
- QP-IPM vs NLP     : 1.49×  (QP-IPM faster)

### Problems where pounce-QP is correct but another solver is not

| problem | opt | clarabel | nlp |
|---|---|---|---|
| PRIMALC2 | -3551.31 | DualInfeasible | ✓ |
| PRIMALC1 | -6155.25 | DualInfeasible | ✓ |
| QISRAEL | 2.53478e+07 | InsufficientProgress | ✓ |
| YAO | 197.704 | off re=5.4e-01 | off re=7.7e-03 |
| POWELL20 | 5.20896e+10 | PrimalInfeasible | ✓ |
| UBH1 | 1.116 | off re=5.6e-04 | ✓ |
| LISWET7 | 498.841 | off re=9.2e-01 | off re=2.2e-01 |
| LISWET10 | 49.4858 | off re=4.9e-01 | off re=2.1e-01 |
| LISWET1 | 36.1224 | off re=2.7e-01 | off re=2.5e-01 |
| LISWET8 | 714.47 | off re=9.2e-01 | off re=8.9e-02 |
| LISWET11 | 49.524 | off re=2.6e-01 | off re=6.0e-02 |
| LISWET12 | 1736.93 | off re=8.6e-01 | off re=3.7e-02 |
| LISWET9 | 1963.25 | off re=7.2e-01 | off re=3.3e-02 |
| BOYD2 | 21.2568 | MaxIterations | TimeOut |

