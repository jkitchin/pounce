g3 1 1 0	# problem opf_vv_lower_knee
 24 27 1 0 17 	# vars, constraints, objectives, ranges, eqns
 7 0 6 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 4 0 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 62 1 	# nonzeros in Jacobian, obj. gradient
 13 8	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#balance_p[1]
o16	#-
o16	#-
o2	#*
o2	#*
n0.9149530905953817
v1	#v[2]
o0	#+
o2	#*
n2.941176470588235
o46	#cos
o2	#*
n-1
v3	#theta[2]
o2	#*
n-11.76470588235294
o41	#sin
o2	#*
n-1
v3	#theta[2]
C1	#balance_p[2]
o16	#-
o0	#+
o2	#*
n2.941176470588235
o5	#^
v1	#v[2]
n2
o16	#-
o2	#*
o2	#*
n0.9149530905953817
v1	#v[2]
o0	#+
o2	#*
n2.941176470588235
o46	#cos
v3	#theta[2]
o2	#*
n-11.76470588235294
o41	#sin
v3	#theta[2]
C2	#balance_q[1]
o16	#-
o16	#-
o2	#*
o2	#*
n0.9149530905953817
v1	#v[2]
o0	#+
o2	#*
n2.941176470588235
o41	#sin
o2	#*
n-1
v3	#theta[2]
o2	#*
n11.76470588235294
o46	#cos
o2	#*
n-1
v3	#theta[2]
C3	#balance_q[2]
o16	#-
o0	#+
o2	#*
n11.76470588235294
o5	#^
v1	#v[2]
n2
o16	#-
o2	#*
o2	#*
n0.9149530905953817
v1	#v[2]
o0	#+
o2	#*
n2.941176470588235
o41	#sin
v3	#theta[2]
o2	#*
n11.76470588235294
o46	#cos
v3	#theta[2]
C4	#capability[1]
o0	#+
o5	#^
v0	#p[1]
n2
o5	#^
v2	#q[1]
n2
C5	#rate_from[1]
o0	#+
o5	#^
o0	#+
o16	#-
o2	#*
o2	#*
n0.9149530905953817
v1	#v[2]
o0	#+
o2	#*
n2.941176470588235
o46	#cos
o2	#*
n-1
v3	#theta[2]
o2	#*
n-11.76470588235294
o41	#sin
o2	#*
n-1
v3	#theta[2]
n2.4621739940883547
n2
o5	#^
o0	#+
o16	#-
o2	#*
o2	#*
n0.9149530905953817
v1	#v[2]
o0	#+
o2	#*
n2.941176470588235
o41	#sin
o2	#*
n-1
v3	#theta[2]
o2	#*
n11.76470588235294
o46	#cos
o2	#*
n-1
v3	#theta[2]
n9.848695976353419
n2
C6	#rate_to[1]
o0	#+
o5	#^
o0	#+
o2	#*
n2.941176470588235
o5	#^
v1	#v[2]
n2
o16	#-
o2	#*
o2	#*
n0.9149530905953817
v1	#v[2]
o0	#+
o2	#*
n2.941176470588235
o46	#cos
v3	#theta[2]
o2	#*
n-11.76470588235294
o41	#sin
v3	#theta[2]
n2
o5	#^
o0	#+
o2	#*
n11.76470588235294
o5	#^
v1	#v[2]
n2
o16	#-
o2	#*
o2	#*
n0.9149530905953817
v1	#v[2]
o0	#+
o2	#*
n2.941176470588235
o41	#sin
v3	#theta[2]
o2	#*
n11.76470588235294
o46	#cos
v3	#theta[2]
n2
C7	#hinge[1]
n0
C8	#hinge[2]
n0
C9	#hinge[3]
n0
C10	#hinge[4]
n0
C11	#hinge[5]
n0
C12	#hinge[6]
n0
C13	#volt_var[1]
n0
C14	#volt_watt[1]
n0
C15	#cc[1].c
n0
C16	#cc[1].bc
n0
C17	#cc[2].c
n0
C18	#cc[2].bc
n0
C19	#cc[3].c
n0
C20	#cc[3].bc
n0
C21	#cc[4].c
n0
C22	#cc[4].bc
n0
C23	#cc[5].c
n0
C24	#cc[5].bc
n0
C25	#cc[6].c
n0
C26	#cc[6].bc
n0
O0 0	#obj
n0.0
x18	# initial guess
0 0.4	#p[1]
1 0.9149530905953817	#v[2]
2 0.3	#q[1]
3 0.0	#theta[2]
4 0.0	#pg
5 0.0	#r[1]
6 0.0	#r[2]
7 0.0	#r[3]
8 0.0	#r[4]
9 0.0	#r[5]
10 0.0	#r[6]
11 0.02504690940461829	#s[1]
12 0.06504690940461832	#s[2]
13 0.10504690940461836	#s[3]
14 0.1450469094046184	#s[4]
15 0.1550469094046184	#s[5]
16 0.20504690940461845	#s[6]
17 0.0	#qg
r	#27 ranges (rhs's)
4 2.4621739940883547	#balance_p[1]
4 0.4	#balance_p[2]
4 9.848695976353419	#balance_q[1]
4 0.1	#balance_q[2]
1 1.0	#capability[1]
1 2.25	#rate_from[1]
1 2.25	#rate_to[1]
4 -0.94	#hinge[1]
4 -0.98	#hinge[2]
4 -1.02	#hinge[3]
4 -1.06	#hinge[4]
4 -1.07	#hinge[5]
4 -1.12	#hinge[6]
4 0.3	#volt_var[1]
1 1.0	#volt_watt[1]
5 1 12	#cc[1].c
4 0	#cc[1].bc
5 1 13	#cc[2].c
4 0	#cc[2].bc
5 1 14	#cc[3].c
4 0	#cc[3].bc
5 1 15	#cc[4].c
4 0	#cc[4].bc
5 1 16	#cc[5].c
4 0	#cc[5].bc
5 1 17	#cc[6].c
4 0	#cc[6].bc
b	#24 bounds (on variables)
0 0.0 0.8	#p[1]
0 0.88 1.18	#v[2]
0 -0.4 0.4	#q[1]
0 -0.3 0.3	#theta[2]
0 -2.0 2.0	#pg
2 0.0	#r[1]
2 0.0	#r[2]
2 0.0	#r[3]
2 0.0	#r[4]
2 0.0	#r[5]
2 0.0	#r[6]
2 0.0	#s[1]
2 0.0	#s[2]
2 0.0	#s[3]
2 0.0	#s[4]
2 0.0	#s[5]
2 0.0	#s[6]
0 -2.0 2.0	#qg
3	#cc[1].bv
3	#cc[2].bv
3	#cc[3].bv
3	#cc[4].bv
3	#cc[5].bv
3	#cc[6].bv
k23	#intermediate Jacobian column lengths
3
15
18
24
25
28
31
34
37
40
43
44
45
46
47
48
49
50
52
54
56
58
60
J0 3	#balance_p[1]
1 0
3 0
4 1
J1 3	#balance_p[2]
0 1
1 0
3 0
J2 3	#balance_q[1]
1 0
3 0
17 1
J3 3	#balance_q[2]
1 0
2 1
3 0
J4 2	#capability[1]
0 0
2 0
J5 2	#rate_from[1]
1 0
3 0
J6 2	#rate_to[1]
1 0
3 0
J7 3	#hinge[1]
1 -1
5 1
11 -1
J8 3	#hinge[2]
1 -1
6 1
12 -1
J9 3	#hinge[3]
1 -1
7 1
13 -1
J10 3	#hinge[4]
1 -1
8 1
14 -1
J11 3	#hinge[5]
1 -1
9 1
15 -1
J12 3	#hinge[6]
1 -1
10 1
16 -1
J13 5	#volt_var[1]
2 1
5 7.499999999999993
6 -7.499999999999993
7 7.499999999999993
8 -7.499999999999993
J14 3	#volt_watt[1]
0 1
9 15.999999999999988
10 -15.999999999999988
J15 1	#cc[1].c
18 1
J16 2	#cc[1].bc
5 -1
18 1
J17 1	#cc[2].c
19 1
J18 2	#cc[2].bc
6 -1
19 1
J19 1	#cc[3].c
20 1
J20 2	#cc[3].bc
7 -1
20 1
J21 1	#cc[4].c
21 1
J22 2	#cc[4].bc
8 -1
21 1
J23 1	#cc[5].c
22 1
J24 2	#cc[5].bc
9 -1
22 1
J25 1	#cc[6].c
23 1
J26 2	#cc[6].bc
10 -1
23 1
G0 1	#obj
4 1
