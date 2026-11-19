g3 1 1 0	# problem unknown
 31 21 1 0 21 	# vars, constraints, objectives, ranges, eqns
 21 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 7 8 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 70 1 	# nonzeros in Jacobian, obj. gradient
 5 2	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#rdef2
o2	#*
n1250
o46	#cos
v0	#x1
C1	#qdef2
o2	#*
n15625.0
o41	#sin
v0	#x1
C2	#sdef2
o2	#*
n1250
o41	#sin
v0	#x1
C3	#rdef3
o2	#*
n1250
o46	#cos
v1	#x2
C4	#qdef3
o2	#*
n15625.0
o41	#sin
v1	#x2
C5	#sdef3
o2	#*
n1250
o41	#sin
v1	#x2
C6	#rdef4
o2	#*
n3750
o46	#cos
v2	#x3
C7	#qdef4
o2	#*
n93750.0
o41	#sin
v2	#x3
C8	#sdef4
o2	#*
n3750
o41	#sin
v2	#x3
C9	#rdef5
o2	#*
n3750
o46	#cos
v3	#x4
C10	#qdef5
o2	#*
n93750.0
o41	#sin
v3	#x4
C11	#sdef5
o2	#*
n3750
o41	#sin
v3	#x4
C12	#rdef6
o2	#*
n3750
o46	#cos
v4	#x5
C13	#qdef6
o2	#*
n93750.0
o41	#sin
v4	#x5
C14	#sdef6
o2	#*
n3750
o41	#sin
v4	#x5
C15	#rdef7
o2	#*
n9000
o46	#cos
v5	#x6
C16	#qdef7
o2	#*
n405000.0
o41	#sin
v5	#x6
C17	#sdef7
o2	#*
n9000
o41	#sin
v5	#x6
C18	#rdef8
o2	#*
n9000
o46	#cos
v6	#x7
C19	#qdef8
o2	#*
n405000.0
o41	#sin
v6	#x7
C20	#sdef8
o2	#*
n9000
o41	#sin
v6	#x7
O0 0	#obj
o2	#*
o2	#*
n-1
v7	#r8
v7	#r8
x7	# initial guess
0 0.5	#x1
1 0.5	#x2
2 0.5	#x3
3 0.5	#x4
4 0.5	#x5
5 0.5	#x6
6 0.5	#x7
r	#21 ranges (rhs's)
4 0	#rdef2
4 10000.0	#qdef2
4 800.0	#sdef2
4 0	#rdef3
4 10000.0	#qdef3
4 800.0	#sdef3
4 0	#rdef4
4 40000.0	#qdef4
4 1600.0	#sdef4
4 0	#rdef5
4 40000.0	#qdef5
4 1600.0	#sdef5
4 0	#rdef6
4 40000.0	#qdef6
4 1600.0	#sdef6
4 0	#rdef7
4 129600.0	#qdef7
4 2880.0	#sdef7
4 0	#rdef8
4 100000.0	#qdef8
4 1000.0	#sdef8
b	#31 bounds (on variables)
0 0 1.58	#x1
0 0 1.58	#x2
0 0 1.58	#x3
0 0 1.58	#x4
0 0 1.58	#x5
0 0 1.58	#x6
0 0 1.58	#x7
3	#r8
4 0	#r1
4 0	#q1
4 0	#s1
3	#r2
3	#q2
3	#s2
3	#r3
3	#q3
3	#s3
3	#r4
3	#q4
3	#s4
3	#r5
3	#q5
3	#s5
3	#r6
3	#q6
3	#s6
3	#r7
3	#q7
3	#s7
3	#q8
3	#s8
k30	#intermediate Jacobian column lengths
3
6
9
12
15
18
21
22
23
24
26
28
30
33
35
37
40
42
44
47
49
51
54
56
58
61
63
65
68
69
J0 3	#rdef2
0 0
8 1
11 -1
J1 4	#qdef2
0 0
9 1
10 25
12 -1
J2 3	#sdef2
0 0
10 1
13 -1
J3 3	#rdef3
1 0
11 1
14 -1
J4 4	#qdef3
1 0
12 1
13 25
15 -1
J5 3	#sdef3
1 0
13 1
16 -1
J6 3	#rdef4
2 0
14 1
17 -1
J7 4	#qdef4
2 0
15 1
16 50
18 -1
J8 3	#sdef4
2 0
16 1
19 -1
J9 3	#rdef5
3 0
17 1
20 -1
J10 4	#qdef5
3 0
18 1
19 50
21 -1
J11 3	#sdef5
3 0
19 1
22 -1
J12 3	#rdef6
4 0
20 1
23 -1
J13 4	#qdef6
4 0
21 1
22 50
24 -1
J14 3	#sdef6
4 0
22 1
25 -1
J15 3	#rdef7
5 0
23 1
26 -1
J16 4	#qdef7
5 0
24 1
25 90
27 -1
J17 3	#sdef7
5 0
25 1
28 -1
J18 3	#rdef8
6 0
7 -1
26 1
J19 4	#qdef8
6 0
27 1
28 90
29 -1
J20 3	#sdef8
6 0
28 1
30 -1
G0 1	#obj
7 0
