g3 1 1 0	# qpec_small: min (x-1)^2 + (y1-1)^2 + y2^2, 0<=x<=2, y1 _|_ 2y1-1-x, y2 _|_ x+2y2-1
 3 2 1 0 0	# vars, constraints, objectives, ranges, eqns
 0 1 2 0	# nonlinear constraints, objectives; ccons: lin, nonlin
 0 0	# network constraints: nonlinear, linear
 0 3 0	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0	# discrete variables: binary, integer, nonlinear (b,c,o)
 4 3	# nonzeros in Jacobian, gradients
 0 0	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0
n-1
C1
n-1
O0 0
o54
4
o5
v0
n2
o5
v1
n2
o5
v2
n2
n2
x3
0 0
1 0.5
2 0.5
r
5 1 2
5 1 3
b
0 0 2
2 0
2 0
k2
2
3
J0 2
0 -1
1 2
J1 2
0 1
2 2
G0 3
0 -2
1 -2
2 0
