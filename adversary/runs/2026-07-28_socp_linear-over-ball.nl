g3 1 1 0	# problem unknown
 4 1 1 0 0 	# vars, constraints, objectives, ranges, eqns
 1 0 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 4 0 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 4 4 	# nonzeros in Jacobian, obj. gradient
 4 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#ball
o54	# sumlist
4	# (n)
o5	#^
v0	#x[0]
n2
o5	#^
v1	#x[1]
n2
o5	#^
v2	#x[2]
n2
o5	#^
v3	#x[3]
n2
O0 0	#obj
n0
x4	# initial guess
0 0.1	#x[0]
1 0.1	#x[1]
2 0.1	#x[2]
3 0.1	#x[3]
r	#1 ranges (rhs's)
1 9.0	#ball
b	#4 bounds (on variables)
0 -10 10	#x[0]
0 -10 10	#x[1]
0 -10 10	#x[2]
0 -10 10	#x[3]
k3	#intermediate Jacobian column lengths
1
2
3
J0 4	#ball
0 0
1 0
2 0
3 0
G0 4	#obj
0 -1.118
1 -1.974
2 -0.325
3 -3.718
