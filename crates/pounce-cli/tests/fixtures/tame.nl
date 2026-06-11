g3 1 1 0	# problem TAME
 2 1 1 0 1 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 2 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 2 2 	# nonzeros in Jacobian, obj. gradient
 4 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c[0]
n0
O0 0	#obj
o54	# sumlist
4	# (n)
o2	#*
v0	#x[0]
v0	#x[0]
o2	#*
o2	#*
n-1.0
v1	#x[1]
v0	#x[0]
o2	#*
o2	#*
n-1.0
v0	#x[0]
v1	#x[1]
o2	#*
v1	#x[1]
v1	#x[1]
x2	# initial guess
0 0.0	#x[0]
1 0.0	#x[1]
r	#1 ranges (rhs's)
4 1.0	#c[0]
b	#2 bounds (on variables)
2 0.0	#x[0]
2 0.0	#x[1]
k1	#intermediate Jacobian column lengths
1
J0 2	#c[0]
0 1
1 1
G0 2	#obj
0 0
1 0
