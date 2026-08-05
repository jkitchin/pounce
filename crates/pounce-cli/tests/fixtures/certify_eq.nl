g3 1 1 0	# problem unknown
 2 1 1 0 1 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 2 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 2 2 	# nonzeros in Jacobian, obj. gradient
 3 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c0
n0
O0 0	#obj
o0	#+
o5	#^
o0	#+
v0	#x[0]
n-1
n2
o5	#^
o0	#+
v1	#x[1]
n-2
n2
x0	# initial guess
r	#1 ranges (rhs's)
4 2	#c0
b	#2 bounds (on variables)
3	#x[0]
3	#x[1]
k1	#intermediate Jacobian column lengths
1
J0 2	#c0
0 1
1 1
G0 2	#obj
0 0
1 0
