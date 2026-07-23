g3 1 1 0	# problem unknown
 2 0 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 2 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 0 2 	# nonzeros in Jacobian, obj. gradient
 3 1	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
O0 0	#obj
o0	#+
o5	#^
o0	#+
v0	#x
n-3
n2
o5	#^
o0	#+
v1	#y
n2
n2
x2	# initial guess
0 0.5	#x
1 0.0	#y
r	#0 ranges (rhs's)
b	#2 bounds (on variables)
0 0 1	#x
0 -1 1	#y
k1	#intermediate Jacobian column lengths
0
G0 2	#obj
0 0
1 0
