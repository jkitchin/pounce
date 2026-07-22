g3 1 1 0	# problem unknown
 2 3 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 0 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 0 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 4 2 	# nonzeros in Jacobian, obj. gradient
 3 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c0
n0
C1	#c1
n0
C2	#c2
n0
O0 0	#obj
n0
x0	# initial guess
r	#3 ranges (rhs's)
2 2	#c0
2 0	#c1
2 0	#c2
b	#2 bounds (on variables)
3	#x[0]
3	#x[1]
k1	#intermediate Jacobian column lengths
2
J0 2	#c0
0 1
1 1
J1 1	#c1
0 -1
J2 1	#c2
1 -1
G0 2	#obj
0 1
1 1
