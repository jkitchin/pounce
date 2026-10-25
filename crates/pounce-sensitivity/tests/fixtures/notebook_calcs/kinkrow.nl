g3 1 1 0	# problem unknown
 2 2 1 0 1 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 2 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 2 2 	# nonzeros in Jacobian, obj. gradient
 5 1	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#pin_p
n0
C1	#cap
n0
O0 0	#obj
o5	#^
o0	#+
v0	#x
o2	#*
n-1
v1	#p
n2
x2	# initial guess
0 0.5	#x
1 1.0	#p
r	#2 ranges (rhs's)
4 1.0	#pin_p
1 1.0	#cap
b	#2 bounds (on variables)
3	#x
3	#p
k1	#intermediate Jacobian column lengths
1
J0 1	#pin_p
1 1
J1 1	#cap
0 1
G0 2	#obj
0 0
1 0
