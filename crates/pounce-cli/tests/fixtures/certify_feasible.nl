g3 1 1 0	# problem certify_feasible
 2 2 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 2 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 4 2 	# nonzeros in Jacobian, obj. gradient
 3 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#c0
n0
C1	#c1
n0
O0 0	#obj
o0	#+
o2	#*
n-0.05
o5	#^
v0	#x[0]
n2
o2	#*
n-0.05
o5	#^
v1	#x[1]
n2
x2	# initial guess
0 1.5
1 1.5
r	#2 ranges (rhs's)
2 4	#c0
2 4	#c1
b	#2 bounds (on variables)
0 0 3	#x[0]
0 0 3	#x[1]
k1	#intermediate Jacobian column lengths
2
J0 2	#c0
0 1
1 2
J1 2	#c1
0 2
1 1
G0 2	#obj
0 1
1 1
