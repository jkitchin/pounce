g3 1 1 0	# problem unknown
 3 2 1 0 2 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 3 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 4 3 	# nonzeros in Jacobian, obj. gradient
 8 2	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
S4 1 scaling_factor
0 2.0
S5 1 scaling_factor
0 0.5
C0	#tagged
n0
C1	#untagged
n0
O0 0	#obj
o54	# sumlist
3	# (n)
o5	#^
v0	#x0
n2
o5	#^
v1	#x1
n2
o5	#^
v2	#x2
n2
x3	# initial guess
0 1.0	#x0
1 1.0	#x1
2 1.0	#x2
r	#2 ranges (rhs's)
4 0	#tagged
4 5	#untagged
b	#3 bounds (on variables)
3	#x0
3	#x1
3	#x2
k2	#intermediate Jacobian column lengths
2
4
J0 2	#tagged
0 2
1 1
J1 2	#untagged
0 1000
1 1
G0 3	#obj
0 0
1 0
2 0
