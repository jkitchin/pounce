g3 1 1 0	# problem unknown
 3 1 1 0 1 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 3 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 1 3 	# nonzeros in Jacobian, obj. gradient
 5 1	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#pin_p
n0
O0 0	#obj
o54	# sumlist
4	# (n)
o2	#*
n0.5
o5	#^
v0	#k
n2
o2	#*
o2	#*
n0.999499874937461
v0	#k
v1	#y
o2	#*
n0.5
o5	#^
v1	#y
n2
o16	#-
o2	#*
o2	#*
n1.1
v2	#p
v0	#k
x3	# initial guess
0 0.3	#k
1 0.0	#y
2 0.0	#p
r	#1 ranges (rhs's)
4 0.0	#pin_p
b	#3 bounds (on variables)
0 0.0 10.0	#k
3	#y
3	#p
k2	#intermediate Jacobian column lengths
0
0
J0 1	#pin_p
2 1
G0 3	#obj
0 0
1 0
2 0
