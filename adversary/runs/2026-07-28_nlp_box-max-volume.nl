g3 1 1 0	# problem unknown
 3 1 1 0 1 	# vars, constraints, objectives, ranges, eqns
 1 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 3 3 3 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 3 3 	# nonzeros in Jacobian, obj. gradient
 7 1	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#surface
o54	# sumlist
3	# (n)
o2	#*
v0	#x
v1	#y
o2	#*
v1	#y
v2	#z
o2	#*
v2	#z
v0	#x
O0 0	#obj
o16	#-
o2	#*
o2	#*
v0	#x
v1	#y
v2	#z
x3	# initial guess
0 2.0	#x
1 0.5	#y
2 3.0	#z
r	#1 ranges (rhs's)
4 3	#surface
b	#3 bounds (on variables)
0 0.1 10	#x
0 0.1 10	#y
0 0.1 10	#z
k2	#intermediate Jacobian column lengths
1
2
J0 3	#surface
0 0
1 0
2 0
G0 3	#obj
0 0
1 0
2 0
