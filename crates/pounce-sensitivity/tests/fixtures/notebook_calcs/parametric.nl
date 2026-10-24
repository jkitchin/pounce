g3 1 1 0	# problem unknown
 5 3 1 0 2 	# vars, constraints, objectives, ranges, eqns
 0 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 5 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 4 5 	# nonzeros in Jacobian, obj. gradient
 5 2	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#pin_p
n0
C1	#pin_q
n0
C2	#cap
n0
O0 0	#obj
o54	# sumlist
5	# (n)
o5	#^
o0	#+
v0	#x1
o2	#*
n-1
v1	#p
n2
o5	#^
o54	# sumlist
3	# (n)
v2	#x2
o2	#*
n-1
v3	#q
n-1.5
n2
o2	#*
o2	#*
n0.2
v0	#x1
v2	#x2
o5	#^
o0	#+
v4	#x3
o2	#*
n-0.3
v1	#p
n2
o2	#*
n0.05
o44	#exp
o2	#*
n0.5
v4	#x3
x5	# initial guess
0 0.5	#x1
1 1.19	#p
2 1.0	#x2
3 0.5	#q
4 0.0	#x3
r	#3 ranges (rhs's)
4 1.19	#pin_p
4 0.5	#pin_q
1 2.3	#cap
b	#5 bounds (on variables)
0 0.0 1.0	#x1
3	#p
0 0.0 10.0	#x2
3	#q
0 -5.0 5.0	#x3
k4	#intermediate Jacobian column lengths
0
1
2
3
J0 1	#pin_p
1 1
J1 1	#pin_q
3 1
J2 2	#cap
2 1
4 1
G0 5	#obj
0 0
1 0
2 0
3 0
4 0
