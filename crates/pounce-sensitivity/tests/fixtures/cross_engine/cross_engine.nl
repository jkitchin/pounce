g3 1 1 0	# problem unknown
 4 3 1 1 1 	# vars, constraints, objectives, ranges, eqns
 1 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 2 4 2 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 7 4 	# nonzeros in Jacobian, obj. gradient
 4 2	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#band
o2	#*
v0	#x1
v1	#x2
C1	#pin
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
n-2.0
v2	#p
n2
o5	#^
o0	#+
v1	#x2
n-1.5
n2
o2	#*
o2	#*
n0.1
v0	#x1
v1	#x2
o5	#^
o0	#+
v3	#x3
o2	#*
n-0.3
v2	#p
n2
o2	#*
n0.05
o44	#exp
o2	#*
n0.5
v3	#x3
x4	# initial guess
0 0.5	#x1
1 0.5	#x2
2 1.0	#p
3 0.0	#x3
r	#3 ranges (rhs's)
0 -1.0 4.0	#band
4 1.0	#pin
1 2.4	#cap
b	#4 bounds (on variables)
0 0.0 1.5	#x1
0 0.0 10.0	#x2
3	#p
0 -5.0 5.0	#x3
k3	#intermediate Jacobian column lengths
2
4
6
J0 3	#band
0 0
1 0
3 -1
J1 1	#pin
2 1
J2 3	#cap
0 1
1 1
2 -0.5
G0 4	#obj
0 0
1 0
2 0
3 0
