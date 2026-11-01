g3 1 1 0	# problem unknown
 5 4 1 1 2 	# vars, constraints, objectives, ranges, eqns
 4 0 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 4 0 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 11 1 	# nonzeros in Jacobian, obj. gradient
 5 5	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#cons1
o3	# /
o2	#*
n117.370892
v0	#tph
o2	#*
v1	#wid
v2	#thick
C1	#cons2
o2	#*
n0.020833333333333332
o2	#*
o5	#^
v2	#thick
n2
v3	#ipm
C2	#cons3
o3	# /
v1	#wid
v2	#thick
C3	#cons4
o2	#*
v2	#thick
v1	#wid
O0 0	#f
n0
x5	# initial guess
0 43.0768	#tph
1 536.6054	#wid
2 1.2061	#thick
3 58.0536	#ipm
4 94.103	#len
r	#4 ranges (rhs's)
4 0.0	#cons1
4 0.0	#cons2
1 2.0	#cons3
0 200.0 250.0	#cons4
b	#5 bounds (on variables)
2 45.0	#tph
2 0.0	#wid
2 7.0	#thick
2 0.0	#ipm
2 0.0	#len
k4	#intermediate Jacobian column lengths
1
4
8
10
J0 4	#cons1
0 0
1 0
2 0
3 -1
J1 3	#cons2
2 0
3 0
4 -1
J2 2	#cons3
1 0
2 0
J3 2	#cons4
1 0
2 0
G0 1	#f
4 1
