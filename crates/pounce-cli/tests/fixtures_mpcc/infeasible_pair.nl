g3 1 1 0	# problem unknown
 3 4 1 0 1 	# vars, constraints, objectives, ranges, eqns
 0 0 1 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 0 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 5 2 	# nonzeros in Jacobian, obj. gradient
 5 5	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#lx
n0
C1	#ly
n0
C2	#cc.c
n0
C3	#cc.bc
n0
O0 0	#obj
n0
x3	# initial guess
0 2	#x
1 2	#y
2 0.5	#cc.bv
r	#4 ranges (rhs's)
2 1	#lx
2 1	#ly
5 1 2	#cc.c
4 0	#cc.bc
b	#3 bounds (on variables)
2 0	#x
2 0	#y
3	#cc.bv
k2	#intermediate Jacobian column lengths
2
3
J0 1	#lx
0 1
J1 1	#ly
1 1
J2 1	#cc.c
2 1
J3 2	#cc.bc
0 -1
2 1
G0 2	#obj
0 1
1 1
