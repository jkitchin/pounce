g3 1 1 0	# problem unknown
 6 5 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 0 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 0 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 30 6 	# nonzeros in Jacobian, obj. gradient
 7 4	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0	#cons[0]
n0
C1	#cons[1]
n0
C2	#cons[2]
n0
C3	#cons[3]
n0
C4	#cons[4]
n0
O0 0	#obj
n0
x6	# initial guess
0 0.1	#x[0]
1 0.1	#x[1]
2 0.1	#x[2]
3 0.1	#x[3]
4 0.1	#x[4]
5 0.1	#x[5]
r	#5 ranges (rhs's)
1 27.023	#cons[0]
1 24.058	#cons[1]
1 34.591	#cons[2]
1 30.278	#cons[3]
1 38.502	#cons[4]
b	#6 bounds (on variables)
0 0 8.0	#x[0]
0 0 8.0	#x[1]
0 0 8.0	#x[2]
0 0 8.0	#x[3]
0 0 8.0	#x[4]
0 0 8.0	#x[5]
k5	#intermediate Jacobian column lengths
5
10
15
20
25
J0 6	#cons[0]
0 3.545
1 1.592
2 0.965
3 0.068
4 4.968
5 4.565
J1 6	#cons[1]
0 4.886
1 4.549
2 0.741
3 2.812
4 3.514
5 2.047
J2 6	#cons[2]
0 3.516
1 3.414
2 3.739
3 1.517
4 4.726
5 2.432
J3 6	#cons[3]
0 0.27
1 0.694
2 3.618
3 4.459
4 2.553
5 3.078
J4 6	#cons[4]
0 4.508
1 0.77
2 2.554
3 4.493
4 3.322
5 2.795
G0 6	#obj
0 1.293
1 -5.943
2 -4.241
3 5.323
4 -2.005
5 -4.026
