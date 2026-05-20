$title HS071 test problem for POUNCE GAMS solver link
$onText
Hock & Schittkowski problem 71:

  min  x1*x4*(x1+x2+x3) + x3
  s.t. x1*x2*x3*x4 >= 25
       x1^2+x2^2+x3^2+x4^2 = 40
       1 <= xi <= 5,  i=1..4

Expected solution:
  x* = [1.000, 4.743, 3.821, 1.379]
  f* = 17.0140173
$offText

Variables x1, x2, x3, x4, obj;

Equations defobj, con1, con2;

defobj.. obj =e= x1*x4*(x1+x2+x3) + x3;
con1..   x1*x2*x3*x4 =g= 25;
con2..   sqr(x1) + sqr(x2) + sqr(x3) + sqr(x4) =e= 40;

x1.lo = 1; x1.up = 5;
x2.lo = 1; x2.up = 5;
x3.lo = 1; x3.up = 5;
x4.lo = 1; x4.up = 5;

* Starting point
x1.l = 1; x2.l = 5; x3.l = 5; x4.l = 1;

Model hs071 / all /;

option nlp = pounce;

Solve hs071 using nlp minimizing obj;

display x1.l, x2.l, x3.l, x4.l, obj.l;

* Verify solution
abort$(abs(obj.l - 17.014) > 1e-2) 'Objective mismatch', obj.l;
abort$(hs071.solvestat <> 1)       'Unexpected solve status', hs071.solvestat;
abort$(hs071.modelstat <> 2)       'Unexpected model status', hs071.modelstat;
