$title hand-written test model

Variables x, y, obj;

Equations e1, e2, def_obj;

e1.. x + y =E= 1;
e2.. x*y =G= 0.1;
def_obj.. obj =E= sqr(x) + sqr(y);

Model m / all /;
Solve m using NLP maximizing obj;
