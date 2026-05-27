*  NLP written by GAMS Convert at 07/19/01 13:40:01
*
*  Equation counts
*     Total       E       G       L       N       X
*       109     108       0       1       0       0
*
*  Variable counts
*                 x       b       i     s1s     s2s      sc      si
*     Total    cont  binary integer    sos1    sos2   scont    sint
*       142     142       0       0       0       0       0       0
*  FX     0       0       0       0       0       0       0       0
*
*  Nonzero counts
*     Total   const      NL     DLL
*       729     162     567       0
*
*  Solve m using NLP minimizing objvar;

$offlisting
Variables  objvar,x2;

Equations e1;
e1.. objvar =E= sqr(x2);

Model m / all /;
Solve m using NLP minimizing objvar;
