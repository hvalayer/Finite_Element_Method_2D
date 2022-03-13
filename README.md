# Solve 2D Laplace equation with Finite Element Method

Classical FEM framework to solve the Laplace equation over a unit square domain:
```math
\nabla^2u = 0 \;,\; \forall (x,y)\in[0,1]\times[0,1]
```
The code is provided with the following example:
```math
\forall (x,y)\in[0,1]\times[0,1]\;,
\begin{cases}
    &u(x,0) &= &\sin(\pi x) \\[0.5em]
    &u(x,1) &= &0 \\[0.5em]
    &u(0,y) &= &0 \\[0.5em]
    &u(1,y) &= &0
\end{cases},
```
which has analytical solution:
```math
u(x,y) = \sin(\pi x)\left[\cos(\pi y) - \coth(\pi)\sinh(\pi y)\right]
```

The distribution of files is as follows:

  * <ins>exampleFunctions.py :</ins> file containing Dirichlet boundary conditions function and analytical solution function
  * <ins>femFunctions.py :</ins> FEM architecture functions, including some additional plot functions (mesh and errors convergence)
  * <ins>FEM_2D.py :</ins> main file performing the assembly and solving the system, plots generation
