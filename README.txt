# CSC 445 Programming Project - Matthew McLean

How to run the solver:

In the lpSolver directory, run the following command:
'py main.py'
The above command runs the solver on some of the less computationally expensive sample LPs.

To run a specific LP, include its file path as an argument, for example:
'py main.py "test_LPs_volume1/input/445k22_A1_juice.txt"'

The results of the solver are displayed in standard output.



This implementation of a linear program solver includes the following features:

- It uses Primal-Dual methods for solving initially infeasible LPs
- It uses the Revised Linear Algebraic Simplex Method
    - Guassian elimination with partial pivoting is used to solve systems of linear equations in place of computing matrix inverses
- It only has functionality for using Bland's rule for pivot selections
    - Bland's rule was selected for its simplicity and cycle avoidance
- Computations are done with the 'fractions' module objects for numerical accuracy