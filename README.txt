# CSC 445 Programming Project - Matthew McLean
# If for some reason there are any problems with my file submitted, please pull my code from https://github.com/mattmc-00/lpSolver



How to run the solver:

In the lpSolver directory, run the following command:
"py main.py"

Then in standard input, provide the linear program in the standard encoding format with each line separated by '\n'.
For example:

To input the LP from '445k22_A1_juice.txt', input the following:
"py main.py" <press enter>
"13 12 9 \n 0.5 0.4 0.4 10 \n 0.3 0 0 5 \n 0.1 0.2 0.4 10 \n 0 0.3 0 1 \n 0 0.1 0.2 2" <press enter>

To input the LP from 'vanderbei_example2.1.txt', input the following:
"py main.py" <press enter>
"5	4	3\n2	3	1	5\n4	1	2	11\n3	4	2	8"

The results of the solver are displayed in standard output.



This implementation of a linear program solver includes the following features:

- It uses Primal-Dual methods for solving initially infeasible LPs
- It uses the Revised Linear Algebraic Simplex Method
    - Guassian elimination with partial pivoting is used to solve systems of linear equations in place of computing matrix inverses
- It only has functionality for using Bland's rule for pivot selections
    - Bland's rule was selected for its simplicity and cycle avoidance



The following python modules are used:
- numpy
- fractions
- bisect
- sys