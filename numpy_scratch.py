import numpy as np
import fractions as fr


def f(a, b):
    fraction = fr.Fraction(a, b)
    return fraction


def vector_builder1():
    v_a = np.array([[f(-1, 1)],
                    [f(2, 1)],
                    [f(-4, 1)],
                    [f(0, 1)]
                    ])
    return v_a


def main():
    temp = float(fr.Fraction(31, 3))
    to_print = f'{temp:.3}'
    print(to_print.__class__)


if __name__ == '__main__':
    main()