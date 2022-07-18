import numpy as np
import fractions as fr
import bisect as bi


def print_m(m_a):
    row = len(m_a[:, 0])
    col = len(m_a[0, :])
    print("A = [")
    for i in range(row):
        row_list = []
        for j in range(col):
            row_list.append(str(m_a[i, j]))
        to_print = ",\t".join(row_list)
        print("\t" + to_print)
    print("]")


def print_v(v_a):
    print("V = [")
    v_list = []
    for v in v_a:
        v_list.append(str(v))
    to_print = ",\t".join(v_list)
    print("\t" + to_print)
    print("]")


def i_matrix_list(n):
    identity = [fr.Fraction(0, 1)] * n
    identity_m = []
    for i in range(n):
        identity_row = identity.copy()
        identity_row[i] = fr.Fraction(1, 1)
        identity_m.append(identity_row)
    return identity_m


def gauss(m_a, v_a):
    aug_a = np.hstack((m_a, v_a))

    a_rows = len(aug_a[:, 0])
    a_cols = len(aug_a[0, :])

    for i in range(a_rows - 1):
        curr_col = np.absolute(np.array(aug_a[i:, i]))

        # Index of curr_col's max value
        col_max_i = curr_col.argmax() + i
        if aug_a[i, i] != aug_a[col_max_i, i]:
            aug_a[[i, col_max_i]] = aug_a[[col_max_i, i]]

        for j in range(i + 1, a_rows):
            ratio = aug_a[j, i] / aug_a[i, i]
            for k in range(a_cols):
                aug_a[j, k] = aug_a[j, k] - (ratio * aug_a[i, k])

    i = a_rows - 1
    j = a_cols - 1
    solutions = [fr.Fraction(0, 1)] * a_rows
    solutions[i] = aug_a[i, j] / aug_a[i, i]
    for k in range(i - 1, -1, -1):
        sol = aug_a[k, j]
        for l in range(k + 1, j):
            sol = sol - (aug_a[k, l] * solutions[l])
        solutions[k] = sol / aug_a[k, k]

    return np.array(solutions)


def reshape_v(v):
    v_len = len(v)
    v_reshaped = v.reshape(v_len, 1)
    return v_reshaped


def primal_simplex(a, b, c, B, N):
    return_dict = {
        "result": ""
    }
    n_plus_m = len(N) + len(B)
    b_vect = reshape_v(b)
    c_vect = reshape_v(c)

    x = [fr.Fraction(0, 1)] * n_plus_m
    x_b = gauss(a[:, B], b_vect)
    for k in x_b:
        if k < 0:
            return_dict["result"] = "infeasible"
            return return_dict

    for k in range(len(B)):
        x[B[k]] = x_b[k]

    while True:
        v = gauss(np.transpose(a[:, B]), c_vect[B])
        v = reshape_v(v)

        z = [fr.Fraction(0, 1)] * n_plus_m
        z_n = (np.transpose(a[:, N])).dot(v) - c_vect[N]
        for k in range(len(N)):
            z[N[k]] = z_n[k]

        # Check if optimal
        for k in N:
            if z[k] < 0:
                # pivot rule: bland
                j = k
                break
            elif k == N[-1]:
                return_dict["result"] = "optimal"
                return_dict["solutions"] = x
                return_dict["basis"] = B
                return_dict["non-basis"] = N
                return return_dict

        a_j = reshape_v(np.array(a[:, j]))

        delta_x_b = gauss(a[:, B], a_j)
        delta_x = [fr.Fraction(0, 1)] * n_plus_m
        for k in range(len(B)):
            delta_x[B[k]] = delta_x_b[k]

        t_list = []
        for k in B:
            if delta_x[k] > 0:
                t_list.append(x[k] / delta_x[k])

        if len(t_list) == 0:
            return_dict["result"] = "unbounded"
            return return_dict

        t = min(t_list)

        # bland again
        for k in B:
            if delta_x[k] > 0:
                if (x[k] / delta_x[k]) == t:
                    i = k
                    break

        for k in B:
            x[k] = x[k] - (t * delta_x[k])

        x[j] = t
        bi.insort(B, j)
        B.remove(i)
        bi.insort(N, i)
        N.remove(j)
        print(B)


def dual_simplex(a, b, c, B, N):
    return_dict = {
        "result": ""
    }
    n_plus_m = len(N) + len(B)
    b_vect = reshape_v(b)
    c_vect = reshape_v(c)

    v = gauss(np.transpose(a[:, B]), c_vect[B])
    v = reshape_v(v)
    z = [fr.Fraction(0, 1)] * n_plus_m
    z_n = (np.transpose(a[:, N])).dot(v) - c_vect[N]

    for k in z_n:
        if k < 0:
            return_dict["result"] = "infeasible"
            return return_dict

    for k in range(len(N)):
        z[N[k]] = z_n[k]

    while True:
        x = [fr.Fraction(0, 1)] * n_plus_m
        x_b = gauss(a[:, B], b_vect)
        for k in range(len(B)):
            x[B[k]] = x_b[k]

        for k in B:
            if x[k] < 0:
                # pivot rule: bland
                i = k
                break
            if k == B[-1]:
                return_dict["result"] = "optimal"
                return_dict["solutions"] = x
                return_dict["basis"] = B
                return_dict["non-basis"] = N
                return return_dict

        # x_b_vect = reshape_v(x_b)
        # i = x.index(min(x_b_vect))

        u_list = [fr.Fraction(0, 1)] * len(B)
        u_list[B.index(i)] = fr.Fraction(1, 1)
        u = reshape_v(np.array(u_list))

        v = gauss(np.transpose(a[:, B]), u)
        v = reshape_v(v)

        delta_z = [fr.Fraction(0, 1)] * n_plus_m
        delta_z_n = (-1 * np.transpose(a[:, N])).dot(v)
        for k in range(len(N)):
            delta_z[N[k]] = delta_z_n[k]

        s_list = []
        for j in N:
            if delta_z[j] > 0:
                s_list.append(z[j] / delta_z[j])

        if len(s_list) == 0:
            # d unbounded, implies the primal problem is infeasible
            return_dict["result"] = "infeasible"
            return return_dict

        s = min(s_list)

        # bland again
        for k in N:
            if delta_z[k] > 0:
                if (z[k] / delta_z[k]) == s:
                    j = k
                    break

        for k in N:
            z[k] = z[k] - s * delta_z[k]

        z[i] = s
        bi.insort(B, j)
        B.remove(i)
        bi.insort(N, i)
        N.remove(j)
        print(B)


def main():
    files = [
        "test_LPs_volume1/input/445k22_A1_juice.txt",
        "test_LPs_volume1/input/445k22_Lecture01_bakery.txt",
        # "test_LPs_volume1/input/netlib_adlittle.txt",
        # "test_LPs_volume1/input/netlib_afiro.txt",
        # "test_LPs_volume1/input/netlib_bgprtr.txt",
        # "test_LPs_volume1/input/netlib_itest2.txt",
        # "test_LPs_volume1/input/netlib_itest6.txt",
        # "test_LPs_volume1/input/netlib_klein1.txt",
        # "test_LPs_volume1/input/netlib_klein2.txt",
        # "test_LPs_volume1/input/netlib_sc50a.txt",
        # "test_LPs_volume1/input/netlib_sc50b.txt",
        # "test_LPs_volume1/input/netlib_sc105.txt",
        # "test_LPs_volume1/input/netlib_scagr7.txt",
        # "test_LPs_volume1/input/netlib_share1b.txt",
        # "test_LPs_volume1/input/netlib_share2b.txt",
        # "test_LPs_volume1/input/netlib_stocfor1.txt",
        "test_LPs_volume1/input/vanderbei_example2.1.txt",
        "test_LPs_volume1/input/vanderbei_example3.6.txt",
        "test_LPs_volume1/input/vanderbei_example5.6.txt",
        "test_LPs_volume1/input/vanderbei_example6.3.txt",
        "test_LPs_volume1/input/vanderbei_example14.1.txt",
        "test_LPs_volume1/input/vanderbei_exercise2.1.txt",
        "test_LPs_volume1/input/vanderbei_exercise2.2.txt",
        "test_LPs_volume1/input/vanderbei_exercise2.3.txt",
        "test_LPs_volume1/input/vanderbei_exercise2.4.txt",
        "test_LPs_volume1/input/vanderbei_exercise2.5.txt",
        "test_LPs_volume1/input/vanderbei_exercise2.6.txt",
        "test_LPs_volume1/input/vanderbei_exercise2.7.txt",
        "test_LPs_volume1/input/vanderbei_exercise2.8.txt",
        "test_LPs_volume1/input/vanderbei_exercise2.9.txt",
        "test_LPs_volume1/input/vanderbei_exercise2.10.txt",
        "test_LPs_volume1/input/vanderbei_exercise2.11.txt",
    ]
    for file_name in files:
        simplex_solver(file_name)


def simplex_solver(file_path):
    with open(file_path) as input_file:
        input_data = input_file.read()
    input_list = []

    for row in input_data.splitlines():
        curr_row = []
        row_values = row.split()
        for value in row_values:
            curr_row.append(fr.Fraction(value))
        input_list.append(curr_row)

    variables = len(input_list[0])
    constraints = len(input_list) - 1
    a_list = []
    b_list = []
    c_list = input_list[0] + ([fr.Fraction(0, 1)] * constraints)
    identity_m = i_matrix_list(constraints)
    i = 0
    for row in input_list[1:]:
        a_list.append(row[:-1] + identity_m[i])
        b_list.append(row[-1])
        i = i + 1

    a = np.array(a_list)
    b = np.array(b_list)
    c = np.array(c_list)

    n_plus_m = len(c)
    n = n_plus_m - len(b)
    N = []
    B = []
    for i in range(n_plus_m):
        if i < n:
            N.append(i)
        else:
            B.append(i)

    if min(b) >= 0:
        # Primal feasible
        print("primal")
        sol = primal_simplex(a, b, c, B, N)
    elif max(c) <= 0:
        # Dual feasible
        print("dual")
        sol = dual_simplex(a, b, c, B, N)
    else:
        # Neither primal nor dual feasible
        print("neither")
        zeros_list = [fr.Fraction(0, 1)] * len(c)
        zeros = reshape_v(np.array(zeros_list))
        aux_sol = dual_simplex(a, b, zeros, B, N)
        if aux_sol["result"] == "infeasible":
            print(aux_sol["result"])
            return 0
        elif aux_sol["result"] == "unbounded":
            print(aux_sol["result"])
            return 0
        elif aux_sol["result"] == "optimal":
            B_prime = aux_sol["basis"]
            N_prime = aux_sol["non-basis"]
            sol = primal_simplex(a, b, c, B_prime, N_prime)

    if sol["result"] == "infeasible":
        print(sol["result"])
        return 0
    elif sol["result"] == "unbounded":
        print(sol["result"])
        return 0
    elif sol["result"] == "optimal":
        print(sol["result"])
        solutions = []
        optimum = 0
        for i in range(variables):
            value = sol["solutions"][i]
            if value == fr.Fraction(0, 1):
                formatted = "0"
            else:
                float_value = float(value)
                formatted = f'{float_value:.7}'
            solutions.append(formatted)
            optimum = optimum + (value * c[i])
        float_optimum = float(optimum)
        formatted_optimum = f'{float_optimum:.7}'
        print(formatted_optimum)
        print(" ".join(solutions))
    return 0


if __name__ == '__main__':
    main()
