import numpy as np


def mat_vec_mul(A, x):
    """Умножение матрицы на вектор"""
    n = A.shape[0]
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(A.shape[1]):
            s += A[i, j] * x[j]
        y[i] = s
    return y


def argmax_col_from(A, col, start):
    """Поиск максимального по модулю элемента в столбце col, начиная со строки start"""
    n = A.shape[0]
    max_idx = start
    max_val = abs(A[start, col])
    for i in range(start + 1, n):
        val = abs(A[i, col])
        if val > max_val:
            max_val = val
            max_idx = i
    return max_idx


def vector_norm(v):
    """Вычисление нормы вектора"""
    return float(np.sqrt(np.sum(v * v)))


def error(A, x, f):
    """Вычисление ошибки решения"""
    return vector_norm(f - mat_vec_mul(A, x))


def print_matrix(M, name="Матрица"):
    """Печать матрицы"""
    print(f"{name}:")
    for row in M:
        print("  [ " + "  ".join(f"{v:10.6f}" for v in row) + " ]")


def gauss_solve(A_in, f_in):
    """Решение СЛАУ методом Гаусса с выбором главного элемента"""
    A = A_in.astype(float).copy()
    f = f_in.astype(float).copy()
    n = len(f)

    # Прямой ход с выбором главного элемента
    for k in range(n):
        # Выбор главного элемента
        max_row = argmax_col_from(A, k, k)
        if abs(A[max_row, k]) == 0:
            raise ValueError("Матрица вырожденная.")

        # Перестановка строк
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            f[[k, max_row]] = f[[max_row, k]]

        # Исключение переменной
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:n] -= m * A[k, k:n]
            f[i] -= m * f[k]

    # Обратный ход
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = sum(A[i, j] * x[j] for j in range(i + 1, n))
        x[i] = (f[i] - s) / A[i, i]

    # Проверка решения
    δ = error(A_in, x, f_in)
    print(f"Подстановка решения (Гаусс): δ = {δ}")

    return x


def tst_gauss_solution(A, f, name=""):
    """Тестирование решения СЛАУ методом Гаусса"""
    print(f"=== {name} ===")
    print_matrix(A, "A")
    print("f =", f)

    try:
        print("Метод Гаусса:")
        x_gauss = gauss_solve(A, f)
        print("Решение x =", x_gauss)
        print()
    except Exception as e:
        print("Ошибка:", e)
        print()


if __name__ == "__main__":
    print("ЗАДАНИЕ 1: Решение СЛАУ методом Гаусса с выбором главного элемента")
    print("=" * 70)

    # Тест 1: Явная матрица
    A1 = np.array([[4, 2, 1],
                   [1, 5, 1],
                   [2, 1, 6]], float)
    f1 = np.array([7, 8, 9], float)
    tst_gauss_solution(A1, f1, "Явная матрица")

    # Тест 2: Случайная матрица
    np.random.seed(0)
    A2 = np.random.rand(3, 3)
    f2 = np.random.rand(3)
    tst_gauss_solution(A2, f2, "Случайная матрица")

    # Тест 3: Единичная матрица
    A3 = np.eye(3)
    f3 = np.array([1, 2, 3], float)
    tst_gauss_solution(A3, f3, "Единичная матрица")

    # Тест 4: Матрица Гильберта
    n = 4
    A4 = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)], float)
    f4 = np.ones(n)
    tst_gauss_solution(A4, f4, "Матрица Гильберта (4x4)")