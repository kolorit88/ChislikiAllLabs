import numpy as np


def mat_mul(A, B):
    """Умножение матриц"""
    n, m = A.shape
    p = B.shape[1]
    C = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            s = 0.0
            for k in range(m):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C


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


def print_matrix(M, name="Матрица"):
    """Печать матрицы"""
    print(f"{name}:")
    for row in M:
        print("  [ " + "  ".join(f"{v:10.6f}" for v in row) + " ]")


def gauss_det(A_in):
    """Вычисление определителя методом Гаусса с выбором главного элемента"""
    A = A_in.astype(float).copy()
    n = A.shape[0]
    swaps = 0  # счетчик перестановок строк

    # Прямой ход с выбором главного элемента
    for k in range(n):
        # Выбор главного элемента
        max_row = argmax_col_from(A, k, k)
        if abs(A[max_row, k]) == 0:
            return 0.0  # определитель равен нулю

        # Перестановка строк
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            swaps += 1

        # Исключение переменной
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:n] -= m * A[k, k:n]

    # Определитель = произведение диагональных элементов с учетом перестановок
    det = np.prod(np.diag(A))
    return -det if swaps % 2 else det


def gauss_inverse(A_in, eps=1e-6):
    """Вычисление обратной матрицы методом Гаусса-Жордана с выбором главного элемента"""
    A = A_in.astype(float).copy()
    n = A.shape[0]
    I = np.eye(n)  # единичная матрица

    # Метод Гаусса-Жордана
    for k in range(n):
        # Выбор главного элемента
        max_row = argmax_col_from(A, k, k)
        if abs(A[max_row, k]) == 0:
            raise ValueError("Матрица вырожденная, обратной не существует.")

        # Перестановка строк
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            I[[k, max_row]] = I[[max_row, k]]

        # Нормировка строки
        div = A[k, k]
        A[k] /= div
        I[k] /= div

        # Исключение переменной из всех строк
        for i in range(n):
            if i != k:
                factor = A[i, k]
                A[i] -= factor * A[k]
                I[i] -= factor * I[k]

    # Проверка: A * A^{-1} должно быть равно единичной матрице
    R = mat_mul(A_in, I)
    delta = vector_norm(R - np.eye(n))
    print(f"Проверка (A * A^-1): δ = {delta}")

    return I


def tst_det_inverse(A, name=""):
    """Тестирование вычисления определителя и обратной матрицы"""
    print(f"=== {name} ===")
    print_matrix(A, "A")

    try:
        # 1. Вычисление определителя
        print("1. Вычисление определителя:")
        detA = gauss_det(A)
        print(f"det(A) = {detA:.10f}")
        print(f"Проверка (numpy): det(A) = {np.linalg.det(A):.10f}")

        # 2. Вычисление обратной матрицы (если определитель не равен нулю)
        print("\n2. Вычисление обратной матрицы:")
        if abs(detA) > 1e-10:  # если матрица не вырожденная
            A_inv = gauss_inverse(A)
            print_matrix(A_inv, "A^{-1} (метод Гаусса)")
            print("\nПроверка (numpy):")
            print_matrix(np.linalg.inv(A), "A^{-1} (numpy)")
        else:
            print("Матрица вырожденная — A^{-1} не существует.")

    except Exception as e:
        print("Ошибка:", e)

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    print("ЗАДАНИЕ 2: Нахождение определителя и обратной матрицы методом Гаусса")
    print("=" * 70)

    # Тест 1: Явная матрица
    A1 = np.array([[4, 2, 1],
                   [1, 5, 1],
                   [2, 1, 6]], float)
    tst_det_inverse(A1, "Явная матрица 3x3")

    # Тест 2: Случайная матрица
    np.random.seed(0)
    A2 = np.random.rand(4, 4)
    tst_det_inverse(A2, "Случайная матрица 4x4")

    # Тест 3: Единичная матрица
    A3 = np.eye(5)
    tst_det_inverse(A3, "Единичная матрица 5x5")

    # Тест 4: Вырожденная матрица
    A4 = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [1, 1, 1]], float)
    tst_det_inverse(A4, "Вырожденная матрица")

    # Тест 5: Матрица Гильберта
    n = 4
    A5 = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)], float)
    tst_det_inverse(A5, "Матрица Гильберта 4x4")