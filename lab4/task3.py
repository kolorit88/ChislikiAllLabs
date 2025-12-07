import numpy as np


def jacobi(A, f, eps=1e-6, max_iter=1000):
    """Метод Якоби: x_i = (f_i - Σ_{j≠i} a_ij*x_j) / a_ii"""
    n = len(f)
    x = np.zeros(n)

    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            s = np.dot(A[i], x) - A[i, i] * x[i]  # сумма без диагонального элемента
            x_new[i] = (f[i] - s) / A[i, i]

        if np.max(np.abs(x_new - x)) < eps:
            return x_new, k + 1
        x = x_new

    return x, max_iter


def seidel(A, f, eps=1e-6, max_iter=1000):
    """Метод Зейделя: использует обновленные значения сразу"""
    n = len(f)
    x = np.zeros(n)

    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])  # уже обновленные
            s2 = np.dot(A[i, i + 1:], x[i + 1:])  # еще старые
            x[i] = (f[i] - s1 - s2) / A[i, i]

        if np.max(np.abs(x - x_old)) < eps:
            return x, k + 1

    return x, max_iter


# Пример использования
if __name__ == "__main__":
    # Тестовая матрица с диагональным преобладанием
    A = np.array([[10, 2, 1],
                  [1, 8, 2],
                  [2, 1, 12]], float)
    f = np.array([13, 11, 15], float)
    eps = 1e-6

    print("Метод Якоби:")
    x_j, iter_j = jacobi(A, f, eps)
    print(f"Решение: {x_j}, итераций: {iter_j}")
    print(f"Ошибка: {np.linalg.norm(f - A @ x_j):.2e}")

    print("\nМетод Зейделя:")
    x_s, iter_s = seidel(A, f, eps)
    print(f"Решение: {x_s}, итераций: {iter_s}")
    print(f"Ошибка: {np.linalg.norm(f - A @ x_s):.2e}")