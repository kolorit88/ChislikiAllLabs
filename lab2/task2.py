import numpy as np
import math


def factorial(n):
    fact = 1
    for i in range(2, n + 1):
        fact *= i
    return fact


# Исходная функция: y = x * sin(x^2 + x)
def f(x):
    return x * math.sin(x ** 2 + x)


def f_deriv_3(x):
    """Третья производная функции f(x) = x * sin(x^2 + x)

    Вычисляется аналитически:
    1) f(x) = x * sin(x^2 + x)
    2) f'(x) = sin(x^2 + x) + x * cos(x^2 + x) * (2x + 1)
    3) f''(x) = cos(x^2 + x)*(2x + 1) + cos(x^2 + x)*(2x + 1) +
                x * (-sin(x^2 + x))*(2x + 1)^2 + x * cos(x^2 + x)*2
    4) f'''(x) упрощенное выражение
    """
    # Упрощенная формула третьей производной
    return -(2 * x + 1) ** 3 * x * math.cos(x ** 2 + x) + 3 * (2 * x + 1) ** 2 * math.sin(x ** 2 + x) + 6 * (
                2 * x + 1) * math.cos(x ** 2 + x) - 6 * math.sin(x ** 2 + x)


def gauss_interpolation_2(x, x_j, h, y_nodes):
    """Интерполяционный полином Гаусса для n=2"""
    y_jm1, y_j, y_jp1 = y_nodes
    t = (x - x_j) / h
    dy1 = y_jp1 - y_j  # разность первого порядка
    dy2 = y_jp1 - 2 * y_j + y_jm1  # разность второго порядка
    return y_j + t * dy1 + (t * (t - 1) / factorial(2)) * dy2


def estimate_remainder_max(x_j, h):
    """Оценка остаточного члена интерполяции"""
    # Генерируем точки на интервале [x_j - h, x_j + h]
    xi_points = np.linspace(x_j - h, x_j + h, 100)
    max_f3 = max(abs(f_deriv_3(xi)) for xi in xi_points)

    # Максимальное значение |t(t^2 - 1)| для t ∈ [0, 1]
    # Производная t(t^2 - 1) = 3t^2 - 1 = 0 при t = 1/√3
    t_max = 1 / math.sqrt(3)
    max_abs_t_factor = abs(t_max * (t_max ** 2 - 1))
    R2_max = abs(max_f3 * h ** 3 * max_abs_t_factor / factorial(3))
    return R2_max


def find_optimal_h(x0, epsilon):
    """Подбор оптимального шага h для заданной точности epsilon"""
    h = 1.0
    max_iterations = 20

    for _ in range(max_iterations):
        # Определяем ближайший узел x_j
        x_j = round(x0 / h) * h
        x_jm1 = x_j - h
        x_jp1 = x_j + h

        # Значения функции в узлах
        y_values = [f(x_jm1), f(x_j), f(x_jp1)]

        # Оценка максимальной погрешности
        R2_max = estimate_remainder_max(x_j, h)

        print(f"h = {h:.6f}, |max R2| = {R2_max:.10f}")

        if R2_max <= epsilon:
            return h, x_j, y_values

        h *= 0.5

    # Если не достигли требуемой точности, возвращаем последние значения
    return h, x_j, y_values


def main():
    # Заданные параметры
    x0 = 1.3
    epsilon = 1e-6

    print(f"Точка x0 = {x0}")
    print(f"Точность ε = {epsilon} (10^(-6))")

    # Точное значение функции
    exact_value = f(x0)
    print(f"Точное значение f({x0}) = {exact_value:.10f}\n")

    # Поиск оптимального шага
    print("Подбор оптимального шага h:")
    h, x_j, y_values = find_optimal_h(x0, epsilon)

    print(f"\n\tОптимальный h: {h:.6f}")
    print(f"\tУзлы интерполяции:")
    print(f"\t  x_j-1 = {x_j - h:.6f}, f(x_j-1) = {y_values[0]:.10f}")
    print(f"\t  x_j   = {x_j:.6f}, f(x_j)   = {y_values[1]:.10f}")
    print(f"\t  x_j+1 = {x_j + h:.6f}, f(x_j+1) = {y_values[2]:.10f}")

    # Интерполированное значение
    interp_value = gauss_interpolation_2(x0, x_j, h, y_values)

    print(f"\nИнтерполированное значение: {interp_value:.10f}")
    print(f"Абсолютная погрешность: {abs(interp_value - exact_value):.10f}")
    print(f"Относительная погрешность: {abs(interp_value - exact_value) / abs(exact_value) * 100:.6f}%")

    # Оценка остаточного члена для выбранного h
    R_estimate = estimate_remainder_max(x_j, h)
    print(f"\nОценка остаточного члена для h = {h:.6f}: {R_estimate:.10f}")


if __name__ == "__main__":
    main()