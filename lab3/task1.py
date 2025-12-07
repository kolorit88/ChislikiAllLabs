import numpy as np
from scipy import integrate
import sympy as sp


def f(x):
    """Исходная функция для интегрирования: (x+1)*cos(x^2)."""
    return (x + 1) * np.cos(x ** 2)


def analytical_integral(a, b):
    """Аналитическое вычисление интеграла через SymPy."""
    x = sp.symbols('x')
    expr = (x + 1) * sp.cos(x ** 2)  # (x+1)*cos(x^2)
    F = sp.integrate(expr, (x, a, b))
    return float(F.evalf())


def rectangle_method(f, a, b, n):
    """Метод средних прямоугольников."""
    h = (b - a) / n
    x = np.linspace(a + h / 2, b - h / 2, n)
    return h * np.sum(f(x)), h


def trapezoidal_method(f, a, b, n):
    """Метод трапеций."""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1]), h


def simpson_method(f, a, b, n):
    """Обобщенная формула Симпсона (n должно быть четным)."""
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])), h


def three_eighths_method(f, a, b, n):
    """Метод трех восьмых (n должно быть кратно 3)."""
    if n % 3 != 0:
        n = n + (3 - n % 3)
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    sum_3 = 3 * np.sum(y[1:-1:3] + y[2:-1:3])
    sum_2 = 2 * np.sum(y[3:-2:3])
    return 3 * h / 8 * (y[0] + y[-1] + sum_3 + sum_2), h


def gauss_legendre(f, a, b, n_points=4):
    """Квадратурная формула Гаусса с n точками."""
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    t = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    return 0.5 * (b - a) * np.dot(weights, f(t))


def derivative(f, x, n=1, dx=1e-6):
    """Численное дифференцирование."""
    if n == 1:
        return (f(x + dx) - f(x - dx)) / (2 * dx)
    elif n == 2:
        return (f(x + dx) - 2 * f(x) + f(x - dx)) / (dx ** 2)
    elif n == 4:
        return (f(x + 2 * dx) - 4 * f(x + dx) + 6 * f(x) - 4 * f(x - dx) + f(x - 2 * dx)) / (dx ** 4)
    else:
        raise ValueError("Поддерживаются только n=1,2,4")


def estimate_error(method_name, f, a, b, n, h, I_approx):
    """Оценка остаточного члена для различных методов."""
    xi = (a + b) / 2

    if method_name == 'rectangle':
        # R = -f''(ξ)(b-a)h²/24
        f2 = derivative(f, xi, n=2, dx=1e-4)
        R = -f2 * (b - a) * h ** 2 / 24
    elif method_name == 'trapezoidal':
        # R = f''(ξ)(b-a)h²/12
        f2 = derivative(f, xi, n=2, dx=1e-4)
        R = f2 * (b - a) * h ** 2 / 12
    elif method_name == 'simpson':
        # R = -f⁽⁴⁾(ξ)(b-a)h⁴/180
        f4 = derivative(f, xi, n=4, dx=1e-3)
        R = -f4 * (b - a) * h ** 4 / 180
    elif method_name == 'three_eighths':
        # R = -f⁽⁴⁾(ξ)(b-a)h⁴/80
        f4 = derivative(f, xi, n=4, dx=1e-3)
        R = -f4 * (b - a) * h ** 4 / 80
    else:
        R = 0

    return R


def main():
    # Параметры интегрирования
    a, b = 0.2, 1.0
    n = 12  # Кратно 2 и 3 (12, 18, 24, ...)

    print(f"Интегрирование f(x) = (x+1)*cos(x^2) на [{a}, {b}]")
    print(f"Количество отрезков n = {n}\n")

    # Аналитическое значение
    try:
        I_analytical = analytical_integral(a, b)
        print(f"Аналитическое значение: {I_analytical:.10f}")
    except:
        print("Не удалось вычислить аналитический интеграл")
        # Используем SciPy как опорное значение
        I_analytical, _ = integrate.quad(f, a, b)
        print(f"Используем SciPy как опорное: {I_analytical:.10f}")

    # Встроенное вычисление (верификация)
    I_scipy, err_scipy = integrate.quad(f, a, b)
    print(f"SciPy quad: {I_scipy:.10f} ± {err_scipy:.2e}")

    methods = [
        ('rectangle', rectangle_method),
        ('trapezoidal', trapezoidal_method),
        ('simpson', simpson_method),
        ('three_eighths', three_eighths_method),
    ]

    print("\n" + "=" * 50)
    for name, method in methods:
        I, h = method(f, a, b, n)
        R = estimate_error(name, f, a, b, n, h, I)
        print(f"\n{name.capitalize()}:")
        print(f"  Приближение: {I:.10f}")
        print(f"  Ошибка: {abs(I - I_analytical):.2e}")
        print(f"  Оценка R: {R:.2e}")
        print(f"  h = {h:.6f}")

    # Метод Гаусса
    print("\n" + "=" * 50)
    print("Метод Гаусса:")
    for n_points in [4, 5, 7]:
        I_gauss = gauss_legendre(f, a, b, n_points)
        error = abs(I_gauss - I_analytical)
        print(f"  n={n_points}: {I_gauss:.10f}, ошибка: {error:.2e}")

    # Дополнительно: сравнение для разных n
    print("\n" + "=" * 50)
    print("Сравнение методов для разных n (ошибка относительно аналитического):")
    n_values = [6, 12, 24, 48]

    print(f"\n{'n':<6} {'Прямоуг.':<12} {'Трапеции':<12} {'Симпсон':<12} {'3/8':<12}")
    for n_val in n_values:
        results = []
        for name, method in methods:
            I, _ = method(f, a, b, n_val)
            error = abs(I - I_analytical)
            results.append(f"{error:.2e}")
        print(f"{n_val:<6} {results[0]:<12} {results[1]:<12} {results[2]:<12} {results[3]:<12}")


if __name__ == "__main__":
    main()