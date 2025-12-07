import numpy as np
import math

from scipy import integrate

def f(x):
    return (x + 1) * math.cos(x * x)  # (x+1)cos(x²)


def trapezoidal_rule(a, b, n):
    """Формула трапеций для n отрезков"""
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        total += f(a + i * h)
    return total * h


def simpson_rule(a, b, n):
    """Формула Симпсона (n должно быть четным)"""
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        xi = a + i * h
        if i % 2 == 0:
            total += 2 * f(xi)
        else:
            total += 4 * f(xi)
    return total * h / 3


# Параметры задачи
a, b = 0.2, 1.0
epsilon = 1e-6

print("1. Метод с выбором шага из оценки остаточного члена")
print("=" * 50)

# Для оценки остаточного члена нужна оценка производных
# Остаточный член трапеций ~ -(b-a)h²/12 * f''(ξ)
# Оценим максимальную вторую производную на отрезке [0.2, 1]
x_samples = np.linspace(a, b, 1000)
# Численная оценка второй производной
h_small = 1e-5
f2_approx = [(f(x + h_small) - 2 * f(x) + f(x - h_small)) / (h_small * h_small) for x in x_samples]
M2 = max(abs(min(f2_approx)), abs(max(f2_approx)))  # оценка max|f''|

# Подбор шага для трапеций
h_trap = math.sqrt(12 * epsilon / ((b - a) * M2))
n_trap = max(2, int((b - a) / h_trap) + 1)
result_trap = trapezoidal_rule(a, b, n_trap)
print(f"Трапеции: n = {n_trap}, результат = {result_trap:.8f}")


# Для Симпсона: остаточный член ~ -(b-a)h⁴/180 * f⁽⁴⁾(ξ)
# Оценка четвертой производной
def fourth_deriv_approx(x):
    h = 1e-4
    return (f(x - 2 * h) - 4 * f(x - h) + 6 * f(x) - 4 * f(x + h) + f(x + 2 * h)) / (h ** 4)


f4_approx = [fourth_deriv_approx(x) for x in x_samples[2:-2]]
M4 = max(abs(min(f4_approx)), abs(max(f4_approx)))

h_simp = (180 * epsilon / ((b - a) * M4)) ** 0.25
n_simp = max(2, int((b - a) / h_simp) + 1)
if n_simp % 2 != 0:
    n_simp += 1
result_simp = simpson_rule(a, b, n_simp)
print(f"Симпсон:  n = {n_simp}, результат = {result_simp:.8f}")

print("\n2. Метод последовательного удвоения числа шагов")
print("=" * 50)

# Для трапеций
print("\nТрапеции:")
n = 2
prev = trapezoidal_rule(a, b, n)
while True:
    n *= 2
    curr = trapezoidal_rule(a, b, n)
    if abs(curr - prev) < epsilon:
        break
    prev = curr
print(f"n = {n}, результат = {curr:.8f}, погрешность ≈ {abs(curr - prev):.2e}")

# Для Симпсона
print("\nСимпсон:")
n = 2
prev = simpson_rule(a, b, n)
while True:
    n *= 2
    curr = simpson_rule(a, b, n)
    if abs(curr - prev) < epsilon:
        break
    prev = curr
print(f"n = {n}, результат = {curr:.8f}, погрешность ≈ {abs(curr - prev):.2e}")

# Верификация с помощью scipy (если установлена)
print("\n3. Верификация")
print("=" * 50)



quad_result, quad_error = integrate.quad(f, a, b, epsabs=epsilon)
print(f"SciPy quad: {quad_result:.8f} (ошибка ≈ {quad_error:.2e})")

# Сравнение с нашими результатами
print(f"\nРазница с методом трапеций: {abs(curr - quad_result):.2e}")
print(f"Разница с методом Симпсона:  {abs(result_simp - quad_result):.2e}")


# Альтернативная проверка численным интегрированием
print("\n4. Проверка высокой точности (n=10000)")
print("=" * 50)
trap_check = trapezoidal_rule(a, b, 10000)
simp_check = simpson_rule(a, b, 10000)
print(f"Трапеции (n=10000): {trap_check:.8f}")
print(f"Симпсон  (n=10000): {simp_check:.8f}")
print(f"Разница между методами: {abs(trap_check - simp_check):.2e}")