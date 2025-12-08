import numpy as np
import matplotlib.pyplot as plt

# Исходная функция
def f(x):
    return x * np.sin(x**2 + x)

# Линейный сплайн
def linear_spline(x_nodes, y_nodes, x):
    n = len(x_nodes) - 1
    for i in range(n):
        if x_nodes[i] <= x <= x_nodes[i + 1]:
            return y_nodes[i] + (y_nodes[i + 1] - y_nodes[i]) / (x_nodes[i + 1] - x_nodes[i]) * (x - x_nodes[i])
    # Если x вне диапазона, используем крайние отрезки
    if x < x_nodes[0]:
        return y_nodes[0] + (y_nodes[1] - y_nodes[0]) / (x_nodes[1] - x_nodes[0]) * (x - x_nodes[0])
    else:
        return y_nodes[-2] + (y_nodes[-1] - y_nodes[-2]) / (x_nodes[-1] - x_nodes[-2]) * (x - x_nodes[-2])

# Параболический сплайн
def parabolic_spline(x_nodes, y_nodes):
    n = len(x_nodes) - 1
    # Количество парабол: n (по одной на каждый интервал)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    
    # Используем условия непрерывности и гладкости
    # В каждом узле (кроме первого и последнего) совпадают значения и первые производные
    A = np.zeros((3*n, 3*n))
    rhs = np.zeros(3*n)
    
    # Условия совпадения значений в узлах
    for i in range(n):
        # Левое значение параболы i в точке x_nodes[i]
        A[i, 3*i] = x_nodes[i]**2
        A[i, 3*i + 1] = x_nodes[i]
        A[i, 3*i + 2] = 1
        rhs[i] = y_nodes[i]
        
        # Правое значение параболы i в точке x_nodes[i+1]
        A[n + i, 3*i] = x_nodes[i+1]**2
        A[n + i, 3*i + 1] = x_nodes[i+1]
        A[n + i, 3*i + 2] = 1
        rhs[n + i] = y_nodes[i+1]
    
    # Условия непрерывности первых производных во внутренних узлах
    for i in range(n-1):
        idx = 2*n + i
        # Производная параболы i в точке x_nodes[i+1]
        A[idx, 3*i] = 2 * x_nodes[i+1]
        A[idx, 3*i + 1] = 1
        
        # Производная параболы i+1 в точке x_nodes[i+1] (с минусом)
        A[idx, 3*(i+1)] = -2 * x_nodes[i+1]
        A[idx, 3*(i+1) + 1] = -1
        
        # Производная параболы i+2 в точке x_nodes[i+1] (если есть)
        if i + 2 < n:
            A[idx, 3*(i+2)] = 0
            A[idx, 3*(i+2) + 1] = 0
        
        A[idx, 3*i + 2] = 0
        if i + 1 < n:
            A[idx, 3*(i+1) + 2] = 0
        
        rhs[idx] = 0
    
    # Условие на вторую производную в первом узле (естественный сплайн)
    idx = 3*n - 1
    A[idx, 0] = 2  # a0'' = 2*a0
    A[idx, 1] = 0
    A[idx, 2] = 0
    rhs[idx] = 0
    
    # Решаем систему
    coeffs = np.linalg.lstsq(A, rhs, rcond=None)[0]
    
    # Преобразуем коэффициенты
    for i in range(n):
        a[i] = coeffs[3*i]
        b[i] = coeffs[3*i + 1]
        c[i] = coeffs[3*i + 2]
    
    return a, b, c

def eval_parabolic_spline(x_nodes, coeffs, x):
    a, b, c = coeffs
    n = len(x_nodes) - 1
    
    for i in range(n):
        if x_nodes[i] <= x <= x_nodes[i + 1]:
            return a[i] * x**2 + b[i] * x + c[i]
    
    # Если x вне диапазона, используем крайние параболы
    if x < x_nodes[0]:
        return a[0] * x**2 + b[0] * x + c[0]
    else:
        return a[-1] * x**2 + b[-1] * x + c[-1]

# Кубический сплайн (естественный)
def cubic_spline(x_nodes, y_nodes):
    n = len(x_nodes) - 1
    h = np.diff(x_nodes)
    
    # Матрица для нахождения вторых производных (M)
    A = np.zeros((n + 1, n + 1))
    b_vec = np.zeros(n + 1)
    
    # Естественные граничные условия
    A[0, 0] = 1
    A[n, n] = 1
    
    # Заполняем внутренние строки
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b_vec[i] = 3 * ((y_nodes[i + 1] - y_nodes[i]) / h[i] - (y_nodes[i] - y_nodes[i - 1]) / h[i - 1])
    
    # Решаем систему для M (вторых производных в узлах)
    M = np.linalg.solve(A, b_vec)
    
    # Коэффициенты сплайна
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    
    for i in range(n):
        a[i] = y_nodes[i]
        b[i] = (y_nodes[i + 1] - y_nodes[i]) / h[i] - h[i] * (2 * M[i] + M[i + 1]) / 3
        c[i] = M[i]
        d[i] = (M[i + 1] - M[i]) / (3 * h[i])
    
    return a, b, c, d

def eval_cubic_spline(x_nodes, coeffs, x):
    a, b, c, d = coeffs
    n = len(x_nodes) - 1
    
    for i in range(n):
        if x_nodes[i] <= x <= x_nodes[i + 1]:
            dx = x - x_nodes[i]
            return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
    
    # Если x вне диапазона, используем крайние отрезки
    if x < x_nodes[0]:
        dx = x - x_nodes[0]
        return a[0] + b[0] * dx + c[0] * dx**2 + d[0] * dx**3
    else:
        dx = x - x_nodes[-2]
        return a[-1] + b[-1] * dx + c[-1] * dx**2 + d[-1] * dx**3

# Проведение расчетов для разных n
def run_calculations_for_n(n_values, a, b):
    results = {}
    
    for n in n_values:
        print(f"\n{'='*50}")
        print(f"РАСЧЕТ ДЛЯ n = {n}")
        print(f"{'='*50}")
        
        # Узлы интерполяции
        x_nodes = np.linspace(a, b, n + 1)
        y_nodes = f(x_nodes)
        
        # Точки для построения графиков
        x_dense = np.linspace(a, b, 1000)
        y_exact = f(x_dense)
        
        # Линейный сплайн
        y_linear = np.array([linear_spline(x_nodes, y_nodes, x) for x in x_dense])
        linear_errors = np.abs(y_exact - y_linear)
        max_linear_error = np.max(linear_errors)
        
        # Параболический сплайн
        coeffs_parabolic = parabolic_spline(x_nodes, y_nodes)
        y_parabolic = np.array([eval_parabolic_spline(x_nodes, coeffs_parabolic, x) for x in x_dense])
        parabolic_errors = np.abs(y_exact - y_parabolic)
        max_parabolic_error = np.max(parabolic_errors)
        
        # Кубический сплайн
        coeffs_cubic = cubic_spline(x_nodes, y_nodes)
        y_cubic = np.array([eval_cubic_spline(x_nodes, coeffs_cubic, x) for x in x_dense])
        cubic_errors = np.abs(y_exact - y_cubic)
        max_cubic_error = np.max(cubic_errors)
        
        # Сохраняем результаты
        results[n] = {
            'x_nodes': x_nodes,
            'y_nodes': y_nodes,
            'x_dense': x_dense,
            'y_exact': y_exact,
            'linear': {'y': y_linear, 'errors': linear_errors, 'max_error': max_linear_error},
            'parabolic': {'y': y_parabolic, 'errors': parabolic_errors, 'max_error': max_parabolic_error},
            'cubic': {'y': y_cubic, 'errors': cubic_errors, 'max_error': max_cubic_error}
        }
        
        # Вывод максимальных погрешностей
        print(f"Максимальная погрешность линейного сплайна: {max_linear_error:.6f}")
        print(f"Максимальная погрешность параболического сплайна: {max_parabolic_error:.6f}")
        print(f"Максимальная погрешность кубического сплайна: {max_cubic_error:.6f}")
        
        # Построение графиков для каждого n
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Интерполяция функции $x\\sin(x^2 + x)$ сплайнами (n={n})', fontsize=14)
        
        # График 1: Функция и сплайны
        ax = axes[0, 0]
        ax.plot(x_dense, y_exact, 'k-', linewidth=2, label='$f(x) = x\\sin(x^2 + x)$')
        ax.plot(x_dense, y_linear, 'b--', linewidth=1, label='Линейный сплайн $S_1(x)$')
        ax.plot(x_dense, y_parabolic, 'g-.', linewidth=1, label='Параболический сплайн $S_2(x)$')
        ax.plot(x_dense, y_cubic, 'r:', linewidth=1, label='Кубический сплайн $S_3(x)$')
        ax.scatter(x_nodes, y_nodes, c='red', s=30, zorder=5, label='Узлы интерполяции')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Функция и интерполирующие сплайны')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # График 2: Погрешности
        ax = axes[0, 1]
        ax.plot(x_dense, linear_errors, 'b--', linewidth=1, label='Линейный сплайн')
        ax.plot(x_dense, parabolic_errors, 'g-.', linewidth=1, label='Параболический сплайн')
        ax.plot(x_dense, cubic_errors, 'r:', linewidth=1, label='Кубический сплайн')
        ax.set_xlabel('x')
        ax.set_ylabel('Абсолютная погрешность')
        ax.set_title('Погрешности интерполяции')
        ax.set_yscale('log')  # Логарифмическая шкала для лучшей видимости
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # График 3: Отдельно погрешность линейного сплайна
        ax = axes[1, 0]
        ax.plot(x_dense, linear_errors, 'b-', linewidth=1)
        ax.fill_between(x_dense, 0, linear_errors, alpha=0.3, color='blue')
        ax.set_xlabel('x')
        ax.set_ylabel('Абсолютная погрешность')
        ax.set_title(f'Погрешность линейного сплайна (макс: {max_linear_error:.4f})')
        ax.grid(True, alpha=0.3)
        
        # График 4: Отдельно погрешность параболического и кубического сплайнов
        ax = axes[1, 1]
        ax.plot(x_dense, parabolic_errors, 'g-', linewidth=1, label='Параболический')
        ax.plot(x_dense, cubic_errors, 'r-', linewidth=1, label='Кубический')
        ax.fill_between(x_dense, 0, parabolic_errors, alpha=0.2, color='green')
        ax.fill_between(x_dense, 0, cubic_errors, alpha=0.2, color='red')
        ax.set_xlabel('x')
        ax.set_ylabel('Абсолютная погрешность')
        ax.set_title(f'Погрешности параболического (макс: {max_parabolic_error:.4f}) и кубического (макс: {max_cubic_error:.4f}) сплайнов')
        ax.set_yscale('log')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return results

# Основная часть программы
def main():
    # Параметры интерполяции
    a = 0.0  # начало интервала
    b = 5.0  # конец интервала (длина 5)
    
    # Значения n для исследования
    n_values = [5, 10, 25, 50, 100]
    
    print("ИНТЕРПОЛЯЦИЯ ФУНКЦИИ f(x) = x*sin(x^2 + x)")
    print(f"Интервал: [{a}, {b}]")
    print(f"Длина интервала: {b - a}")
    print(f"\nБудут проведены расчеты для n = {n_values}")
    
    # Проводим расчеты
    results = run_calculations_for_n(n_values, a, b)
    
    # Сводная таблица максимальных погрешностей
    print("\n" + "="*60)
    print("СВОДНАЯ ТАБЛИЦА МАКСИМАЛЬНЫХ ПОГРЕШНОСТЕЙ")
    print("="*60)
    print(f"{'n':>5} | {'Линейный':>12} | {'Параболический':>15} | {'Кубический':>12}")
    print("-"*60)
    
    max_errors_summary = []
    for n in n_values:
        res = results[n]
        max_errors_summary.append({
            'n': n,
            'linear': res['linear']['max_error'],
            'parabolic': res['parabolic']['max_error'],
            'cubic': res['cubic']['max_error']
        })
        print(f"{n:5d} | {res['linear']['max_error']:12.6f} | {res['parabolic']['max_error']:15.6f} | {res['cubic']['max_error']:12.6f}")
    
    # График зависимости максимальной погрешности от n
    plt.figure(figsize=(10, 6))
    
    n_list = [err['n'] for err in max_errors_summary]
    linear_errors = [err['linear'] for err in max_errors_summary]
    parabolic_errors = [err['parabolic'] for err in max_errors_summary]
    cubic_errors = [err['cubic'] for err in max_errors_summary]
    
    plt.loglog(n_list, linear_errors, 'bo-', linewidth=2, markersize=8, label='Линейный сплайн')
    plt.loglog(n_list, parabolic_errors, 'gs-', linewidth=2, markersize=8, label='Параболический сплайн')
    plt.loglog(n_list, cubic_errors, 'r^-', linewidth=2, markersize=8, label='Кубический сплайн')
    
    plt.xlabel('Число интервалов (n)', fontsize=12)
    plt.ylabel('Максимальная абсолютная погрешность', fontsize=12)
    plt.title('Зависимость максимальной погрешности от числа интервалов', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Добавляем сетку
    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='-', alpha=0.7)
    plt.grid(True, which='minor', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("ВЫВОДЫ:")
    print("="*60)
    print("1. Кубический сплайн имеет наименьшую погрешность для всех n.")
    print("2. С увеличением n погрешность всех сплайнов уменьшается.")
    print("3. Параболический сплайн имеет промежуточную точность между")
    print("   линейным и кубическим сплайнами.")
    print("4. На интервале [0, 5] функция x*sin(x^2 + x) имеет осцилляции,")
    print("   что требует достаточно большого n для точной интерполяции.")

if __name__ == "__main__":
    main()
