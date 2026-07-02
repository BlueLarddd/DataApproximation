"""
Набор тестовых данных
"""
import numpy as np
import pandas as pd
import os

def generate_test_data():
    """Генерация набора тестовых данных"""
    
    # 1. Линейная зависимость (идеальная)
    x1 = np.linspace(0, 10, 50)
    y1 = 2 * x1 + 3
    
    # 2. Линейная зависимость с шумом
    np.random.seed(42)
    x2 = np.linspace(0, 10, 50)
    y2 = 2 * x2 + 3 + np.random.normal(0, 0.5, 50)
    
    # 3. Квадратичная зависимость
    x3 = np.linspace(-5, 5, 50)
    y3 = x3**2 - 2*x3 + 1
    
    # 4. Квадратичная с шумом
    x4 = np.linspace(-5, 5, 50)
    y4 = 0.5*x4**2 + x4 + 2 + np.random.normal(0, 0.5, 50)
    
    # 5. Кубическая зависимость
    x5 = np.linspace(-3, 3, 50)
    y5 = x5**3 - 2*x5**2 + x5 + 4
    
    # 6. Синусоидальная (для проверки полиномиальной аппроксимации)
    x6 = np.linspace(0, 2*np.pi, 50)
    y6 = np.sin(x6) + np.random.normal(0, 0.1, 50)
    
    # 7. Экспоненциальная
    x7 = np.linspace(0, 5, 50)
    y7 = np.exp(0.5*x7) + np.random.normal(0, 0.5, 50)
    
    # 8. Граничный случай: мало точек (5 точек)
    x8 = np.array([0, 1, 2, 3, 4])
    y8 = np.array([1, 3, 5, 7, 9])
    
    # 9. Граничный случай: очень много точек (200)
    x9 = np.linspace(0, 10, 200)
    y9 = 0.5 * x9**2 - 2 * x9 + 3 + np.random.normal(0, 0.3, 200)
    
    # 10. Граничный случай: неравномерное распределение
    x10 = np.array([0, 0.1, 0.2, 1, 2, 5, 7, 8, 9, 10])
    y10 = 2 * x10 + 1 + np.random.normal(0, 0.2, 10)
    
    return {
        'linear_ideal': (x1, y1),
        'linear_noisy': (x2, y2),
        'quadratic': (x3, y3),
        'quadratic_noisy': (x4, y4),
        'cubic': (x5, y5),
        'sinusoidal': (x6, y6),
        'exponential': (x7, y7),
        'few_points': (x8, y8),
        'many_points': (x9, y9),
        'irregular': (x10, y10)
    }

def create_test_csv_files():
    """Создание CSV файлов для тестирования"""
    
    test_data = generate_test_data()
    
    # Создаем директорию для тестовых данных
    os.makedirs('test_data', exist_ok=True)
    
    for name, (x, y) in test_data.items():
        df = pd.DataFrame({'x': x, 'y': y})
        df.to_csv(f'test_data/{name}.csv', index=False)
        print(f"✓ Создан {name}.csv с {len(x)} точками")
    
    print("\n✅ Все тестовые файлы созданы в папке 'test_data'")

def run_quick_test():
    """Быстрый тест для проверки основных функций"""
    from A_14 import PolynomialApproximation
    
    approx = PolynomialApproximation()
    
    # Тест 1: Линейная зависимость
    print("Тест 1: Линейная зависимость")
    x = np.array([1, 2, 3, 4, 5, 6])
    y = np.array([2, 4, 6, 8, 10, 12])
    approx.x_data = x
    approx.y_data = y
    approx.fit(1)
    print(f"  Коэффициенты: {approx.coefficients}")
    print(f"  R^2 = {approx.r2:.6f}")
    assert approx.r2 > 0.99, "R^2 должен быть близок к 1"
    
    # Тест 2: Квадратичная зависимость
    print("\nТест 2: Квадратичная зависимость")
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([0, 1, 4, 9, 16, 25])
    approx.x_data = x
    approx.y_data = y
    approx.fit(2)
    print(f"  Коэффициенты: {approx.coefficients}")
    print(f"  R^2 = {approx.r2:.6f}")
    assert approx.r2 > 0.99, "R^2 должен быть близок к 1"
    
    # Тест 3: Сравнение степеней
    print("\nТест 3: Сравнение степеней")
    x = np.linspace(0, 10, 30)
    y = 0.3 * x**3 - 2 * x**2 + 3 * x + 5 + np.random.normal(0, 0.5, 30)
    approx.x_data = x
    approx.y_data = y
    approx.compare_degrees(5)
    
    print("\n✅ Все тесты пройдены успешно!")

if __name__ == "__main__":
    print("Генерация тестовых данных...")
    create_test_csv_files()
    
    print("\n" + "="*60)
    print("Запуск быстрого тестирования...")
    print("="*60)
    run_quick_test()