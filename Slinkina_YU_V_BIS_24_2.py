"""
Аппроксимация данных методом МНК
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

class PolynomialApproximation:
    def __init__(self):
        self.x_data = None
        self.y_data = None
        self.coefficients = None
        self.degree = None
        
    def load_data_from_csv(self, filename="data.csv"):
        """загрузка данных из CSV файла data.csv"""
        try:
            data = pd.read_csv(filename)
            self.x_data = data['x'].values
            self.y_data = data['y'].values
            print(f" данные загружены: {len(self.x_data)} точек")
            print(f"  X: от {self.x_data.min():.4f} до {self.x_data.max():.4f}")
            print(f"  Y: от {self.y_data.min():.4f} до {self.y_data.max():.4f}")
            return True
        except FileNotFoundError:
            print(f"ошибка: файл '{filename}' не найден")
            return False
        except KeyError as e:
            print(f"ошибка: колонка {e} не найдена в файле")
            print(" файл должен содержать колонки 'x' и 'y'")
            return False
        except Exception as e:
            print(f"ошибка при загрузке данных: {e}")
            return False
    
    def fit(self, degree):
        """аппроксимация полиномом заданной степени"""
        if self.x_data is None or self.y_data is None:
            print("ошибка: данные не загружены")
            return False
        
        if degree < 1:
            print("ошибка: степень полинома должна быть >= 1")
            return False
        
        if len(self.x_data) < degree + 1:
            print(f"ошибка: для степени {degree} нужно минимум {degree+1} точек, в данный момент {len(self.x_data)}")
            return False
        
        self.degree = degree
        
        # тут вычисляем коэффициенты полинома
        self.coefficients = np.polyfit(self.x_data, self.y_data, degree)
        
        # а тут вычисляем значения аппроксимации
        y_pred = self.predict(self.x_data)
        
        # вычисляем метрики качества
        self.r2 = r2_score(self.y_data, y_pred)
        self.rmse = np.sqrt(mean_squared_error(self.y_data, y_pred))
        self.mse = mean_squared_error(self.y_data, y_pred)
        
        return True

    def predict(self, x):
        """предсказание значений по полиному"""
        if self.coefficients is None:
            raise ValueError("модель не обучена, вызовите fit() сначала.")
        return np.polyval(self.coefficients, x)
    
    def print_results(self):
        """вывод результатов в консоль"""
        if self.coefficients is None:
            print("модель не обучена")
            return
        
        print("\n" + "="*60)
        print(f"Результаты аппроксимации (степень {self.degree})")
        print("="*60)
        
        # вывод коэффициентов
        print("\n Коэффициенты полинома (от старшей степени к младшей):")
        for i, coef in enumerate(self.coefficients):
            power = self.degree - i
            if power > 1:
                print(f"  a_{power} = {coef:.8f}")
            elif power == 1:
                print(f"  a_1 = {coef:.8f}")
            else:
                print(f"  a_0 = {coef:.8f}")
        
        # вывод метрик качества
        print("\n Метрики качества:")
        print(f"  R^2  = {self.r2:.6f}")
        print(f"  RMSE = {self.rmse:.6f}")
        print(f"  MSE  = {self.mse:.6f}")
        
        # интерпретация R^2
        if self.r2 >= 0.9:
            print("   Отличное качество аппроксимации (R^2 ≥ 0.9)")
        elif self.r2 >= 0.7:
            print("   Хорошее качество аппроксимации (R^2 ≥ 0.7)")
        elif self.r2 >= 0.5:
            print("   Удовлетворительное качество аппроксимации (R^2 ≥ 0.5)")
        else:
            print("   Низкое качество аппроксимации (R^2 < 0.5)")
        
        # вывод уравнения полинома
        print("\n Уравнение полинома:")
        equation = "y = "
        for i, coef in enumerate(self.coefficients):
            power = self.degree - i
            if power > 1:
                equation += f"{coef:+.6f}x^{power} "
            elif power == 1:
                equation += f"{coef:+.6f}x "
            else:
                equation += f"{coef:+.6f}"
        print(f"  {equation}")
        
        print("\n" + "="*60)
    
    def plot(self):
        """Построение графика данных и аппроксимирующей кривой"""
        if self.x_data is None or self.y_data is None:
            print("ошибка: данные не загружены")
            return
        
        if self.coefficients is None:
            print("модель не обучена")
            return
        
        # создаем фигуру
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.canvas.manager.set_window_title('Аппроксимация данных методом МНК | Слинкина Ю.В. | БИС-24-2')
        
        # точки данных
        ax.scatter(self.x_data, self.y_data, color='blue', alpha=0.6, 
                    label='Экспериментальные данные', s=50)
        
        # аппроксимирующая кривая (гладкая линия)
        x_smooth = np.linspace(self.x_data.min() - 0.5, self.x_data.max() + 0.5, 200)
        y_smooth = self.predict(x_smooth)
        ax.plot(x_smooth, y_smooth, color='red', linewidth=2.5, 
                label=f'Аппроксимация (степень {self.degree})')
        
        # добавляем предсказанные значения для точек данных
        y_pred = self.predict(self.x_data)
        ax.plot(self.x_data, y_pred, 'ro', alpha=0.3, markersize=4)
        
        # настройка графика
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title(f'Аппроксимация данных методом наименьших квадратов (степень {self.degree})', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)
        
        # добавляем информацию на график
        info_text = f'R^2 = {self.r2:.4f}\nRMSE = {self.rmse:.4f}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # добавляем уравнение
        equation = "y = "
        for i, coef in enumerate(self.coefficients):
            power = self.degree - i
            if power > 1:
                equation += f"{coef:+.3f}x^{power} "
            elif power == 1:
                equation += f"{coef:+.3f}x "
            else:
                equation += f"{coef:+.3f}"
        ax.text(0.02, 0.88, equation, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    def compare_degrees(self, max_degree):
        """Сравнение аппроксимации для разных степеней полинома"""
        print("\n" + "="*60)
        print(f"Сравнение различных степеней полинома (от 1 до {max_degree})")
        print("="*60)
        
        results = []
        for deg in range(1, max_degree + 1):
            if deg > len(self.x_data) - 1:
                break
            self.fit(deg)
            results.append({
                'degree': deg,
                'r2': self.r2,
                'rmse': self.rmse,
                'mse': self.mse
            })
            print(f"Степень {deg:2d}: R^2 = {self.r2:.6f}, RMSE = {self.rmse:.6f}")
        
        # находим лучшую степень по R^2
        best = max(results, key=lambda x: x['r2'])
        print(f"\n Лучшая степень по R^2: {best['degree']} (R^2 = {best['r2']:.6f})")
        
        # восстанавливаем модель для лучшей степени
        self.fit(best['degree'])
        print("="*60)


def main():
    """Основная функция программы"""
    print("\n" + "="*60)
    print("Аппроксимация данных методом наименьших квадратов")
    print("Слинкина Ю.В. | БИС-24-2")
    print("="*60)
    
    # создаем экземпляр класса
    approx = PolynomialApproximation()
    
    # загружаем данные из data.csv
    if not approx.load_data_from_csv("data.csv"):
        print("\n Программа завершена. Создайте файл data.csv с колонками x и y")
        return
    
    # основной цикл
    while True:
        print("\n" + "-"*60)
        print("ДОСТУПНЫЕ ДЕЙСТВИЯ:")
        print("1. Выполнить аппроксимацию полиномом заданной степени")
        print("2. Сравнить разные степени полинома")
        print("3. Показать график")
        print("4. Выйти из программы")
        
        action = input("> ").strip()
        
        if action == '1':
            try:
                degree = int(input("Введите степень полинома (1-10): "))
                if approx.fit(degree):
                    approx.print_results()
                    show_plot = input("Показать график? (y/n): ").strip().lower()
                    if show_plot == 'y':
                        approx.plot()
            except ValueError:
                print("ошибка: введите целое число")
        
        elif action == '2':
            try:
                max_degree = int(input("Введите максимальную степень для сравнения: "))
                approx.compare_degrees(max_degree)
                show_plot = input("Показать график для лучшей степени? (y/n): ").strip().lower()
                if show_plot == 'y':
                    approx.plot()
            except ValueError:
                print("ошибка: введите целое число")
        
        elif action == '3':
            if approx.coefficients is not None:
                approx.plot()
            else:
                print("Сначала выполните аппроксимацию (действие 1)")
        
        elif action == '4':
            print("\nПрограмма завершена")
            break
        
        else:
            print("Неверный выбор, попробуйте снова.")


if __name__ == "__main__":
    main()