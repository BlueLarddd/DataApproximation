"""
Юнит-тесты
"""
import unittest
import numpy as np
import sys
import os

# Добавляем путь к основной программе
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Slinkina_YU_V_BIS_24_2 import PolynomialApproximation

class TestPolynomialApproximation(unittest.TestCase):
    
    def setUp(self):
        """Подготовка тестовых данных"""
        self.approx = PolynomialApproximation()
        # Тестовые данные: линейная зависимость y = 2x + 1
        self.x_test = np.array([0, 1, 2, 3, 4, 5])
        self.y_test = np.array([1, 3, 5, 7, 9, 11])
        self.approx.x_data = self.x_test
        self.approx.y_data = self.y_test
    
    def test_fit_linear(self):
        """Тест линейной аппроксимации (степень 1)"""
        result = self.approx.fit(1)
        self.assertTrue(result)
        self.assertEqual(self.approx.degree, 1)
        
        # Проверяем коэффициенты (должны быть y = 2x + 1)
        coefs = self.approx.coefficients
        self.assertAlmostEqual(coefs[0], 2.0, places=10)  # a_1
        self.assertAlmostEqual(coefs[1], 1.0, places=10)  # a_0
    
    def test_fit_quadratic(self):
        """Тест квадратичной аппроксимации"""
        # Генерируем квадратичные данные y = x^2
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([0, 1, 4, 9, 16])
        self.approx.x_data = x
        self.approx.y_data = y
        
        result = self.approx.fit(2)
        self.assertTrue(result)
        self.assertEqual(self.approx.degree, 2)
        
        # Проверяем коэффициенты (должны быть y = x^2)
        coefs = self.approx.coefficients
        self.assertAlmostEqual(coefs[0], 1.0, places=10)  # a_2
        self.assertAlmostEqual(coefs[1], 0.0, places=10)  # a_1
        self.assertAlmostEqual(coefs[2], 0.0, places=10)  # a_0
    
    def test_predict(self):
        """Тест предсказания значений"""
        self.approx.fit(1)
        
        # Предсказание для x = 10
        pred = self.approx.predict(np.array([10]))
        self.assertAlmostEqual(pred[0], 21.0, places=10)
        
        # Предсказание для нескольких точек
        x_test = np.array([0, 2, 4, 6])
        pred = self.approx.predict(x_test)
        expected = np.array([1, 5, 9, 13])
        np.testing.assert_array_almost_equal(pred, expected, decimal=10)
    
    def test_r2_score(self):
        """Тест вычисления R^2"""
        self.approx.fit(1)
        # Для идеальной линейной зависимости R^2 должен быть равен 1
        self.assertAlmostEqual(self.approx.r2, 1.0, places=10)
    
    def test_rmse_score(self):
        """Тест вычисления RMSE"""
        self.approx.fit(1)
        # Для идеальной зависимости RMSE должен быть 0
        self.assertAlmostEqual(self.approx.rmse, 0.0, places=10)
    
    def test_fit_with_insufficient_points(self):
        """Тест аппроксимации с недостаточным количеством точек"""
        # Создаем объект с 2 точками
        approx = PolynomialApproximation()
        approx.x_data = np.array([1, 2])
        approx.y_data = np.array([1, 2])
        
        # Попытка аппроксимировать полиномом степени 2 (нужно 3 точки)
        result = approx.fit(2)
        self.assertFalse(result)
        self.assertIsNone(approx.coefficients)
    
    def test_fit_with_invalid_degree(self):
        """Тест с невалидной степенью полинома"""
        result = self.approx.fit(0)  # степень 0 недопустима
        self.assertFalse(result)
        self.assertIsNone(self.approx.coefficients)
    
    def test_predict_without_training(self):
        """Тест предсказания без обучения модели"""
        approx = PolynomialApproximation()
        with self.assertRaises(ValueError):
            approx.predict(np.array([1, 2, 3]))
    
    def test_high_degree_polynomial(self):
        """Тест полинома высокой степени"""
        # Генерируем 10 точек для степени 9
        x = np.linspace(0, 10, 10)
        y = x**9 + 2*x**8 + 3*x**7  # сложная зависимость
        self.approx.x_data = x
        self.approx.y_data = y
        
        result = self.approx.fit(9)
        self.assertTrue(result)
        self.assertEqual(self.approx.degree, 9)
        # R^2 должен быть близок к 1 для точной подгонки
        self.assertGreater(self.approx.r2, 0.99)
    
    def test_load_data_from_csv(self):
        """Тест загрузки данных из CSV"""
        # Создаем временный CSV файл
        import tempfile
        import pandas as pd
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("x,y\n1,2\n3,4\n5,6\n")
            temp_filename = f.name
        
        try:
            approx = PolynomialApproximation()
            result = approx.load_data_from_csv(temp_filename)
            self.assertTrue(result)
            np.testing.assert_array_equal(approx.x_data, np.array([1, 3, 5]))
            np.testing.assert_array_equal(approx.y_data, np.array([2, 4, 6]))
        finally:
            os.unlink(temp_filename)
    
    def test_load_data_from_csv_file_not_found(self):
        """Тест загрузки несуществующего файла"""
        approx = PolynomialApproximation()
        result = approx.load_data_from_csv("non_existent_file.csv")
        self.assertFalse(result)
        self.assertIsNone(approx.x_data)


if __name__ == '__main__':
    unittest.main()