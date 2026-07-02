"""
Интеграционные тесты
"""
import unittest
import numpy as np
import sys
import os
import tempfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Slinkina_YU_V_BIS_24_2 import PolynomialApproximation

class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        """Создание экземпляра для тестов"""
        self.approx = PolynomialApproximation()
    
    def test_full_workflow_linear(self):
        """Полный рабочий процесс: загрузка -> обучение -> предсказание"""
        self.approx.x_data = np.array([1, 2, 3, 4, 5])
        self.approx.y_data = np.array([2, 4, 6, 8, 10])
        
        result = self.approx.fit(1)
        self.assertTrue(result)
        
        pred = self.approx.predict(np.array([6, 7, 8]))
        expected = np.array([12, 14, 16])
        np.testing.assert_array_almost_equal(pred, expected)
        
        self.assertAlmostEqual(self.approx.r2, 1.0)
        self.assertAlmostEqual(self.approx.rmse, 0.0)
    
    def test_full_workflow_quadratic(self):
        """Полный рабочий процесс для квадратичной зависимости"""
        np.random.seed(42)
        x = np.linspace(0, 10, 20)
        y = 0.5 * x**2 - 2 * x + 3 + np.random.normal(0, 0.3, 20)
        
        self.approx.x_data = x
        self.approx.y_data = y
        
        self.approx.fit(2)
        
        coefs = self.approx.coefficients
        # Проверяем с большим допуском из-за шума
        self.assertAlmostEqual(coefs[0], 0.5, delta=0.5)
        self.assertAlmostEqual(coefs[1], -2.0, delta=0.5)
        self.assertAlmostEqual(coefs[2], 3.0, delta=1.0)
        
        self.assertGreater(self.approx.r2, 0.9)
    
    def test_compare_degrees_integration(self):
        """Интеграционный тест сравнения степеней"""
        np.random.seed(123)
        x = np.linspace(0, 10, 30)
        y = 2 * x**3 - 3 * x**2 + x + 5 + np.random.normal(0, 2, 30)
        
        self.approx.x_data = x
        self.approx.y_data = y
        
        self.approx.compare_degrees(5)
        
        self.assertIsNotNone(self.approx.coefficients)
        self.assertGreaterEqual(self.approx.degree, 1)
        self.assertLessEqual(self.approx.degree, 5)
        self.assertGreater(self.approx.r2, 0.5)
    
    def test_data_loading_and_training(self):
        """Взаимодействие загрузки данных и обучения"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("x,y\n1,3\n2,5\n3,7\n4,9\n5,11\n")
            temp_filename = f.name
        
        try:
            self.approx.load_data_from_csv(temp_filename)
            self.approx.fit(1)
            
            pred = self.approx.predict(np.array([6]))
            self.assertAlmostEqual(pred[0], 13.0, places=5)
            self.assertAlmostEqual(self.approx.r2, 1.0, places=5)
        finally:
            os.unlink(temp_filename)
    
    def test_multiple_operations_sequence(self):
        """Последовательность множественных операций"""
        self.approx.x_data = np.array([0, 1, 2, 3, 4])
        self.approx.y_data = np.array([0, 1, 4, 9, 16])
        
        self.approx.fit(1)
        r2_linear = self.approx.r2
        
        self.approx.fit(2)
        r2_quadratic = self.approx.r2
        
        self.assertGreater(r2_quadratic, r2_linear)
        
        pred = self.approx.predict(np.array([5]))
        self.assertAlmostEqual(pred[0], 25.0, places=5)
    
    def test_edge_case_single_point(self):
        """Граничный случай: одна точка данных"""
        self.approx.x_data = np.array([5])
        self.approx.y_data = np.array([10])
        
        result = self.approx.fit(1)
        self.assertFalse(result)
        self.assertIsNone(self.approx.coefficients)
    
    def test_edge_case_two_points(self):
        """Граничный случай: две точки данных"""
        self.approx.x_data = np.array([1, 2])
        self.approx.y_data = np.array([3, 5])
        
        result = self.approx.fit(1)
        self.assertTrue(result)
        self.assertAlmostEqual(self.approx.r2, 1.0)
        
        result = self.approx.fit(2)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()