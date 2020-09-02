import unittest
from ex4 import rpn_calc as calc


class RpnCalculatorTests(unittest.TestCase):
    def test_inversion(self):
        data = '12 -'
        self.assertEqual(calc.calculate_rpn(data), -12.0)

    def test_addition_1(self):
        data = '1 1 +'
        self.assertEqual(calc.calculate_rpn(data), 2.0)

    def test_addition_2(self):
        data = '1 2 3 4 5 6 7 8 9 + + + + + + + +'
        self.assertEqual(calc.calculate_rpn(data), 45.0)

    def test_addition_3(self):
        data = '1 2 + - 3.5 + 12.37 + 8.01 +'
        self.assertEqual(calc.calculate_rpn(data), 20.88)

    def test_subtraction_1(self):
        data = '13 13 -'
        self.assertEqual(calc.calculate_rpn(data), 0.0)

    def test_subtraction_2(self):
        data = '0 21.37 - 1124 - 9 9 - 9 9 - - -'
        self.assertEqual(calc.calculate_rpn(data), -1145.37)

    def test_multiplication_1(self):
        data = '7 5 *'
        self.assertEqual(calc.calculate_rpn(data), 35.0)

    def test_multiplication_2(self):
        data = '3 6 * 4 5 * *'
        self.assertEqual(calc.calculate_rpn(data), 360.0)

    def test_multiplication_3(self):
        data = '10 0.1 *'
        self.assertEqual(calc.calculate_rpn(data), 1.0)

    def test_division_1(self):
        data = '10 2 /'
        self.assertEqual(calc.calculate_rpn(data), 5.0)

    def test_division_2(self):
        data = '2 0.5 / 0.5 / 0.5 / 1 / 0.1 /'
        self.assertEqual(calc.calculate_rpn(data), 160.0)

    def test_requirements_1(self):
        data = '4 9 +'
        self.assertEqual(calc.calculate_rpn(data), 13.0)

    def test_requirements_2(self):
        data = '3 7 + 3 4 5 * + -'
        self.assertEqual(calc.calculate_rpn(data), -13.0)

    def test_requirements_3(self):
        data = '3 4 ^'
        self.assertEqual(calc.calculate_rpn(data), 81.0)

    def test_requirements_4(self):
        data = '15 7 1 1 + - / 3 * 2 1 1 + + -'
        self.assertEqual(calc.calculate_rpn(data), 5.0)
