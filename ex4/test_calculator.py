import unittest
from ex4 import calc


class CalculatorTests(unittest.TestCase):
    def test_basic_operations(self):
        data = '4+4; 2-2; 1*10;2/5'
        self.assertEqual(calc.calculator_output(data), [8, 0, 10, 0.4])

    def test_exponentiation(self):
        data = '2^2^2^2'
        self.assertEqual(calc.calculator_output(data), [65536])

    def test_math_functions(self):
        data = 'sin 0; cos 0; tg 0; sqrt 9; log 1; exp 0; asin 0; acos 1'
        self.assertEqual(calc.calculator_output(data), [0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 0.0])

    def test_assignment(self):
        data = 'a = 2137; a'
        self.assertEqual(calc.calculator_output(data), [None, 2137])

    def test_logical_operations(self):
        data = '1 > 2; 18 >= 15; -10 < 9; -30 <= -40; 1 == 1; True != False; -1.1 > 19.09'
        self.assertEqual(calc.calculator_output(data), [False, True, True, False, True, True, False])

    def test_inversion(self):
        data = '1----1; 1-1; 1---1; -213.3; 3-3--3'
        self.assertEqual(calc.calculator_output(data), [2, 0, 0, -213.3, 3])

    def test_negation(self):
        data = '!True; !False; !(1==2); ! (1 > 2)'
        self.assertEqual(calc.calculator_output(data), [False, True, True, True])

    def test_if_1(self):
        data = 'if (True){1+1;2+2}'
        self.assertEqual(calc.calculator_output(data), [[2, 4]])

    def test_if_2(self):
        data = 'if(sin 0 == 0){1+1; if (True){12; if (1+1 != 2){3}}}'
        self.assertEqual(calc.calculator_output(data), [[2, [12, None]]])

    def test_if_else(self):
        data = 'if(1==1){1}else{2}; if(1!=1){1}else{2}'
        self.assertEqual(calc.calculator_output(data), [[1], [2]])

    def test_while(self):
        data = 'a = 1; while(a < 5){a = a + 1}; a'
        self.assertEqual(calc.calculator_output(data), [None, [[None], [None], [None], [None]], 5])

    def test_for(self):
        data = '''for(i = 0; i < 3; i = i + 1){
                        b = 15.15;
                        b;
                  }'''
        self.assertEqual(calc.calculator_output(data), [[[None, 15.15], [None, 15.15], [None, 15.15]]])

    def test_nested_statements(self):
        data = '''x = 1;
                  for(i = 0; i < 3; i = i + 1){
                        j = 0;
                        while (j < 2){
                            j = j + 1;
                            if (1==1){
                                x = x + 1;
                            };
                        };
                  };
                  x;
                  '''
        self.assertEqual(calc.calculator_output(data), [None,
                                                        [[None, [[None, [None]], [None, [None]]]],
                                                         [None, [[None, [None]], [None, [None]]]],
                                                         [None, [[None, [None]], [None, [None]]]]],
                                                        7])
