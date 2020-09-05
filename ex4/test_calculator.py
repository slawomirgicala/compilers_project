import unittest
from ex4 import calc
from ex4.calc import Integer, Real, String, Bool


class CalculatorTests(unittest.TestCase):
    def tearDown(self):
        calc.names.clear()
        calc.functions.clear()

    def test_basic_operations(self):
        data = '4+4; 2-2; 1*10;2/5'
        self.assertEqual(calc.calculator_output(data), [Integer(8), Integer(0), Integer(10), Integer(0)])

    def test_exponentiation(self):
        data = '2^2^2^2'
        self.assertEqual(calc.calculator_output(data), [Integer(65536)])

    def test_math_functions(self):
        data = 'sin 0; cos 0; tg 0; sqrt 9; log 1; exp 0; asin 0; acos 1'
        self.assertEqual(calc.calculator_output(data), [Real(0.0), Real(1.0), Real(0.0),
                                                        Real(3.0), Real(0.0), Real(1.0),
                                                        Real(0.0), Real(0.0)])

    def test_assignment(self):
        data = 'int a = 2137; a'
        self.assertEqual(calc.calculator_output(data), [None, Integer(2137)])

    def test_logical_operations(self):
        data = '1 > 2; 18 >= 15; -10 < 9; -30 <= -40; 1 == 1; True != False; -1.1 > 19.09'
        self.assertEqual(calc.calculator_output(data), [Bool(False), Bool(True), Bool(True), Bool(False),
                                                        Bool(True), Bool(True), Bool(False)])

    def test_inversion(self):
        data = '1----1; 1-1; 1---1; -213.3; 3-3--3'
        self.assertEqual(calc.calculator_output(data), [Integer(2), Integer(0), Integer(0),
                                                        Real(-213.3), Integer(3)])

    def test_negation(self):
        data = '!True; !False; !(1==2); ! (1 > 2)'
        self.assertEqual(calc.calculator_output(data), [Bool(False), Bool(True), Bool(True), Bool(True)])

    def test_if_1(self):
        data = 'if (True){1+1;2+2}'
        self.assertEqual(calc.calculator_output(data), [[Integer(2), Integer(4)]])

    def test_if_2(self):
        data = 'if(sin 0 == 0.0){1+1; if (True){12; if (1+1 != 2){3}}}'
        self.assertEqual(calc.calculator_output(data), [[Integer(2), [Integer(12), None]]])

    def test_if_else(self):
        data = 'if(1==1){1}else{2}; if(1!=1){1}else{2}'
        self.assertEqual(calc.calculator_output(data), [[Integer(1)], [Integer(2)]])

    def test_while(self):
        data = 'int a = 1; while(a < 5){a = a + 1}; a'
        self.assertEqual(calc.calculator_output(data), [None, [[None], [None], [None], [None]], Integer(5)])

    def test_for(self):
        data = '''int i;
                  real b; 
                  for(i = 0; i < 3; i = i + 1){
                        b = 15.15;
                        b;
                  }'''
        self.assertEqual(calc.calculator_output(data), [None, None,
            [[None, Real(15.15)], [None, Real(15.15)], [None, Real(15.15)]]])

    def test_nested_statements(self):
        data = '''int x = 1;
                  int i;
                  int j;
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
        self.assertEqual(calc.calculator_output(data), [None, None, None,
                                                        [[None, [[None, [None]], [None, [None]]]],
                                                         [None, [[None, [None]], [None, [None]]]],
                                                         [None, [[None, [None]], [None, [None]]]]],
                                                        Integer(7)])

    def test_function_recursion(self):
        data = '''declare f(int i){
                    if (i == 0){
                        1;
                    } else{
                        if (i == 1){
                            1;
                        }else{
                            f(i-1) + f(i-2);
                        };
                    };
              };
              f(6)'''
        result = []
        calc.parse_output(calc.calculator_output(data), result)
        self.assertEqual(result, [13])
