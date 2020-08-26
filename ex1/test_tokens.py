import unittest
from ex1 import calc


class TokenTests(unittest.TestCase):
    def test_integer_1(self):
        data = '4'
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('INTEGER', 4)])

    def test_integer_2(self):
        data = '2137'
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('INTEGER', 2137)])

    def test_integer_3(self):
        data = '0003'
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('INTEGER', 3)])

    def test_real_1(self):
        data = '1.01'
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('REAL', 1.01)])

    def test_real_2(self):
        data = '.3333'
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('REAL', 0.3333)])

    def test_real_3(self):
        data = '212.'
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('REAL', 212.0)])

    def test_plus(self):
        data = '1+2'
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('INTEGER', 1), ('PLUS', '+'), ('INTEGER', 2)])

    def test_minus_1(self):
        data = '1  -2'
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('INTEGER', 1), ('MINUS', '-'), ('INTEGER', 2)])

    def test_minus_2(self):
        data = '1+-2'
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('INTEGER', 1), ('PLUS', '+'), ('MINUS', '-'), ('INTEGER', 2)])

    def test_times(self):
        data = '21*32'
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('INTEGER', 21), ('TIMES', '*'), ('INTEGER', 32)])

    def test_divide(self):
        data = '1./.2'
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('REAL', 1.0), ('DIVIDE', '/'), ('REAL', 0.2)])

    def test_paren(self):
        data = '((1)-(-.2))'
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('LPAREN', '('), ('LPAREN', '('), ('INTEGER', 1),
                                  ('RPAREN', ')'), ('MINUS', '-'), ('LPAREN', '('),
                                  ('MINUS', '-'), ('REAL', 0.2), ('RPAREN', ')'),
                                  ('RPAREN', ')')])

    def test_id(self):
        data = 'hello my name is slawek'
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('ID', 'hello'), ('ID', 'my'), ('ID', 'name'),
                                  ('ID', 'is'), ('ID', 'slawek')])

    def test_assign(self):
        data = 'x = 1'
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('ID', 'x'), ('ASSIGN', '='), ('INTEGER', 1)])

    def test_equals(self):
        data = '1 == 1 = 1 === 1 ==== 1'
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('INTEGER', 1), ('EQUALS', '=='), ('INTEGER', 1),
                                  ('ASSIGN', '='), ('INTEGER', 1), ('EQUALS', '=='),
                                  ('ASSIGN', '='), ('INTEGER', 1), ('EQUALS', '=='),
                                  ('EQUALS', '=='), ('INTEGER', 1)])

    def test_power(self):
        data = '1^2'
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('INTEGER', 1), ('POWER', '^'), ('INTEGER', 2)])

    def test_sinus_1(self):
        data = 'sin'
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('SIN', 'sin')])

    def test_sinus_2(self):
        data = 'sin 2'
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('SIN', 'sin'), ('INTEGER', 2)])

    def test_sinus_3(self):
        data = 'sin(3)'
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('SIN', 'sin'), ('LPAREN', '('), ('INTEGER', 3),
                                  ('RPAREN', ')')])

    def test_sinus_4(self):
        data = 'sin4'
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('ID', 'sin4')])

    def test_sinus_5(self):
        data = 'sinsin12'
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('ID', 'sinsin12')])

    def test_sinus_6(self):
        data = 'sin x'
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('SIN', 'sin'), ('ID', 'x')])

    def test_expression_1(self):
        data = '''3 + 4 * 10
                  + -20 *2'''
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('INTEGER', 3), ('PLUS', '+'), ('INTEGER', 4),
                                  ('TIMES', '*'), ('INTEGER', 10), ('PLUS', '+'),
                                  ('MINUS', '-'), ('INTEGER', 20), ('TIMES', '*'),
                                  ('INTEGER', 2)])

    def test_expression_2(self):
        data = 'asin sin 1+2 tgctg3 sqrt 2 exp e'
        lexer = calc.build_lexer(data)
        tokens = calc.tokenize(lexer)
        self.assertEqual(tokens, [('ASIN', 'asin'), ('SIN', 'sin'), ('INTEGER', 1),
                                  ('PLUS', '+'), ('INTEGER', 2), ('ID', 'tgctg3'),
                                  ('SQRT', 'sqrt'), ('INTEGER', 2), ('EXP', 'exp'),
                                  ('ID', 'e')])


if __name__ == '__main__':
    unittest.main()
