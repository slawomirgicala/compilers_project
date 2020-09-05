import unittest
from ex5 import calc as calc


def parse_output(data, container):
    for d in data:
        if isinstance(d, list):
            parse_output(d, container)
        else:
            if d is not None:
                container.append(d.value)


class VariableRangeTests(unittest.TestCase):
    def tearDown(self):
        calc.names.clear()

    def test_redeclaration(self):
        data = '''int i = 0;
                  if (1 == 1){
                    int i = 1;
                    i;
                  };
                  i;'''
        results = []
        parse_output(calc.calculator_output(data), results)
        self.assertEqual(results, [1, 0])

    def test_reassignment(self):
        data = '''int i = 0;
                  if (1 == 1){
                    i = 1;
                    i;
                  };
                  i;'''
        results = []
        parse_output(calc.calculator_output(data), results)
        self.assertEqual(results, [1, 1])

    def test_variable_shadowing(self):
        data = '''int i = 0;
                  if (i == 0){
                        int i = 1;
                        if (i == 1){
                            int i = 2;
                            if (i == 2){
                                i = 13;
                                i;
                            };
                        };
                  };
                  i;'''
        results = []
        parse_output(calc.calculator_output(data), results)
        self.assertEqual(results, [13, 0])
