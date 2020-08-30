import sys
import math
sys.path.insert(0, "../..")



# LEXING

math_reserved = {
    'sin': 'SIN',
    'cos': 'COS',
    'tg': 'TAN',
    'ctg': 'COT',
    'sqrt': 'SQRT',
    'log': 'LOG',
    'exp': 'EXP',
    'asin': 'ASIN',
    'acos': 'ACOS',
    'atg': 'ATAN',
    'actg': 'ACOT'
}

reserved = {
    **math_reserved,
    'if': 'IF',
    'else': 'ELSE',
    'while': 'WHILE',
    'for': 'FOR'
}

tokens = [
    'INTEGER',
    'REAL',
    'PLUS',
    'MINUS',
    'TIMES',
    'DIVIDE',
    'LPAREN',
    'RPAREN',
    'ID',
    'ASSIGN',
    'EQUALS',
    'POWER',
    'NOTEQUALS',
    'GT',
    'GTE',
    'LT',
    'LTE',
    'NOT',
    'BOOL',
    'SEMICOLON',
    'LBRACKET',
    'RBRACKET'
] + list(reserved.values())

t_PLUS = r'\+'
t_MINUS = r'\-'
t_TIMES = r'\*'
t_DIVIDE = r'\/'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_ASSIGN = r'\='
t_EQUALS = r'\=\='
t_POWER = r'\^'
t_NOTEQUALS = r'\!\='
t_GT = r'\>'
t_GTE = r'\>\='
t_LT = r'\<'
t_LTE = r'\<\='
t_NOT = r'\!'
t_SEMICOLON = r'\;'
t_LBRACKET = r'\{'
t_RBRACKET = r'\}'


def t_BOOL(t):
    r'(True|False)'
    if t.value == 'True':
        t.value = True
    else:
        t.value = False
    return t

def t_ID(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    t.type = reserved.get(t.value, 'ID')
    return t

def t_REAL(t):
    r'\d*\.\d+|\d+\.\d*'
    t.value = float(t.value)
    return t

def t_INTEGER(t):
    r'\d+'
    t.value = int(t.value)
    return t

t_ignore = " \t"

def t_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")

def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)

# Build the lexer
def build_lexer(data):
    import lex as lex
    lexer = lex.lex()
    lexer.input(data)
    return lexer

# Tokenize
def tokenize(lexer):
    lex_tokens = []
    while True:
        lex_token = lexer.token()
        if not lex_token:
            break
        lex_tokens.append((lex_token.type, lex_token.value))
    return lex_tokens


# PARSING

names = {}

precedence = (
    ('nonassoc', 'GT', 'GTE', 'LT', 'LTE', 'EQUALS', 'NOTEQUALS'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'TIMES', 'DIVIDE'),
    ('left', 'SIN', 'COS', 'TAN', 'COT', 'SQRT', 'LOG', 'EXP', 'ASIN', 'ACOS', 'ATAN', 'ACOT'),
    ('right', 'POWER'),
    ('right', 'NOT'),
    ('right', 'UMINUS')
)


def p_multiple_statement(p):
    'statement : statement SEMICOLON statement'
    p[0] = p[1] + p[3]


def p_statement_semicolon(p):
    'statement : statement SEMICOLON'
    p[0] = p[1]


def p_statement_assignment(p):
    'statement : assignment'
    p[0] = p[1]


def p_statement_expr(p):
    'statement : expression'
    p[0] = [p[1]]


def p_assignment(p):
    'assignment : ID ASSIGN expression'
    p[0] = [(p[2], p[1], p[3])]


def p_expression_binop_arithmetic(p):
    '''expression : expression PLUS expression
                  | expression MINUS expression
                  | expression TIMES expression
                  | expression DIVIDE expression
                  | expression POWER expression'''
    p[0] = (p[2], p[1], p[3])


def p_expression_relation(p):
    'expression : relation'
    p[0] = p[1]


def p_expression_binop_logic(p):
    '''relation : expression EQUALS expression
                  | expression NOTEQUALS expression
                  | expression GT expression
                  | expression GTE expression
                  | expression LT expression
                  | expression LTE expression'''
    p[0] = (p[2], p[1], p[3])


def p_expression_function(p):
    '''expression : SIN expression
                  | COS expression
                  | TAN expression
                  | COT expression
                  | SQRT expression
                  | EXP expression
                  | LOG expression
                  | ASIN expression
                  | ACOS expression
                  | ATAN expression
                  | ACOT expression'''
    p[0] = (p[1], p[2])


def p_expression_uminus(p):
    "expression : MINUS expression %prec UMINUS"
    p[0] = ('NEGATE', p[2])


def p_expression_not(p):
    "expression : NOT expression %prec NOT"
    p[0] = (p[1], p[2])


def p_expression_group(p):
    "expression : LPAREN expression RPAREN"
    p[0] = ('GROUP', p[2])


def p_expression_integer(p):
    '''expression : INTEGER'''
    p[0] = ('INTEGER', p[1])


def p_expression_real(p):
    '''expression : REAL'''
    p[0] = ('REAL', p[1])


def p_expression_bool(p):
    '''relation : BOOL'''
    p[0] = ('BOOL', p[1])


def p_expression_id(p):
    "expression : ID"
    p[0] = ('ID', p[1])


def p_if_statement(p):
    "statement : IF LPAREN relation RPAREN LBRACKET statement RBRACKET"
    p[0] = [('IF', p[3], p[6])]


def p_if_else_statement(p):
    "statement : IF LPAREN relation RPAREN LBRACKET statement RBRACKET ELSE LBRACKET statement RBRACKET"
    p[0] = [('IFELSE', p[3], p[6], p[10])]


def p_while_statement(p):
    "statement : WHILE LPAREN relation RPAREN LBRACKET statement RBRACKET"
    p[0] = [('WHILE', p[3], p[6])]

def p_for_statement(p):
    "statement : FOR LPAREN assignment SEMICOLON relation SEMICOLON statement RPAREN LBRACKET statement RBRACKET"
    p[0] = [('FOR', p[3], p[5], p[7], p[10])]


def p_error(p):
    if p:
        print("Syntax error at '%s'" % p.value)
    else:
        print("Syntax error at EOF")


def calculate(data):
    def get_value(data):
        _, value = data
        return value

    def perform_assignment(data):
        _, name, value = data
        value = calculate(value)
        names[name] = value
        return None

    def get_assigned_value(data):
        _, name = data
        return names[name]

    def skip_grouping(data):
        _, expression = data
        return calculate(expression)

    def inverse(data):
        _, value = data
        value = calculate(value)
        return -value

    def negate(data):
        _, value = data
        value = calculate(value)
        return not value

    def addition(data):
        _, left, right = data
        left = calculate(left)
        right = calculate(right)
        return left + right

    def subtraction(data):
        _, left, right = data
        left = calculate(left)
        right = calculate(right)
        return left - right

    def multiplication(data):
        _, left, right = data
        left = calculate(left)
        right = calculate(right)
        return left * right

    def division(data):
        _, left, right = data
        left = calculate(left)
        right = calculate(right)
        return left / right

    def exponentiation(data):
        _, left, right = data
        left = calculate(left)
        right = calculate(right)
        return left ** right

    def equality(data):
        _, left, right = data
        left = calculate(left)
        right = calculate(right)
        return left == right

    def not_equality(data):
        _, left, right = data
        left = calculate(left)
        right = calculate(right)
        return left != right

    def greater(data):
        _, left, right = data
        left = calculate(left)
        right = calculate(right)
        return left > right

    def greater_or_equal(data):
        _, left, right = data
        left = calculate(left)
        right = calculate(right)
        return left >= right

    def less(data):
        _, left, right = data
        left = calculate(left)
        right = calculate(right)
        return left < right

    def less_or_equal(data):
        _, left, right = data
        left = calculate(left)
        right = calculate(right)
        return left <= right

    def math_function(data):
        f_name, arg = data
        arg = calculate(arg)
        if f_name == 'sin':
            return math.sin(arg)
        elif f_name == 'cos':
            return math.cos(arg)
        elif f_name == 'tg':
            return math.tan(arg)
        elif f_name == 'ctg':
            return 1/math.tan(arg)
        elif f_name == 'sqrt':
            return math.sqrt(arg)
        elif f_name == 'exp':
            return math.exp(arg)
        elif f_name == 'log':
            return math.log(arg)
        elif f_name == 'asin':
            return math.asin(arg)
        elif f_name == 'atg':
            return math.atan(arg)
        elif f_name == 'acos':
            return math.acos(arg)

    def if_statement(data):
        _, condition, if_instructions = data
        condition = calculate(condition)
        if condition:
            return calculate_statements(if_instructions)
        return None

    def if_else_statement(data):
        _, condition, if_instructions, else_instructions = data
        condition = calculate(condition)
        if condition:
            return calculate_statements(if_instructions)
        else:
            return calculate_statements(else_instructions)

    def while_statement(data):
        _, condition, instructions = data
        results = []
        while calculate(condition):
            results.append(calculate_statements(instructions))
        return results

    def for_statement(data):
        _, assignment, condition, expression, instructions = data
        results = []
        calculate_statements(assignment)
        while calculate(condition):
            results.append(calculate_statements(instructions))
            calculate_statements(expression)
        return results

    if data[0] in ('INTEGER', 'REAL', 'BOOL'):
        return get_value(data)
    elif data[0] in list(math_reserved.keys()):
        return math_function(data)
    elif data[0] == '=':
        return perform_assignment(data)
    elif data[0] == 'ID':
        return get_assigned_value(data)
    elif data[0] == 'GROUP':
        return skip_grouping(data)
    elif data[0] == '+':
        return addition(data)
    elif data[0] == '-':
        return subtraction(data)
    elif data[0] == '*':
        return multiplication(data)
    elif data[0] == '/':
        return division(data)
    elif data[0] == '^':
        return exponentiation(data)
    elif data[0] == '>':
        return greater(data)
    elif data[0] == '>=':
        return greater_or_equal(data)
    elif data[0] == '<':
        return less(data)
    elif data[0] == '<=':
        return less_or_equal(data)
    elif data[0] == '==':
        return equality(data)
    elif data[0] == '!=':
        return not_equality(data)
    elif data[0] == '!':
        return negate(data)
    elif data[0] == 'NEGATE':
        return inverse(data)
    elif data[0] == 'IF':
        return if_statement(data)
    elif data[0] == 'IFELSE':
        return if_else_statement(data)
    elif data[0] == 'WHILE':
        return while_statement(data)
    elif data[0] == 'FOR':
        return for_statement(data)


def calculate_statements(statements):
    results = []
    for statement in statements:
        results.append(calculate(statement))
    return results


import yacc as yacc


def calculator_output(data):
    lexer = build_lexer(data)
    parser = yacc.yacc()
    statements = parser.parse(data)
    return calculate_statements(statements)



import yacc as yacc

#data = 'a = sin (-143 + 12 ^ 2); a; a + 1; 1<2+1; if(1<2){2-1}'
# data = '1'
# #data = 'a = sin 0; -;1;a; if(1>2){2-1}else{69}; a=1; while(a < 10){a = a+1}; a; for(i = 0;i<5;i = i+1){i}'
# data = '!True'
#
# lexer = build_lexer(data)
# parser = yacc.yacc()
# statements = parser.parse(data)
# print(statements)
# print(calculate_statements(statements))
