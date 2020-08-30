import sys
import math
sys.path.insert(0, "../..")

# LEXING

tokens = [
    'NUMBER',
    'PLUS',
    'MINUS',
    'TIMES',
    'DIVIDE',
    'POWER'
]

t_PLUS = r'\+'
t_MINUS = r'\-'
t_TIMES = r'\*'
t_DIVIDE = r'\/'
t_POWER = r'\^'
t_ignore = " \t"


def t_NUMBER(t):
    r'\d*\.\d+|\d+\.\d* | \d+'
    t.value = float(t.value)
    return t

#
# def t_INTEGER(t):
#     r'\d+'
#     t.value = int(t.value)
#     return t


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

# PARSING


def p_input(p):
    'input : expression'
    p[0] = p[1]


def p_expression(p):
    '''expression : NUMBER
                  | expression expression PLUS
                  | expression expression MINUS
                  | expression expression TIMES
                  | expression expression DIVIDE
                  | expression expression POWER
                  | expression MINUS'''
    if len(p) == 2:
        p[0] = p[1]
    elif len(p) == 3:
        p[0] = -p[1]
    else:
        if p[3] == '+':
            p[0] = p[1] + p[2]
        elif p[3] == '-':
            p[0] = p[1] - p[2]
        elif p[3] == '*':
            p[0] = p[1] * p[2]
        elif p[3] == '/':
            p[0] = p[1] / p[2]
        elif p[3] == '^':
            p[0] = p[1] ** p[2]


def p_error(p):
    if p:
        print("Syntax error at '%s'" % p.value)
    else:
        print("Syntax error at EOF")


def calculate_rpn(data):
    import yacc as yacc
    lexer = build_lexer(data)
    parser = yacc.yacc()
    return parser.parse(data)


# print(calculate_rpn('5 3 -'))
