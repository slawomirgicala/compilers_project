# -----------------------------------------------------------------------------
# calc.py
#
# A simple calculator with variables.   This is from O'Reilly's
# "Lex and Yacc", p. 63.
# -----------------------------------------------------------------------------

import sys
sys.path.insert(0, "../..")

reserved = {
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
    'POWER'
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

# Tokens

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

# Test it out
data = '''
   sin1 2^4 * sin^2 5 + 4.0^2.5 sin(23) .0 + 4 * 10.112121
   + -20 *2
 '''

# Give the lexer some input
lexer = build_lexer(data)


# Tokenize
def tokenize(lexer):
    lex_tokens = []
    while True:
        lex_token = lexer.token()
        if not lex_token:
            break
        lex_tokens.append((lex_token.type, lex_token.value))
    return lex_tokens


while True:
    tok = lexer.token()
    if not tok:
        break  # No more input
    print(tok)
