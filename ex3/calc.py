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
    'for': 'FOR',
    'int': 'INT_INIT',
    'real': 'REAL_INIT',
    'bool': 'BOOL_INIT',
    'str': 'STR_INIT',
    'inttoreal': 'INTTOREAL',
    'realtoint': 'REALTOINT'
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
    'RBRACKET',
    'STRING'
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

def t_STRING(t):
    r'\".*\"'
    t.value = t.value[1:-1]
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


def p_statement_initialization(p):
    'statement : initialization'
    p[0] = p[1]


def p_statement_init_assign(p):
    'statement : init_assign'
    p[0] = p[1]


def p_statement_expr(p):
    'statement : expression'
    p[0] = [p[1]]


def p_assignment(p):
    'assignment : ID ASSIGN expression'
    p[0] = [(p[2], p[1], p[3])]


def p_initialize(p):
    '''initialization : INT_INIT ID
                      | REAL_INIT ID
                      | STR_INIT ID
                      | BOOL_INIT ID'''
    p[0] = [(p[1], p[2])]


def p_init_assign(p):
    '''init_assign : INT_INIT ID ASSIGN expression
                   | REAL_INIT ID ASSIGN expression
                   | BOOL_INIT ID ASSIGN expression
                   | STR_INIT ID ASSIGN expression'''
    p[0] = [('INIT_ASSIGN', p[1], p[2], p[4])]


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


def p_expression_string(p):
    'expression : STRING'
    p[0] = ('STRING', p[1])


def p_expression_inttoreal(p):
    'expression : INTTOREAL LPAREN expression RPAREN'
    p[0] = ('INTTOREAL', p[3])


def p_expression_realtoint(p):
    'expression : REALTOINT LPAREN expression RPAREN'
    p[0] = ('REALTOINT', p[3])


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


# calculator types


class Integer:
    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        return Integer(self.value + other.value)

    def __sub__(self, other):
        return Integer(self.value - other.value)

    def __mul__(self, other):
        return Integer(self.value * other.value)

    def __truediv__(self, other):
        return Integer(self.value // other.value)

    def __pow__(self, other, modulo=None):
        return Integer(self.value ** other.value)

    def __eq__(self, other):
        return Bool(self.value == other.value)

    def __ne__(self, other):
        return Bool(self.value != other.value)

    def __lt__(self, other):
        return Bool(self.value < other.value)

    def __le__(self, other):
        return Bool(self.value <= other.value)

    def __gt__(self, other):
        return Bool(self.value > other.value)

    def __ge__(self, other):
        return Bool(self.value >= other.value)

    def __str__(self):
        return str(self.value)


class Real:
    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        return Real(self.value + other.value)

    def __sub__(self, other):
        return Real(self.value - other.value)

    def __mul__(self, other):
        return Real(self.value * other.value)

    def __truediv__(self, other):
        return Real(self.value / other.value)

    def __pow__(self, other, modulo=None):
        return Real(self.value ** other.value)

    def __eq__(self, other):
        return Bool(self.value == other.value)

    def __ne__(self, other):
        return Bool(self.value != other.value)

    def __lt__(self, other):
        return Bool(self.value < other.value)

    def __le__(self, other):
        return Bool(self.value <= other.value)

    def __gt__(self, other):
        return Bool(self.value > other.value)

    def __ge__(self, other):
        return Bool(self.value >= other.value)

    def __str__(self):
        return str(self.value)


class Bool:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return Bool(self.value == other.value)

    def __ne__(self, other):
        return Bool(self.value != other.value)

    def __str__(self):
        return str(self.value)


class String:
    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        return String(self.value + other.value)

    def __eq__(self, other):
        return Bool(self.value == other.value)

    def __ne__(self, other):
        return Bool(self.value != other.value)

    def __lt__(self, other):
        return Bool(self.value < other.value)

    def __le__(self, other):
        return Bool(self.value <= other.value)

    def __gt__(self, other):
        return Bool(self.value > other.value)

    def __ge__(self, other):
        return Bool(self.value >= other.value)

    def __str__(self):
        return str(self.value)


# calculations utilities


def calculate(data):
    def get_value(data):
        type_name, value = data
        if type_name == 'INTEGER':
            return Integer(value)
        elif type_name == 'REAL':
            return Real(value)
        elif type_name == 'STRING':
            return String(value)
        elif type_name == 'BOOL':
            return Bool(value)

    def perform_assignment(data):
        _, name, expr = data
        value = calculate(expr)
        if name in names:
            if type(names[name]) is type(value):
                names[name] = value
                return None
            else:
                return "Wrong type assignment to variable " + name
        else:
            return "Variable not initialized"

    def get_assigned_value(data):
        _, name = data
        if name in names:
            if names[name].value is None:
                return "No value assigned to variable"
            else:
                return names[name]
        else:
            return "Variable " + name + " not initialized"

    def skip_grouping(data):
        _, expression = data
        return calculate(expression)

    def inverse(data):
        _, value = data
        value = calculate(value)
        if isinstance(value, Integer) or isinstance(value, Real):
            value.value = -value.value
            return value
        else:
            return "Cannot use minus on that type"

    def negate(data):
        _, value = data
        value = calculate(value)
        if isinstance(value, Bool):
            value.value = not value.value
            return value
        else:
            return "Only bool can be negated"

    def addition(data):
        _, left, right = data
        left = calculate(left)
        right = calculate(right)
        if type(left) is type(right):
            if isinstance(left, Bool):
                return "Cannot add bool"
            else:
                return left + right
        else:
            return "Incompatible types"

    def subtraction(data):
        _, left, right = data
        left = calculate(left)
        right = calculate(right)
        if type(left) is type(right):
            if isinstance(left, Bool) or isinstance(left, String):
                return "Cannot subtract bool and string"
            else:
                return left - right
        else:
            return "Incompatible types"

    def multiplication(data):
        _, left, right = data
        left = calculate(left)
        right = calculate(right)
        if type(left) is type(right):
            if isinstance(left, Bool) or isinstance(left, String):
                return "Cannot multiply bool and string"
            else:
                return left * right
        else:
            return "Incompatible types"

    def division(data):
        _, left, right = data
        left = calculate(left)
        right = calculate(right)
        if type(left) is type(right):
            if isinstance(left, Bool) or isinstance(left, String):
                return "Cannot divide bool and string"
            else:
                return left / right
        else:
            return "Incompatible types"

    def exponentiation(data):
        _, left, right = data
        left = calculate(left)
        right = calculate(right)
        if type(left) is type(right):
            if isinstance(left, Bool) or isinstance(left, String):
                return "Cannot power bool and string"
            else:
                return left ** right
        else:
            return "Incompatible types"

    def equality(data):
        _, left, right = data
        left = calculate(left)
        right = calculate(right)
        if type(left) is type(right):
            return left == right
        else:
            return "Incompatible types"

    def not_equality(data):
        _, left, right = data
        left = calculate(left)
        right = calculate(right)
        if type(left) is type(right):
            return left != right
        else:
            return "Incompatible types"

    def greater(data):
        _, left, right = data
        left = calculate(left)
        right = calculate(right)
        if type(left) is type(right):
            if isinstance(left, Bool):
                return "Bool values have no order"
            else:
                return left > right
        else:
            return "Incompatible types"

    def greater_or_equal(data):
        _, left, right = data
        left = calculate(left)
        right = calculate(right)
        if type(left) is type(right):
            if isinstance(left, Bool):
                return "Bool values have no order"
            else:
                return left >= right
        else:
            return "Incompatible types"

    def less(data):
        _, left, right = data
        left = calculate(left)
        right = calculate(right)
        if type(left) is type(right):
            if isinstance(left, Bool):
                return "Bool values have no order"
            else:
                return left < right
        else:
            return "Incompatible types"

    def less_or_equal(data):
        _, left, right = data
        left = calculate(left)
        right = calculate(right)
        if type(left) is type(right):
            if isinstance(left, Bool):
                return "Bool values have no order"
            else:
                return left <= right
        else:
            return "Incompatible types"

    def math_function(data):
        f_name, arg = data
        arg = calculate(arg)
        if isinstance(arg, String) and isinstance(arg, Bool):
            return "Cannot use math functions on strings and bools"
        if f_name == 'sin':
            return Real(math.sin(arg.value))
        elif f_name == 'cos':
            return Real(math.cos(arg.value))
        elif f_name == 'tg':
            return Real(math.tan(arg.value))
        elif f_name == 'ctg':
            return Real(1/math.tan(arg.value))
        elif f_name == 'sqrt':
            return Real(math.sqrt(arg.value))
        elif f_name == 'exp':
            return Real(math.exp(arg.value))
        elif f_name == 'log':
            return Real(math.log(arg.value))
        elif f_name == 'asin':
            return Real(math.asin(arg.value))
        elif f_name == 'atg':
            return Real(math.atan(arg.value))
        elif f_name == 'acos':
            return Real(math.acos(arg.value))

    def if_statement(data):
        _, condition, if_instructions = data
        condition = calculate(condition)
        if not isinstance(condition, Bool):
            return "Not bool in if statement"
        if condition.value:
            return calculate_statements(if_instructions)
        return None

    def if_else_statement(data):
        _, condition, if_instructions, else_instructions = data
        condition = calculate(condition)
        if not isinstance(condition, Bool):
            return "Not bool in if statement"
        if condition.value:
            return calculate_statements(if_instructions)
        else:
            return calculate_statements(else_instructions)

    def while_statement(data):
        _, condition, instructions = data
        results = []
        cond = calculate(condition)
        if not isinstance(cond, Bool):
            return "Not bool in while statement"
        while cond.value:
            results.append(calculate_statements(instructions))
            cond = calculate(condition)
            if not isinstance(cond, Bool):
                return "Not bool in while statement"
        return results

    def for_statement(data):
        _, assignment, condition, expression, instructions = data
        results = []
        calculate_statements(assignment)
        cond = calculate(condition)
        if not isinstance(cond, Bool):
            return "Not bool in for statement"
        while cond.value:
            results.append(calculate_statements(instructions))
            calculate_statements(expression)
            cond = calculate(condition)
            if not isinstance(cond, Bool):
                return "Not bool in for statement"
        return results

    def initialization(data):
        var_type, name = data
        if name in names:
            return "Already initialized"
        else:
            if var_type == 'int':
                names[name] = Integer(None)
            elif var_type == 'real':
                names[name] = Real(None)
            elif var_type == 'bool':
                names[name] = Bool(None)
            elif var_type == 'str':
                names[name] = String(None)
            return None

    def initialization_assignment(data):
        _, var_type, name, expr = data
        if name in names:
            return "Variable already initialized"
        if var_type == 'int':
            var_type = Integer(None)
        elif var_type == 'real':
            var_type = Real(None)
        elif var_type == 'bool':
            var_type = Bool(None)
        elif var_type == 'str':
            var_type = String(None)
        val = calculate(expr)
        if type(var_type) is not type(val):
            return "Incompatible types"
        names[name] = val
        return None

    def int_to_real_conversion(data):
        _, expr = data
        to_cast = calculate(expr)
        if isinstance(to_cast, Integer):
            return Real(float(to_cast.value))
        else:
            return "Only integer can be cast to real by inttoreal"

    def real_to_int_conversion(data):
        _, expr = data
        to_cast = calculate(expr)
        if isinstance(to_cast, Real):
            return Integer(int(to_cast.value))
        else:
            return "Only real can be cast to int by realtoint"

    if data[0] in ('INTEGER', 'REAL', 'BOOL', 'STRING'):
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
    elif data[0] in ['int', 'bool', 'real', 'str']:
        return initialization(data)
    elif data[0] == 'INIT_ASSIGN':
        return initialization_assignment(data)
    elif data[0] == 'INTTOREAL':
        return int_to_real_conversion(data)
    elif data[0] == 'REALTOINT':
        return real_to_int_conversion(data)


def calculate_statements(statements):
    results = []
    for statement in statements:
        results.append(calculate(statement))
    return results


def calculator_output(data):
    import yacc as yacc
    lexer = build_lexer(data)
    parser = yacc.yacc()
    statements = parser.parse(data)
    return calculate_statements(statements)


class Order:
    count = 0

    @classmethod
    def next(cls):
        cls.count = cls.count + 1
        return '[' + str(cls.count) + ']' + '\n'


def create_node(data, parent):
    from anytree import Node

    def create_primitive_node(data, parent):
        symbol, value = data
        return Node(Order.next() + symbol + '\n' + str(value), parent)

    def create_math_function_node(data, parent):
        symbol, expr = data
        math_function_node = Node(Order.next() + math_reserved.get(symbol), parent)
        create_node(expr, math_function_node)
        return math_function_node

    def create_assignment_node(data, parent):
        symbol, name, expression = data
        assignment_node = Node(Order.next() + symbol, parent)
        Node(Order.next() + 'ID' + '\n' + name, assignment_node)
        create_node(expression, assignment_node)
        return assignment_node

    def create_initialization_node(data, parent):
        symbol, name = data
        initialization_node = Node(Order.next() + symbol, parent)
        Node(Order.next() + 'ID' + '\n' + name, initialization_node)
        return initialization_node

    def create_init_assign_node(data, parent):
        init_assign, var_type, name, expression = data
        init_assign_node = Node(Order.next() + 'INIT_AND_ASSIGN', parent)
        create_initialization_node((var_type, name), init_assign_node)
        create_node(expression, init_assign_node)

    def create_variable_node(data, parent):
        symbol, name = data
        variable_node = Node(Order.next() + symbol + '\n' + name, parent)
        return variable_node

    def create_grouping_node(data, parent):
        symbol, expr = data
        grouping_node = Node(Order.next() + symbol + '\n' + '( )', parent)
        create_node(expr, grouping_node)
        return grouping_node

    def create_binary_operation_node(data, parent):
        symbol, expr1, expr2 = data
        binary_operation_node = Node(Order.next() + symbol, parent)
        create_node(expr1, binary_operation_node)
        create_node(expr2, binary_operation_node)
        return binary_operation_node

    def create_unary_operation_node(data, parent):
        symbol, expr = data
        unary_operation_node = Node(Order.next() + 'NEGATION', parent)
        create_node(expr, unary_operation_node)
        return unary_operation_node

    def create_if_node(data, parent):
        symbol, condition, instructions = data
        if_node = Node(Order.next() + symbol, parent)
        condition_node = Node(Order.next() + 'CONDITION', if_node)
        create_node(condition, condition_node)
        instructions_node = Node(Order.next() + 'INSTRUCTIONS', if_node)
        for i, instruction in enumerate(instructions, 1):
            instruction_node = Node(Order.next() + 'INSTRUCTION_' + str(i), instructions_node)
            create_node(instruction, instruction_node)
        return if_node

    def create_if_else_node(data, parent):
        symbol, condition, if_instructions, else_instructions = data
        if_else_node = Node(Order.next() + symbol, parent)
        condition_node = Node(Order.next() + 'CONDITION', if_else_node)
        create_node(condition, condition_node)
        if_instructions_node = Node(Order.next() + 'IF_INSTRUCTIONS', if_else_node)
        for i, instruction in enumerate(if_instructions, 1):
            instruction_node = Node(Order.next() + 'INSTRUCTION_' + str(i), if_instructions_node)
            create_node(instruction, instruction_node)
        else_instructions_node = Node(Order.next() + 'ELSE_INSTRUCTIONS', if_else_node)
        for i, instruction in enumerate(else_instructions, 1):
            instruction_node = Node(Order.next() + 'INSTRUCTION_' + str(i), else_instructions_node)
            create_node(instruction, instruction_node)
        return if_else_node

    def create_while_node(data, parent):
        symbol, condition, instructions = data
        while_node = Node(Order.next() + symbol, parent)
        condition_node = Node(Order.next() + 'CONDITION', while_node)
        create_node(condition, condition_node)
        instructions_node = Node(Order.next() + 'INSTRUCTIONS', while_node)
        for i, instruction in enumerate(instructions, 1):
            instruction_node = Node(Order.next() + 'INSTRUCTION_' + str(i), instructions_node)
            create_node(instruction, instruction_node)
        return while_node

    def create_for_node(data, parent):
        symbol, assignment, condition, expr, instructions = data
        for_node = Node(Order.next() + symbol, parent)
        assignment_node = Node(Order.next() + 'ASSIGNMENT', for_node)
        create_node(assignment[0], assignment_node)
        condition_node = Node(Order.next() + 'CONDITION', for_node)
        create_node(condition, condition_node)
        expr_node = Node(Order.next() + 'EXPRESSIONS', for_node)
        for i, e in enumerate(expr, 1):
            exp = Node(Order.next() + 'EXPRESSION_' + str(i), expr_node)
            create_node(e, exp)
        instructions_node = Node(Order.next() + 'INSTRUCTIONS', for_node)
        for i, instruction in enumerate(instructions, 1):
            instr = Node(Order.next() + 'INSTRUCTION_' + str(i), instructions_node)
            create_node(instruction, instr)

    def create_int_to_real_node(data, parent):
        symbol, expr = data
        int_to_real_node = Node(Order.next() + 'INTTOREAL', parent)
        create_node(expr, int_to_real_node)
        return int_to_real_node

    def create_real_to_int_node(data, parent):
        symbol, expr = data
        real_to_int_node = Node(Order.next() + 'REALTOINT', parent)
        create_node(expr, real_to_int_node)
        return real_to_int_node

    if data[0] in ('INTEGER', 'REAL', 'BOOL', 'STRING'):
        return create_primitive_node(data, parent)
    elif data[0] in list(math_reserved.keys()):
        return create_math_function_node(data, parent)
    elif data[0] == '=':
        return create_assignment_node(data, parent)
    elif data[0] == 'ID':
        return create_variable_node(data, parent)
    elif data[0] == 'GROUP':
        return create_grouping_node(data, parent)
    elif data[0] in ['+', '-', '*', '/', '^', '>', '>=', '<', "<=", "==", "!="]:
        return create_binary_operation_node(data, parent)
    elif data[0] in ['!', 'NEGATE']:
        return create_unary_operation_node(data, parent)
    elif data[0] == 'IF':
        return create_if_node(data, parent)
    elif data[0] == 'IFELSE':
        return create_if_else_node(data, parent)
    elif data[0] == 'WHILE':
        return create_while_node(data, parent)
    elif data[0] == 'FOR':
        return create_for_node(data, parent)
    elif data[0] in ['INT_INIT', 'REAL_INIT', 'BOOL_INIT', 'STR_INIT']:
        return create_initialization_node(data, parent)
    elif data[0] == 'INIT_ASSIGN':
        return create_init_assign_node(data, parent)
    elif data[0] == 'INTTOREAL':
        return create_int_to_real_node(data, parent)
    elif data[0] == 'REALTOINT':
        return create_real_to_int_node(data, parent)


def build_ast(data):
    import yacc as yacc
    from anytree import Node
    lexer = build_lexer(data)
    parser = yacc.yacc()
    statements = parser.parse(data)
    program = Node("PROGRAM")
    program.id = '1'
    for i, statement in enumerate(statements, 1):
        statement_node = Node("STATEMENT_" + str(i), program)
        create_node(statement, statement_node)
    return program


def draw_ast(data, file):
    from anytree.exporter import DotExporter
    DotExporter(build_ast(data)).to_picture(file)

#
# data = 'a = sin (-143 + 12 ^ 2); a; a + 1; 1<2+1; if(1<2){2-1}'
#data = '1'
#data = 'int i; real j = 0.0;str hi = "hi"; hi;int a = sin 0;a; if(1>2){2-1}else{69}; a=1; while(a < 10){a = a+1}; a; for(i = 0;i<5;i = i+1){i}'
# data = '!True'
# data = 'a = True; a; b = 1; while(b==1){a = False;b=2}; a; b; sin 1; 2 ^ 4; !True; -2'


def printer(data):
    for d in data:
        if isinstance(d, list):
            printer(d)
        else:
            if d is not None:
                print(d)



from anytree.exporter import DotExporter

# DotExporter(build_ast(statements)).to_picture("ast/example.png")
if __name__ == '__main__':
    import yacc as yacc
    data = 'int i; real j = 0.0;str hi = "hi"; hi;int a = sin 0;a;' \
           ' if(1>2){2-1}else{69}; a=1; while(a < 10){a = a+1}; a; for(i = 0;i<5;i = i+1){i}'
    lexer = build_lexer(data)
    parser = yacc.yacc()
    statements = parser.parse(data)
    print(statements)
    printer(calculate_statements(statements))
