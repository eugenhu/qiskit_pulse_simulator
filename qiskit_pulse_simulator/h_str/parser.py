from .lex import lex
from .yacc import yacc


tokens = (
    'NUMBER',
    'PLUS',
    'MINUS',
    'TIMES',
    'DIVIDE',
    'L_PAREN',
    'R_PAREN',
    'L_BRACKET',
    'R_BRACKET',
    'MATH_START',
    'MATH_END',
    'COMMA',
    'TEXT',
    'BOUND',
    'SUM',
    'DOUBLE_PIPE',
)


states = (
    ('math', 'exclusive'),
)


t_ANY_PLUS   = r'\+'
t_ANY_MINUS  = r'-'
t_ANY_TIMES  = r'\*'
t_ANY_DIVIDE = r'/'
t_ANY_L_PAREN = r'\('
t_ANY_R_PAREN = r'\)'

t_TEXT = r'[a-zA-Z][a-zA-Z0-9_]*'
t_math_BOUND = t_TEXT

t_DOUBLE_PIPE   = r'\|\|'

t_SUM = r'_SUM'
t_L_BRACKET   = r'\['
t_R_BRACKET   = r'\]'
t_COMMA  = r','

t_ANY_ignore = ' \t'


def t_begin_math(t):
    r'{'
    t.lexer.push_state('math')
    t.type = 'MATH_START'
    return t


def t_math_end(t):
    r'}'
    t.lexer.pop_state()
    t.type = 'MATH_END'
    return t

    
def t_ANY_NUMBER(t):
    r'\d+\.?\d*(?:e\-?\d+)?|\.\d*(?:e\-?\d+)?'

    if '.' in t.value or 'e' in t.value:
        t.value = float(t.value)
    else:
        t.value = int(t.value)
    return t


def t_ANY_error(t):
    return t


precedence = (
    ('left', 'PLUS', 'MINUS'),
    ('left', 'TIMES', 'DIVIDE'),
    ('left', 'POS', 'NEG'),
    ('left', 'CONCAT'),
)


def p_expression_plus(p):
    'expression : expression PLUS expression'
    p[0] = ('add', p[1], p[3])

    
def p_expression_minus(p):
    'expression : expression MINUS expression'
    p[0] = ('minus', p[1], p[3])

    
def p_expression_pos(p):
    'expression : PLUS expression %prec POS'
    p[0] = p[2]

    
def p_expression_neg(p):
    'expression : MINUS expression %prec NEG'
    p[0] = ('times', ('number', -1), p[2])

    
def p_expression_term(p):
    'expression : term'
    p[0] = p[1]

    
def p_term_times(p):
    'term : term TIMES term'
    p[0] = ('times', p[1], p[3])

    
def p_term_divide(p):
    'term : term DIVIDE term'
    p[0] = ('divide', p[1], p[3])

    
def p_term_double_pipe(p):
    'term : term DOUBLE_PIPE string'
    p[0] = ('times', p[1], ('channel', p[3]))

    
def p_term_factor(p):
    'term : factor'
    p[0] = p[1]

    
def p_factor_sum(p):
    'factor : SUM L_BRACKET TEXT COMMA NUMBER COMMA NUMBER COMMA expression R_BRACKET'
    p[0] = ('sum', p[3], p[5], p[7], p[9])

    
def p_factor_bound(p):
    'factor : BOUND'
    p[0] = ('bound', p[1])

    
def p_factor_symbol(p):
    'factor : symbol'
    p[0] = p[1]

    
def p_factor_number(p):
    'factor : number'
    p[0] = p[1]

    
def p_factor_expression(p):
    'factor : L_PAREN expression R_PAREN'
    p[0] = p[2]

    
def p_symbol_string(p):
    'symbol : string'
    p[0] = ('symbol', p[1])

    
def p_string_text(p):
    'string : TEXT'
    p[0] = ('string-literal', p[1])

    
def p_string_math(p):
    'string : MATH_START expression MATH_END'
    p[0] = ('string-math', p[2])

    
def p_string_concat(p):
    'string : string string %prec CONCAT'
    p[0] = ('string-concat', p[1], p[2])

    
def p_number_number(p):
    'number : NUMBER'
    p[0] = ('number', p[1])

    
def p_error(p):
    if p is not None:
        raise SyntaxError(f"Line {p.lineno}, illegal token '{p.value}'")
    else:
        raise SyntaxError("Unexpected EOF")


lexer = lex()
parser = yacc()
