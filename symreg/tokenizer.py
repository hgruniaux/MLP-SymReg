from symreg.formula import *


class Tokenizer(ExpressionVisitor):
    def __init__(self):
        super().__init__()
        self.tokens = []

    def visit_constant_expr(self, expr):
        self.tokens.append(expr.value)

    def visit_variable_expr(self, expr):
        self.tokens.append(expr.idx)

    def visit_unary_expr(self, expr):
        self.tokens.append(expr.op)
        self.accept(expr.operand)

    def visit_binary_expr(self, expr):
        self.tokens.append(expr.op)
        self.accept(expr.lhs)
        self.accept(expr.rhs)

def tokenize(expr: Expression | Formula) -> list:
    visitor = Tokenizer()
    if isinstance(expr, Formula):
        visitor.accept(expr.expr)
    elif isinstance(expr, Expression):
        visitor.accept(expr)
    else:
        raise TypeError
    return visitor.tokens

