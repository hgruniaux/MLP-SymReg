from expr import *
from expr import BinaryExpression, Expression, UnaryExpression

def is_constant(x: Expression, v) -> bool:
    return isinstance(x, ConstantExpression) and x.value == v


def is_same_symbol(x: Expression, y: Expression) -> bool:
    if isinstance(x, VariableExpression) and isinstance(y, VariableExpression):
        return x.name == y.name
    else:
        return False

class SimplifierVisitor(Visitor):
    def visit_expr(self, expr: Expression):
        return expr
    
    def visit_unary_expr(self, expr: UnaryExpression):
        x = self.visit(expr.expr)

        # Constant folding
        if isinstance(x, ConstantExpression):
            return ConstantExpression(expr.op.evaluate(x.value))

        return UnaryExpression(expr.op, x)

    def visit_binary_expr(self, expr: BinaryExpression):
        lhs = self.visit(expr.lhs)
        rhs = self.visit(expr.rhs)

        # Constant folding
        if isinstance(lhs, ConstantExpression) and isinstance(rhs, ConstantExpression):
            return ConstantExpression(expr.op.evaluate(lhs.value, rhs.value))

        # Algebraic simplification
        if expr.op == BinaryOp.SUB:
            if is_constant(lhs, 0): # 0 - x ==> -x
                return UnaryExpression(UnaryOp.NEG, rhs)
            elif is_constant(rhs, 0): # x - 0 ==> x
                return lhs
            elif is_same_symbol(lhs, rhs): # x - x ==> 0
                return ConstantExpression(0)
        elif expr.op == BinaryOp.ADD:
            if is_constant(lhs, 0): # 0 + x ==> x
                return rhs
            elif is_constant(rhs, 0): # x + 0 ==> x
                return lhs
            elif is_same_symbol(lhs, rhs): # x + x ==> 2*x
                return BinaryExpression(BinaryOp.MUL, ConstantExpression(2), rhs)
            elif isinstance(lhs, ConstantExpression): # K + x ==> x + K
                return BinaryExpression(BinaryOp.ADD, rhs, lhs)
        elif expr.op == BinaryOp.MUL:
            if is_constant(lhs, 0) or is_constant(rhs, 0): # 0 * x = x * 0 = 0
                return ConstantExpression(0)
            elif is_constant(lhs, 1): # 1 * x = x
                return rhs
            elif is_constant(rhs, 1): # x * 1 = x
                return lhs
            elif is_same_symbol(lhs, rhs):
                return BinaryExpression(BinaryOp.POW, lhs, ConstantExpression(2))
            elif isinstance(rhs, ConstantExpression): # x * K = K * x
                return BinaryExpression(BinaryOp.MUL, rhs, lhs)
        elif expr.op == BinaryOp.DIV:
            if is_same_symbol(lhs, rhs): # x / x = 1
                return ConstantExpression(1)

        return BinaryExpression(expr.op, lhs, rhs)