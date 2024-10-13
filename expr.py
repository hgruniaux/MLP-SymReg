from enum import Enum
import numpy as np


class BinaryOp(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    POW = "^"

    def evaluate(self, lhs, rhs):
        if self == BinaryOp.ADD:
            return lhs + rhs
        elif self == BinaryOp.SUB:
            return lhs - rhs
        elif self == BinaryOp.MUL:
            return lhs * rhs
        elif self == BinaryOp.DIV:
            return lhs / rhs
        elif self == BinaryOp.POW:
            return np.power(lhs, rhs)
        else:
            raise RuntimeError("Invalid binary operator.")

    @property
    def precedence(self) -> int:
        if self == BinaryOp.ADD or self == BinaryOp.SUB:
            return 1
        elif self == BinaryOp.MUL or self == BinaryOp.DIV:
            return 2
        elif self == BinaryOp.POW:
            return 3
        else:
            raise RuntimeError("Invalid binary operator.")


class UnaryOp(Enum):
    NEG = "-"
    EXP = "exp"
    SIN = "sin"
    COS = "cos"
    TAN = "tan"
    ASIN = "asin"
    ACOS = "acos"
    ATAN = "atan"
    SQRT = "sqrt"

    def evaluate(self, value):
        if self == UnaryOp.NEG:
            return -value
        elif self == UnaryOp.EXP:
            return np.exp(value)
        elif self == UnaryOp.SIN:
            return np.sin(value)
        elif self == UnaryOp.COS:
            return np.cos(value)
        elif self == UnaryOp.TAN:
            return np.tan(value)
        elif self == UnaryOp.ASIN:
            return np.arcsin(value)
        elif self == UnaryOp.ACOS:
            return np.arccos(value)
        elif self == UnaryOp.ATAN:
            return np.arctan(value)
        elif self == UnaryOp.SQRT:
            return np.sqrt(value)
        else:
            raise RuntimeError("Invalid unary operator.")


class Expression(object):
    def evaluate(self, context):
        raise NotImplemented

    def simplify(self):
        raise NotImplemented

    def display(self, prec: int | None) -> str:
        raise NotImplemented

    def __str__(self) -> str:
        return self.display(0)


class ConstantExpression(Expression):
    """
    Represents a constant.
    """

    def __init__(self, value) -> None:
        super().__init__()
        self.value = value

    def evaluate(self, context):
        return self.value

    def simplify(self):
        return self

    def display(self, prec) -> str:
        return str(self.value)


class VariableExpression(Expression):
    """
    Represents a variable (like an input variable).
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def evaluate(self, context):
        return context[self.name]

    def simplify(self):
        return self

    def display(self, prec) -> str:
        return self.name


class UnaryExpression(Expression):
    """
    Represents an unary expression (like negation, trigonometric functions, etc.)
    """

    def __init__(self, op: UnaryOp, expr: Expression) -> None:
        super().__init__()
        self.op = op
        self.expr = expr

    def evaluate(self, context):
        return self.op.evaluate(self.expr.evaluate(context))

    def simplify(self):
        expr = self.expr.simplify()

        if isinstance(expr, ConstantExpression):
            return ConstantExpression(self.op.evaluate(expr.evaluate(None)))

        return UnaryExpression(self.op, expr)

    def display(self, prec) -> str:
        return f"{self.op.value}({self.expr.display(0)})"


class BinaryExpression(Expression):
    """
    Represents a binary expression.
    """

    def __init__(self, op: BinaryOp, lhs: Expression, rhs: Expression) -> None:
        super().__init__()
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def evaluate(self, context):
        lhs_value = self.lhs.evaluate(context)
        rhs_value = self.rhs.evaluate(context)
        return self.op.evaluate(lhs_value, rhs_value)

    def simplify(self):
        lhs = self.lhs.simplify()
        rhs = self.rhs.simplify()

        if isinstance(lhs, ConstantExpression) and isinstance(rhs, ConstantExpression):
            return ConstantExpression(
                self.op.evaluate(lhs.evaluate(None), rhs.evaluate(None))
            )

        return BinaryExpression(self.op, lhs, rhs)

    def display(self, prec) -> str:
        op_prec = self.op.precedence
        output = f"{self.lhs.display(op_prec + 1)} {self.op.value} {self.rhs.display(op_prec + 1)}"
        if prec > op_prec:
            return f"({output})"
        else:
            return output

class Visitor(object):
    def visit(self, expr: Expression):
        if isinstance(expr, ConstantExpression):
            return self.visit_constant_expr(expr)
        elif isinstance(expr, VariableExpression):
            return self.visit_variable_expr(expr)
        elif isinstance(expr, UnaryExpression):
            return self.visit_unary_expr(expr)
        elif isinstance(expr, BinaryExpression):
            return self.visit_binary_expr(expr)

    def visit_expr(self, expr: Expression):
        pass

    def visit_constant_expr(self, expr: ConstantExpression):
        return self.visit_expr(expr)

    def visit_variable_expr(self, expr: VariableExpression):
        return self.visit_expr(expr)

    def visit_unary_expr(self, expr: UnaryExpression):
        self.visit(expr.expr)
        return self.visit_expr(expr)

    def visit_binary_expr(self, expr: BinaryExpression):
        self.visit(expr.lhs)
        self.visit(expr.rhs)
        return self.visit_expr(expr)

class DepthVisitor(Visitor):
    """
    Computes the maximum depth of an expression tree.
    """

    def visit_expr(self, expr: Expression):
        return 0

    def visit_unary_expr(self, expr: UnaryExpression):
        return 1 + self.visit(expr.expr)

    def visit_binary_expr(self, expr: BinaryExpression):
        return 1 + max(self.visit(expr.lhs), self.visit(expr.rhs))

class Formula:
    def __init__(self, expr: Expression) -> None:
        self.expr = expr

    def __call__(self, *args, **kwds):
        x = args[0]
        return self.expr.evaluate({ "x": x })

    def __str__(self) -> str:
        return self.expr.__str__()

    def simplify(self):
        return Formula(self.expr.simplify())

    @property
    def complexity(self) -> float:
        return DepthVisitor().visit(self.expr)
