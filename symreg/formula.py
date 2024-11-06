"""

"""

import numpy as np
from enum import Enum


def _promote_constant_if_needed(x):
    if isinstance(x, int):
        return ConstantExpression(x)
    elif isinstance(x, float):
        return ConstantExpression(x)
    return x


class Expression:
    """
    Base class for all mathematical expressions.
    """

    def evaluate(self, ctx):
        raise NotImplementedError

    def display(self, prec: int = 0) -> str:
        raise NotImplementedError

    def latex(self, prec: int = 0) -> str:
        return self.display(prec)

    def __str__(self):
        return self.display()

    def __eq__(self, value) -> bool:
        raise NotImplementedError

    def __lt__(self, other) -> bool:
        raise NotImplementedError

    def __hash__(self) -> None:
        raise NotImplementedError

    def __call__(self, *args, **kwds):
        return self.evaluate(args)

    def __add__(self, other):
        return BinaryExpression(BinaryOp.ADD, self, _promote_constant_if_needed(other))

    def __sub__(self, other):
        return BinaryExpression(BinaryOp.SUB, self, _promote_constant_if_needed(other))

    def __mul__(self, other):
        return BinaryExpression(BinaryOp.MUL, self, _promote_constant_if_needed(other))

    def __truediv__(self, other):
        return BinaryExpression(BinaryOp.DIV, self, _promote_constant_if_needed(other))

    def __radd__(self, other):
        return BinaryExpression(BinaryOp.ADD, _promote_constant_if_needed(other), self)

    def __rsub__(self, other):
        return BinaryExpression(BinaryOp.SUB, _promote_constant_if_needed(other), self)

    def __rmul__(self, other):
        return BinaryExpression(BinaryOp.MUL, _promote_constant_if_needed(other), self)

    def __rtruediv__(self, other):
        return BinaryExpression(BinaryOp.DIV, _promote_constant_if_needed(other), self)

    def __neg__(self):
        return BinaryExpression(BinaryOp.MUL, ConstantExpression(-1), self)

    def __pow__(self, other):
        other = _promote_constant_if_needed(other)
        if isinstance(other, ConstantExpression) and other.value == 0:
            return ConstantExpression(1)
        if isinstance(other, ConstantExpression) and other.value == 1:
            return self
        if isinstance(other, ConstantExpression) and other.value == 2:
            return BinaryExpression(BinaryOp.MUL, self, self)
        if isinstance(other, ConstantExpression) and other.value == 0.5:
            return UnaryExpression(UnaryOp.SQRT, self)
        return UnaryExpression(
            UnaryOp.EXP,
            BinaryExpression(BinaryOp.MUL, other, UnaryExpression(UnaryOp.LOG, self)),
        )


class ConstantExpression(Expression):
    """
    A constant value (e.g. a scalar).
    """

    def __init__(self, value):
        super().__init__()
        self.value = value

    def evaluate(self, _):
        return self.value

    def display(self, prec: int = 0) -> str:
        return str(self.value)

    def __eq__(self, value) -> bool:
        if isinstance(value, ConstantExpression):
            return self.value == value.value
        return False

    def __hash__(self):
        return hash(self.value)

    def __lt__(self, other) -> bool:
        if isinstance(other, ConstantExpression):
            return self.value < other.value
        return True


class VariableExpression(Expression):
    """
    A variable input.
    """

    def __init__(self, idx: int = 0):
        super().__init__()
        self.idx = idx

    def evaluate(self, ctx):
        if self.idx < 0 or self.idx >= len(ctx):
            raise RuntimeError(f"Unknown variable 'x{self.idx}' referenced.")
        return ctx[self.idx]

    def display(self, prec: int = 0) -> str:
        return f"x{self.idx}"

    def latex(self, prec=0):
        return f"x_{{{self.idx}}}"

    def __eq__(self, value) -> bool:
        if isinstance(value, VariableExpression):
            return self.idx == value.idx
        return False

    def __hash__(self):
        return hash(self.idx)

    def __lt__(self, other) -> bool:
        if isinstance(other, ConstantExpression):
            return False
        if isinstance(other, VariableExpression):
            return self.idx < other.idx
        return True


class BinaryOp(Enum):
    """
    The different supported binary operators.

    See: BinaryExpression
    """

    ADD = np.add
    SUB = np.subtract
    MUL = np.multiply
    DIV = np.divide

    @property
    def is_commutative(self) -> bool:
        """
        Returns True if the operator is commutative (e.g. x op y = y op x).
        """

        match self:
            case BinaryOp.ADD | BinaryOp.MUL:
                return True
            case BinaryOp.SUB | BinaryOp.DIV:
                return False
            case _:
                raise NotImplementedError

    @property
    def precedence(self) -> int:
        """
        Returns the precedence of the operator (used for pretty printing).
        """

        match self:
            case BinaryOp.ADD | BinaryOp.SUB:
                return 1
            case BinaryOp.MUL | BinaryOp.DIV:
                return 2
            case _:
                raise NotImplementedError

    def __str__(self):
        match self:
            case BinaryOp.ADD:
                return "+"
            case BinaryOp.SUB:
                return "-"
            case BinaryOp.MUL:
                return "*"
            case BinaryOp.DIV:
                return "/"

    def latex(self, prec: int = 0):
        match self:
            case BinaryOp.ADD:
                return "+"
            case BinaryOp.SUB:
                return "-"
            case BinaryOp.MUL:
                return "\\times"
            case BinaryOp.DIV:
                return "/"

    def __lt__(self, other):
        return self.name < other.name


class BinaryExpression(Expression):
    """
    Represents a mathematical operator taking two operands,
    such as addition or multiplication.

    See: BinaryOp
    """

    def __init__(self, op: BinaryOp, lhs: Expression, rhs: Expression):
        super().__init__()
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def evaluate(self, ctx):
        lhs = self.lhs.evaluate(ctx)
        rhs = self.rhs.evaluate(ctx)
        return self.op.value(lhs, rhs)

    def display(self, prec: int = 0) -> str:
        op_prec = self.op.precedence
        output = (
            f"{self.lhs.display(op_prec + 1)} {self.op} {self.rhs.display(op_prec + 1)}"
        )
        if prec > op_prec:
            return f"({output})"
        else:
            return output

    def latex(self, prec: int = 0) -> str:
        op_prec = self.op.precedence
        match self.op:
            case BinaryOp.DIV:
                output = f"\\frac{{{self.lhs.latex()}}}{{{self.rhs.latex()}}}"
            case _:
                output = f"{self.lhs.latex(op_prec + 1)} {self.op.latex()} {self.rhs.latex(op_prec + 1)}"
        if prec > op_prec:
            return f"\\left({output}\\right)"
        else:
            return output

    def __eq__(self, value) -> bool:
        if isinstance(value, BinaryExpression):
            return (
                self.op == value.op and self.lhs == value.lhs and self.rhs == value.rhs
            )
        return False

    def __hash__(self):
        return hash((self.lhs, self.op, self.rhs))

    def __lt__(self, other) -> bool:
        if (
            isinstance(other, ConstantExpression)
            or isinstance(other, VariableExpression)
            or isinstance(other, UnaryExpression)
        ):
            return False
        if isinstance(other, BinaryExpression):
            return self.op < other.op
        return True


class UnaryOp(Enum):
    """
    The different supported unary operators.

    See: UnaryExpression
    """

    EXP = np.exp
    SIN = np.sin
    TAN = np.tan
    ASIN = np.arcsin
    ATAN = np.arctan
    SQRT = np.sqrt
    LOG = np.log

    def __str__(self):
        match self:
            case UnaryOp.EXP:
                return "exp"
            case UnaryOp.SIN:
                return "sin"
            case UnaryOp.TAN:
                return "tan"
            case UnaryOp.ASIN:
                return "asin"
            case UnaryOp.ATAN:
                return "atan"
            case UnaryOp.SQRT:
                return "sqrt"
            case UnaryOp.LOG:
                return "log"

    def latex(self) -> str:
        match self:
            case UnaryOp.EXP:
                return "\\exp"
            case UnaryOp.SIN:
                return "\\sin"
            case UnaryOp.TAN:
                return "\\tan"
            case UnaryOp.ASIN:
                return "\\sin^{-1}"
            case UnaryOp.ATAN:
                return "\\tan^{-1}"
            case UnaryOp.SQRT:
                return "\\sqrt"
            case UnaryOp.LOG:
                return "\\log"

    def __lt__(self, other):
        return self.name < other.name


class UnaryExpression(Expression):
    """
    Represents a mathematical operator taking a single operand,
    such as `sin` or negation.

    See: UnaryOp
    """

    def __init__(self, op: UnaryOp, operand: Expression):
        super().__init__()
        self.op = op
        self.operand = operand

    def evaluate(self, ctx):
        operand = self.operand.evaluate(ctx)
        return self.op.value(operand)

    def display(self, prec: int = 0) -> str:
        return f"{self.op}({self.operand})"

    def latex(self, prec: int = 0) -> str:
        match self.op:
            case UnaryOp.SQRT:
                return f"\\sqrt{{{self.operand.latex()}}}"
            case _:
                return f"{self.op.latex()}\\left({self.operand.latex()}\\right)"

    def __eq__(self, value) -> bool:
        if isinstance(value, UnaryExpression):
            return self.op == value.op and self.operand == value.operand
        return False

    def __hash__(self):
        return hash((self.op, self.operand))

    def __lt__(self, other) -> bool:
        if isinstance(other, ConstantExpression) or isinstance(
            other, VariableExpression
        ):
            return False
        if isinstance(other, UnaryExpression):
            return self.op < other.op
        return True


class ExpressionVisitor:
    """
    Base class to implement the visitor pattern for expressions.

    See: https://en.wikipedia.org/wiki/Visitor_pattern
    """

    def accept(self, expr: Expression):
        if isinstance(expr, ConstantExpression):
            return self.visit_constant_expr(expr)
        elif isinstance(expr, VariableExpression):
            return self.visit_variable_expr(expr)
        elif isinstance(expr, BinaryExpression):
            return self.visit_binary_expr(expr)
        elif isinstance(expr, UnaryExpression):
            return self.visit_unary_expr(expr)
        else:
            raise TypeError(f"Unknown expression type: {type(expr)}")

    def visit_expr(self, expr: Expression):
        pass

    def visit_constant_expr(self, expr: ConstantExpression):
        return self.visit_expr(expr)

    def visit_variable_expr(self, expr: VariableExpression):
        return self.visit_expr(expr)

    def visit_binary_expr(self, expr: BinaryExpression):
        self.accept(expr.lhs)
        self.accept(expr.rhs)
        return self.visit_expr(expr)

    def visit_unary_expr(self, expr: UnaryExpression):
        self.accept(expr.operand)
        return self.visit_expr(expr)


class ExpressionSampler(ExpressionVisitor):
    def __init__(self, k: int = 1, filter=None):
        super().__init__()

        self.k = k
        self.i = 0
        self.reservoir = []
        self.filter = filter

    def visit_expr(self, expr: Expression):
        if self.filter is not None and not self.filter(expr):
            return

        if len(self.reservoir) < self.k:
            self.reservoir.append(expr)
        else:
            j = np.random.randint(0, self.i)
            if j < self.k:
                self.reservoir[j] = expr

        self.i += 1


class Formula:
    """
    Represents a mathematical expression.
    """

    def __init__(self, expr: Expression):
        self.expr = expr

    def pick_random_node(self, k: int = 1, filter=None) -> Expression:
        """
        Pick at random k nodes from the expression tree (uniform distribution).

        If k=1, then an expression is returned.
        If k>1, then a list of expressions is returned.
        """

        assert k > 0

        sampler = ExpressionSampler(k, filter)
        sampler.accept(self.expr)
        if k == 1:
            return sampler.reservoir[0] if len(sampler.reservoir) > 0 else None
        else:
            return sampler.reservoir

    def __str__(self) -> str:
        return str(self.expr)

    def latex(self) -> str:
        return self.expr.latex()

    def __call__(self, *args, **kwds):
        return self.expr.__call__(*args, **kwds)

def exp(x: Expression) -> Expression:
    return UnaryExpression(UnaryOp.EXP, x)

def sin(x: Expression) -> Expression:
    return UnaryExpression(UnaryOp.SIN, x)

def cos(x: Expression) -> Expression:
    return sin(x + (np.pi / 2))

def tan(x: Expression) -> Expression:
    return UnaryExpression(UnaryOp.TAN, x)

def asin(x: Expression) -> Expression:
    return UnaryExpression(UnaryOp.ASIN, x)

def acos(x: Expression) -> Expression:
    return (np.pi / 2) - asin(x)

def atan(x: Expression) -> Expression:
    return UnaryExpression(UnaryOp.ATAN, x)

def sqrt(x: Expression) -> Expression:
    return UnaryExpression(UnaryOp.SQRT, x)

def log(x: Expression) -> Expression:
    return UnaryExpression(UnaryOp.LOG, x)
