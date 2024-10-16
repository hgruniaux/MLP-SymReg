"""

"""

import numpy as np
from enum import Enum


class Expression:
    """
    Base class for all mathematical expressions.
    """

    def evaluate(self, ctx):
        raise NotImplementedError


    def may_be_null(self) -> bool:
        """
        Returns true if the expression may be null.
        """
        return True

    def may_be_negative(self) -> bool:
        """
        Returns true if the expression may be negative.
        """
        return True

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
        x = args[0]
        return self.evaluate({ "x": x })


class ConstantExpression(Expression):
    """
    A constant value (e.g. a scalar).
    """

    def __init__(self, value):
        super().__init__()
        self.value = value

    def evaluate(self, _):
        return self.value

    def may_be_null(self):
        return self.value == 0

    def may_be_negative(self):
        return self.value < 0

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

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def evaluate(self, ctx):
        if self.name not in ctx:
            raise RuntimeError(f"Unknown variable '{self.name}' referenced.")
        return ctx[self.name]

    def display(self, prec: int = 0) -> str:
        return self.name

    def __eq__(self, value) -> bool:
        if isinstance(value, VariableExpression):
            return self.name == value.name
        return False

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other) -> bool:
        if isinstance(other, ConstantExpression):
            return False
        if isinstance(other, VariableExpression):
            return self.name < other.name
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
            case BinaryOp.ADD | BinaryOp.MUL: return True
            case BinaryOp.SUB | BinaryOp.DIV: return False
            case _: raise NotImplementedError

    @property
    def precedence(self) -> int:
        """
        Returns the precedence of the operator (used for pretty printing).
        """

        match self:
            case BinaryOp.ADD | BinaryOp.SUB: return 1
            case BinaryOp.MUL | BinaryOp.DIV: return 2
            case _: raise NotImplementedError

    def __str__(self):
        match self:
            case BinaryOp.ADD: return "+"
            case BinaryOp.SUB: return "-"
            case BinaryOp.MUL: return "*"
            case BinaryOp.DIV: return "/"

    def latex(self, prec: int = 0):
        match self:
            case BinaryOp.ADD: return "+"
            case BinaryOp.SUB: return "-"
            case BinaryOp.MUL: return "\\times"
            case BinaryOp.DIV: return "/"

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

    def may_be_null(self):
        match self.op:
            case BinaryOp.ADD:
                return (self.lhs.may_be_null() and self.rhs.may_be_null()) or (self.lhs.may_be_negative() or self.rhs.may_be_negative())
            case BinaryOp.MUL:
                return self.lhs.may_be_null() or self.rhs.may_be_null()
            case BinaryOp.DIV:
                return self.lhs.may_be_null()
            case _:
                return super().may_be_null()

    def may_be_negative(self):
        match self.op:
            case BinaryOp.ADD, BinaryOp.MUL, BinaryOp.DIV:
                return self.lhs.may_be_negative() or self.rhs.may_be_negative()
            case _:
                return super().may_be_negative()

    def display(self, prec: int = 0) -> str:
        op_prec = self.op.precedence
        output = f"{self.lhs.display(op_prec + 1)} {self.op} {self.rhs.display(op_prec + 1)}"
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
            return self.op == value.op and self.lhs == value.lhs and self.rhs == value.rhs
        return False

    def __hash__(self):
        return hash((self.lhs, self.op, self.rhs))

    def __lt__(self, other) -> bool:
        if isinstance(other, ConstantExpression) or isinstance(other, VariableExpression) or isinstance(other, UnaryExpression):
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
            case UnaryOp.EXP: return "exp"
            case UnaryOp.SIN: return "sin"
            case UnaryOp.TAN: return "tan"
            case UnaryOp.ASIN: return "asin"
            case UnaryOp.ATAN: return "atan"
            case UnaryOp.SQRT: return "sqrt"
            case UnaryOp.LOG: return "log"

    def latex(self) -> str:
        match self:
            case UnaryOp.EXP: return '\\exp'
            case UnaryOp.SIN: return '\\sin'
            case UnaryOp.TAN: return '\\tan'
            case UnaryOp.ASIN: return '\\sin^{-1}'
            case UnaryOp.ATAN: return '\\tan^{-1}'
            case UnaryOp.SQRT: return '\\sqrt'
            case UnaryOp.LOG: return '\\log'

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

    def may_be_null(self):
        match self.op:
            case UnaryOp.EXP:
                return False
            case UnaryOp.SQRT:
                return self.operand.may_be_null()
            case _:
                return super().may_be_null()

    def may_be_negative(self):
        match self.op:
            case UnaryOp.EXP:
                return False
            case UnaryOp.SQRT:
                return False
            case _:
                return super().may_be_negative()

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
        if isinstance(other, ConstantExpression) or isinstance(other, VariableExpression):
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
            raise NotImplementedError

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
    def __init__(self, k: int = 1, filter = None):
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

    def pick_random_node(self, k: int = 1, filter = None) -> Expression:
        """
        Pick at random k nodes from the expression tree (uniform distribution).

        If k=1, then an expression is returned.
        If k>1, then a list of expressions is returned.
        """

        assert(k > 0)

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
