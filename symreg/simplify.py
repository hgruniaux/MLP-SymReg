"""
The mathematical expression simplifier.
"""

from symreg.formula import *
import numpy as np


def _simplify_add(expr: BinaryExpression) -> Expression:
    assert expr.op == BinaryOp.ADD

    if expr.lhs == expr.rhs:
        # x + x = 2x
        return BinaryExpression(BinaryOp.MUL, ConstantExpression(2), expr.lhs)
    elif isinstance(expr.rhs, ConstantExpression) and expr.rhs.value == 0:
        # x + 0 = x
        return expr.lhs

    return expr


def _simplify_sub(expr: BinaryExpression) -> Expression:
    assert expr.op == BinaryOp.SUB

    if expr.lhs == expr.rhs:
        # x - x = 0
        return ConstantExpression(0)
    elif isinstance(expr.rhs, ConstantExpression) and expr.rhs.value == 0:
        # x - 0 = x
        return expr.lhs

    return expr


def _simplify_mul(expr: BinaryExpression) -> Expression:
    assert expr.op == BinaryOp.MUL

    if isinstance(expr.rhs, ConstantExpression) and expr.rhs.value == 0:
        # x * 0 = 0
        return ConstantExpression(0)
    elif isinstance(expr.rhs, ConstantExpression) and expr.rhs.value == 1:
        # x * 1 = x
        return expr.lhs
    elif (
        isinstance(expr.lhs, UnaryExpression)
        and isinstance(expr.rhs, UnaryExpression)
        and expr.lhs.op == UnaryOp.SQRT
        and expr.rhs.op == UnaryOp.SQRT
        and expr.lhs.operand == expr.rhs.operand
    ):
        # sqrt(x)sqrt(x) = x
        return expr.lhs.operand

    return expr


def _simplify_div(expr: BinaryExpression) -> Expression:
    if expr.lhs == expr.rhs:
        # x / x = 1
        # This simplification is not always true as x may be null. However,
        # for our use case this is enough correct.
        return ConstantExpression(1)
    elif isinstance(expr.lhs, ConstantExpression) and expr.lhs.value == 0:
        # 0 / x = 0
        return ConstantExpression(0)
    elif isinstance(expr.rhs, ConstantExpression) and expr.rhs.value == 1:
        # x / 1 = x
        return expr.lhs
    elif isinstance(expr.rhs, ConstantExpression) and expr.rhs.value == 0:
        # Division by zero
        raise ZeroDivisionError

    return expr


class Simplifier(ExpressionVisitor):
    """
    A visitor class that simplifies mathematical expressions by evaluating constant expressions
    and applying mathematical identities and algebraic properties.

    Attributes:
        evaluate (bool): A flag indicating whether to evaluate constant expressions.
    """

    def __init__(self, evaluate: bool = True):
        super().__init__()
        self.evaluate = evaluate

    def visit_expr(self, expr: Expression) -> Expression:
        return expr

    def visit_binary_expr(self, expr) -> Expression:
        expr.lhs = self.accept(expr.lhs)
        expr.rhs = self.accept(expr.rhs)

        # Sort operands of commutative binary operators.
        if expr.op.is_commutative and expr.lhs < expr.rhs:
            expr.lhs, expr.rhs = expr.rhs, expr.lhs

        if self.evaluate:
            # Constant folding
            if isinstance(expr.lhs, ConstantExpression) and isinstance(
                expr.rhs, ConstantExpression
            ):
                return ConstantExpression(expr.op.value(expr.lhs.value, expr.rhs.value))

            # NaN propagation
            if (
                isinstance(expr.lhs, ConstantExpression) and np.isnan(expr.lhs.value)
            ) or (
                isinstance(expr.rhs, ConstantExpression) and np.isnan(expr.rhs.value)
            ):
                return ConstantExpression(np.nan)

            # Constant folding (associativity)
            if (
                isinstance(expr.lhs, BinaryExpression)
                and isinstance(expr.lhs.rhs, ConstantExpression)
                and isinstance(expr.rhs, ConstantExpression)
                and expr.op == expr.lhs.op
            ):
                # (x + k1) + k2 = x + (k1 + k2)
                return BinaryExpression(
                    expr.op,
                    expr.lhs.lhs,
                    ConstantExpression(
                        expr.op.value(expr.lhs.rhs.value, expr.rhs.value)
                    ),
                )

        match expr.op:
            case BinaryOp.ADD:
                return _simplify_add(expr)
            case BinaryOp.SUB:
                return _simplify_sub(expr)
            case BinaryOp.MUL:
                return _simplify_mul(expr)
            case BinaryOp.DIV:
                return _simplify_div(expr)

        return expr

    def visit_unary_expr(self, expr: UnaryExpression) -> Expression:
        expr.operand = self.accept(expr.operand)

        # Constant folding
        if self.evaluate and isinstance(expr.operand, ConstantExpression):
            return ConstantExpression(expr.op.value(expr.operand.value))

        # Simplify composition of bijective functions
        if (
            expr.op == UnaryOp.LOG
            and isinstance(expr.operand, UnaryExpression)
            and expr.operand.op == UnaryOp.EXP
        ):
            # log(exp(x)) = x
            return expr.operand.operand
        elif (
            expr.op == UnaryOp.EXP
            and isinstance(expr.operand, UnaryExpression)
            and expr.operand.op == UnaryOp.LOG
        ):
            # exp(log(x)) = x
            return expr.operand.operand
        elif (
            expr.op == UnaryOp.SIN
            and isinstance(expr.operand, UnaryExpression)
            and expr.operand.op == UnaryOp.ASIN
        ):
            # sin(arcsin(x)) = x
            return expr.operand.operand
        elif (
            expr.op == UnaryOp.TAN
            and isinstance(expr.operand, UnaryExpression)
            and expr.operand.op == UnaryOp.ATAN
        ):
            # tan(arctan(x)) = x
            return expr.operand.operand

        return expr


class ConstantSimplifier(ExpressionVisitor):
    """
    A visitor class that simplifies constant expressions by rounding them to the nearest integer
    if they are within a specified epsilon range.

    Attributes:
        eps (float): The epsilon value used to determine if a constant expression should be rounded.
                     Default is 0.1.
    """

    def __init__(self, eps: float = 1e-1):
        super().__init__()
        self.eps = eps

    def _simplify(self, value):
        if np.abs(np.floor(value) - value) <= self.eps:
            return np.floor(value)
        elif np.abs(np.ceil(value) - value) <= self.eps:
            return np.ceil(value)
        return value

    def visit_constant_expr(self, expr: ConstantExpression):
        expr.value = self._simplify(expr.value)


def simplify(e: Expression) -> Expression | Formula:
    simplifier = Simplifier()
    if isinstance(e, Expression):
        return simplifier.accept(e)
    elif isinstance(e, Formula):
        return Formula(simplifier.accept(e.expr))
    else:
        raise TypeError

def simplify_constants(e: Expression, eps: float = 0.1):
    simplifier = ConstantSimplifier(eps)
    if isinstance(e, Expression):
        simplifier.accept(e)
    elif isinstance(e, Formula):
        simplifier.accept(e.expr)
    else:
        raise TypeError
