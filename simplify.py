import numpy as np
from expr import *
from expr import BinaryExpression, ConstantExpression, Expression, UnaryExpression


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


class SequentialMutator(Visitor):
    def __init__(self, mutators) -> None:
        super().__init__()
        self.mutators = mutators

    def visit_expr(self, expr: Expression):
        for mutator in self.mutators:
            mutator.visit(expr)


class RandomRepeatMutator(Visitor):
    def __init__(self, mutator, max_n: int = 1) -> None:
        super().__init__()
        self.mutator = mutator
        self.max_n = max_n

    def visit_expr(self, expr: Expression) -> None:
        n = np.random.randint(1, self.max_n + 1)
        for _ in range(n):
            self.mutator.visit(expr)


class RandomMutator(Visitor):
    def __init__(self, mutators, p = None) -> None:
        super().__init__()
        self.mutators = mutators
        self.p = p

    def visit_expr(self, expr: Expression):
        selected_mutator = np.random.choice(self.mutators, p=self.p)
        selected_mutator.visit(expr)

class RandomConstantMutator(Visitor):
    def __init__(self, std: float = 1.0) -> None:
        super().__init__()
        self.std = std

    def visit_constant_expr(self, expr: ConstantExpression):
        expr.value = expr.value + np.random.normal(0, self.std, np.shape(expr.value))


class RandomBinaryOperatorMutator(Visitor):
    def __init__(self, operators = [e for e in BinaryOp]) -> None:
        super().__init__()
        self.operators = operators

    def visit_binary_expr(self, expr: BinaryExpression):
        new_operator = np.random.choice(self.operators)
        expr.op = new_operator
        self.visit(expr.lhs)
        self.visit(expr.rhs)


class RandomUnaryOperatorMutator(Visitor):
    def __init__(self, operators = [e for e in UnaryOp]) -> None:
        super().__init__()
        self.operators = operators

    def visit_unary_expr(self, expr: UnaryExpression):
        new_operator = np.random.choice(self.operators)
        expr.op = new_operator
        self.visit(expr.expr)


class RandomUnaryDeleterMutator(Visitor):
    def __init__(self, p: float = 0.05) -> None:
        super().__init__()
        self.p = p

    def _maybe_remove(self, expr: UnaryExpression) -> Expression:
        if np.random.random() < self.p:
            return expr.expr
        else:
            return expr

    def visit_binary_expr(self, expr: BinaryExpression):
        if isinstance(expr.lhs, UnaryExpression):
            expr.lhs = self._maybe_remove(expr.lhs)
        if isinstance(expr.rhs, UnaryExpression):
            expr.rhs = self._maybe_remove(expr.rhs)
        self.visit(expr.lhs)
        self.visit(expr.rhs)

    def visit_unary_expr(self, expr: UnaryExpression):
        if isinstance(expr.expr, UnaryExpression):
            expr.expr = self._maybe_remove(expr.expr)
        self.visit(expr.expr)

class RandomBinaryDeleterMutator(Visitor):
    def __init__(self, p: float = 0.05) -> None:
        super().__init__()
        self.p = p

    def _maybe_remove(self, expr: UnaryExpression) -> Expression:
        if np.random.random() < self.p:
            if np.random.choice([False, True]):
                return expr.lhs
            else:
                return expr.rhs
        else:
            return expr

    def visit_binary_expr(self, expr: BinaryExpression):
        if isinstance(expr.lhs, BinaryExpression):
            expr.lhs = self._maybe_remove(expr.lhs)
        if isinstance(expr.rhs, BinaryExpression):
            expr.rhs = self._maybe_remove(expr.rhs)
        self.visit(expr.lhs)
        self.visit(expr.rhs)

    def visit_unary_expr(self, expr: UnaryExpression):
        if isinstance(expr.expr, BinaryExpression):
            expr.expr = self._maybe_remove(expr.expr)
        self.visit(expr.expr)
