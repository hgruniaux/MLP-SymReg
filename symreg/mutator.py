"""
This module contains the definition of mutators and many implementations of them.

A mutator is a class that modify a part of an expression tree (randomly or not).
They must be fast as they are executed many times on many elements.
The evolutionary algorithm running time depends on the speed of the mutator
execution.

Implemented mutators:
  - SimplifyMutator: try to simplifies the expression (not really a mutation but still useful)
  - ConstantNoiseMutator: adds some noise to a constant
  - VariableMutator: changes a variable
  - SwapBinaryExpressionMutator: swaps left and right operands of a non-commutative binary operator
  - BinaryOperatorMutator: changes a binary operator
  - UnaryOperatorMutator: changes an unary operator
  - BinaryInserterMutator: inserts a binary operator
  - UnaryInserterMutator: inserts an unary operator
  - BinaryRemoverMutator: removes a binary operator
  - UnaryRemoverMutator: removes an unary operator
"""

import numpy as np
from typing import List
from symreg.formula import *
from symreg.simplify import simplify


class Mutator:
    """
    Abstract base class for all mutators.
    """

    def mutate(self, formula: Formula) -> bool:
        """
        Mutates the given formula.
        """
        raise NotImplementedError


class SimplifyMutator(Mutator):
    """
    Mutator that simplifies a formula. This is not strictly speaking a mutation, but it is still useful.
    """

    def mutate(self, formula: Formula) -> bool:
        formula.expr = simplify(formula.expr)
        return True


class RandomMutator(Mutator):
    """
    Mutator that executes a mutator picked randomly from a list of mutators.

    It tries to execute mutators until one has effectively mutated the formula
    (as some mutators only work for some formulas). Actually, there is an upper
    bound in the maximum number of tries (to avoid infinite loops).
    """

    def __init__(self, mutators: List[Mutator], p: List[float] = None):
        super().__init__()
        self.mutators = mutators
        self.p = p

    def mutate(self, formula: Formula) -> bool:
        MAX_TRY = 10
        for _ in range(MAX_TRY):
            mutator = np.random.choice(self.mutators, self.p)
            if mutator.mutate(formula):
                return True
        return False


class SequentialMutator(Mutator):
    """
    Mutator that sequentially executes a list of mutators.
    This is useful for composing different mutations.
    See also RandomMutator.

    Example:
    ```
    mutator = SequentialMutator([
            Mutator1(),
            Mutator2()
        ])
    mutator.mutate(formula)

    # Is equivalent to:
    mutator1 = Mutator1()
    mutator2 = Mutator2()
    mutator1.mutate(formula)
    mutator2.mutate(formula)
    ```
    """

    def __init__(self, mutators: List[Mutator]):
        super().__init__()
        self.mutators = mutators

    def mutate(self, formula: Formula):
        mutated = False
        for mutator in self.mutators:
            mutated |= mutator.mutate(formula)
        return mutated


class BaseVisitorMutator(Mutator, ExpressionVisitor):
    """
    Base class for mutators that are based on the expression visitor pattern.
    """

    def __init__(self, filter=None):
        super().__init__()
        self.filter = filter

    def mutate(self, formula: Formula) -> bool:
        node = formula.pick_random_node(filter=self.filter)
        if node is not None:
            self.accept(node)
            return True
        else:
            return False


class ConstantNoiseMutator(BaseVisitorMutator):
    """
    Mutator that adds some random noise to constants. The noise follows
    a normal distribution with the parameters given to the mutator construction.
    """

    def __init__(self, mean: float = 0, std: float = 1):
        super().__init__(lambda e: isinstance(e, ConstantExpression))
        self.mean = mean
        self.std = std

    def visit_constant_expr(self, expr):
        expr.value += np.random.normal(self.mean, self.std, np.shape(expr.value))


class BinaryOperatorMutator(BaseVisitorMutator):
    """
    Mutator that randomly changes the operator of a binary expression
    (e.g. `x + y` may become `x * y`).
    """

    def __init__(
        self,
        allowed_operators: List[BinaryOp] = [op for op in BinaryOp],
        p: List[float] = None,
    ):
        super().__init__(lambda e: isinstance(e, BinaryExpression))
        self.allowed_operators = allowed_operators
        self.p = p

    def visit_binary_expr(self, expr: BinaryExpression):
        expr.op = np.random.choice(self.allowed_operators, self.p)


class SwapBinaryExpressionMutator(BaseVisitorMutator):
    """
    Mutator that swaps the left and right operands of a non-commutative
    binary operation (e.g. `x - y` becomes `y - x`). Commutative binary
    operation (like addition or multiplication) are not affected by
    this mutation (as the resulting expression will be equivalent).
    """

    def __init__(self):
        super().__init__(
            lambda e: isinstance(e, BinaryExpression) and not e.op.is_commutative
        )

    def visit_binary_expr(self, expr):
        expr.lhs, expr.rhs = expr.rhs, expr.lhs


class UnaryOperatorMutator(BaseVisitorMutator):
    """
    Mutator that randomly changes the operator of a unary expression
    (e.g. `sin(x)` may become `sqrt(x)`).
    """

    def __init__(
        self,
        allowed_operators: List[UnaryOp] = [op for op in UnaryOp],
        p: List[float] = None,
    ):
        super().__init__(lambda e: isinstance(e, UnaryExpression))
        self.allowed_operators = allowed_operators
        self.p = p

    def visit_unary_expr(self, expr: UnaryExpression):
        expr.op = np.random.choice(self.allowed_operators, self.p)


class VariableMutator(BaseVisitorMutator):
    """
    Mutator that randomly changes the name of a variable expression.
    """

    def __init__(self, allowed_names: List[str], p: List[float] = None):
        super().__init__(lambda e: isinstance(e, VariableExpression))
        self.allowed_names = allowed_names
        self.p = p

    def visit_variable_expr(self, expr: VariableExpression):
        expr.name = np.random.choice(self.allowed_names, self.p)


class ReplacerVisitor(ExpressionVisitor):
    def __init__(self, old: Expression, new: Expression):
        super().__init__()
        self.old = old
        self.new = new

    def visit_expr(self, expr: Expression) -> Expression:
        if expr is self.old:
            return self.new
        else:
            return expr

    def visit_binary_expr(self, expr: BinaryExpression) -> Expression:
        expr.lhs = self.accept(expr.lhs)
        expr.rhs = self.accept(expr.rhs)
        return self.visit_expr(expr)

    def visit_unary_expr(self, expr: UnaryExpression) -> Expression:
        expr.operand = self.accept(expr.operand)
        return self.visit_expr(expr)


class BaseInserterMutator(Mutator):
    """
    Base class for mutators that insert some node into the expression tree.
    Subclasses must reimplement the method insert().
    """

    def insert(self, child_expr: Expression) -> Expression:
        """To be reimplemented in subclasses."""
        raise NotImplementedError

    def mutate(self, formula: Formula) -> bool:
        insertion_point = formula.pick_random_node()
        new_node = self.insert(insertion_point)

        replacer = ReplacerVisitor(insertion_point, new_node)
        formula.expr = replacer.accept(formula.expr)

        return True


class UnaryInserterMutator(BaseInserterMutator):
    """
    Mutator that inserts an unary expression in the expression tree.
    """

    def __init__(
        self,
        allowed_operators: List[UnaryOp] = [op for op in UnaryOp],
        p: List[float] = None,
    ):
        super().__init__()
        self.allowed_operators = allowed_operators
        self.p = p

    def insert(self, child_expr: Expression) -> Expression:
        op = np.random.choice(self.allowed_operators, self.p)
        return UnaryExpression(op, child_expr)


class BinaryInserterMutator(BaseInserterMutator):
    """
    Mutator that inserts a binary expression in the expression tree.
    """

    def __init__(
        self,
        allowed_operators: List[BinaryOp] = [op for op in BinaryOp],
        p: List[float] = None,
    ):
        super().__init__()
        self.allowed_operators = allowed_operators
        self.p = p

    def insert(self, child_expr: Expression) -> Expression:
        op = np.random.choice(self.allowed_operators, self.p)
        lhs = child_expr
        rhs = ConstantExpression(1)  # TODO: how to generate the right hand side

        if np.random.choice([False, True]):
            lhs, rhs = rhs, lhs

        return BinaryExpression(op, lhs, rhs)


class RemoverVisitor(ExpressionVisitor):
    def __init__(self, target: Expression):
        super().__init__()
        self.target = target

    def visit_expr(self, expr: Expression) -> Expression:
        if expr is self.target:
            return None
        return expr

    def visit_binary_expr(self, expr: BinaryExpression) -> Expression:
        if expr is self.target:
            return np.random.choice([expr.lhs, expr.rhs])

        expr.lhs = self.accept(expr.lhs)
        expr.rhs = self.accept(expr.rhs)
        return expr

    def visit_unary_expr(self, expr: UnaryExpression) -> Expression:
        if expr is self.target:
            return expr.operand

        expr.operand = self.accept(expr.operand)
        return expr


class BaseRemoverMutator(Mutator):
    """
    Base class for mutators that need to remove one node from the expression tree.
    """

    def filter(self, expr: Expression) -> bool:
        raise NotImplementedError

    def mutate(self, formula: Formula) -> bool:
        target = formula.pick_random_node(filter=self.filter)
        if target is None:
            return False

        remover = RemoverVisitor(target)
        new_expr = remover.accept(formula.expr)
        if new_expr is None:
            return False
        else:
            formula.expr = new_expr
            return True


class UnaryRemoverMutator(BaseRemoverMutator):
    """
    Mutator that randomly removes one unary expression.
    The unary expression is replaced by its operand (e.g. `sin(x)` becomes `x`).
    """

    def filter(self, expr: Expression) -> bool:
        return isinstance(expr, UnaryExpression)


class BinaryRemoverMutator(BaseRemoverMutator):
    """
    Mutator that randomly removes one binary expression from the expression tree.
    The binary expression is replaced by one of its operands (e.g. `x + y` may become `x` or `y`).
    """

    def filter(self, expr: Expression) -> bool:
        return isinstance(expr, BinaryExpression)
