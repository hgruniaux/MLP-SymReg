"""
This module contains the definition of mutators and many implementations of them.

A mutator is a class that modify a part of an expression tree (randomly or not).
They must be fast as they are executed many times on many elements.
The evolutionary algorithm running time depends on the speed of the mutator
execution.
"""

from symreg.formula import *
from typing import List
import numpy as np


class Mutator:
    """
    Abstract base class for all mutators.
    """

    def mutate(self, formula: Formula) -> None:
        """
        Mutates the given formula.
        """
        raise NotImplementedError


class RandomMutator(Mutator):
    """
    Mutator that executes a mutator picked randomly from a list of mutators.
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
                break
        return False


class SequentialMutator(Mutator):
    """
    Mutator that executes, sequentially, a list of mutator on a formula.

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
            mutated = mutated or mutator.mutate(formula)
        return mutated


class BaseVisitorMutator(Mutator, ExpressionVisitor):
    """
    Base class for mutators that are based on the expression visitor.
    """

    def mutate(self, formula: Formula) -> bool:
        self.accept(formula.expr)
        return True


class ConstantNoiseMutator(BaseVisitorMutator):
    """
    Mutator that add some random noise to constants.
    """

    def __init__(self, mean: float = 0, std: float = 1):
        super().__init__()
        self.mean = mean
        self.std = std

    def visit_constant_expr(self, expr):
        expr.value += np.random.normal(self.mean, self.std, np.shape(expr.value))


class BinaryOperatorMutator(BaseVisitorMutator):
    """
    Mutator that randomly change the operator of a binary expression.
    """

    def __init__(self, allowed_operators: List[BinaryOp], p: List[float] = None):
        super().__init__()
        self.allowed_operators = allowed_operators
        self.p = p

    def visit_binary_expr(self, expr: BinaryExpression):
        expr.op = np.random.choice(self.allowed_operators, self.p)


class UnaryOperatorMutator(BaseVisitorMutator):
    """
    Mutator that randomly change the operator of an unary expression.
    """

    def __init__(self, allowed_operators: List[BinaryOp], p: List[float] = None):
        super().__init__()
        self.allowed_operators = allowed_operators
        self.p = p

    def visit_unary_expr(self, expr: UnaryExpression):
        expr.op = np.random.choice(self.allowed_operators, self.p)


class VariableMutator(BaseVisitorMutator):
    """
    Mutator that randomly change the name of a variable expression.
    """

    def __init__(self, allowed_names: List[str], p: List[float] = None):
        super().__init__()
        self.allowed_names = allowed_names
        self.p = p

    def visit_variable_expr(self, expr: VariableExpression):
        expr.name = np.random.choice(self.allowed_names, self.p)


class BaseReplacerMutator(Mutator, ExpressionVisitor):
    """
    Base class for mutators that need to replace some parts of the expression tree.
    """

    def __init__(self):
        super().__init__()

    def replace(self, expr: Expression) -> Expression:
        """
        Replaces the given expression by another expression.
        To be reimplemented in children classes.
        """
        return expr  # no replacement by default

    def visit_expr(self, expr: Expression) -> Expression:
        return self.replace(expr)

    def visit_binary_expr(self, expr: BinaryExpression) -> Expression:
        expr.lhs = self.accept(expr.lhs)
        expr.rhs = self.accept(expr.rhs)
        return self.replace(expr)

    def visit_unary_expr(self, expr: UnaryExpression) -> Expression:
        expr.operand = self.accept(expr.operand)
        return self.replace(expr)

    def mutate(self, formula: Formula) -> bool:
        formula.expr = self.accept(formula.expr)
        return True
