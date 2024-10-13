import numpy as np
from typing import List
from expr import *
from expr import BinaryExpression, ConstantExpression, Expression, Formula, UnaryExpression, VariableExpression
import sympy as sp
    

class ToSympy(Visitor):
    def visit_constant_expr(self, expr: ConstantExpression):
        return expr.value
    
    def visit_variable_expr(self, expr: VariableExpression):
        return sp.symbols(expr.name)
    
    def visit_binary_expr(self, expr: BinaryExpression):
        lhs = self.visit(expr.lhs)
        rhs = self.visit(expr.rhs)
        match expr.op:
            case BinaryOp.ADD: return lhs + rhs
            case BinaryOp.SUB: return lhs - rhs
            case BinaryOp.MUL: return lhs * rhs
            case BinaryOp.DIV: return lhs / rhs
            case _: raise NotImplementedError

    def visit_unary_expr(self, expr: UnaryExpression):
        x = self.visit(expr.expr)
        match expr.op:
            case UnaryOp.NEG: return 0 - x
            case UnaryOp.EXP: return sp.exp(x)
            case UnaryOp.SIN: return sp.sin(x)
            case UnaryOp.COS: return sp.cos(x)
            case UnaryOp.TAN: return sp.tan(x)
            case UnaryOp.ASIN: return sp.asin(x)
            case UnaryOp.ACOS: return sp.acos(x)
            case UnaryOp.ATAN: return sp.atan(x)
            case UnaryOp.SQRT: return sp.sqrt(x)
            case _: raise NotImplementedError

def to_sympy(x: Expression) -> sp.Expr:
    return ToSympy().visit(x)


def is_minus_one(x: Expression) -> bool:
    if not isinstance(x, ConstantExpression):
        return False
    
    return x.value == -1


def from_sympy(x: sp.Expr) -> Expression:
    if isinstance(x, sp.Add):
        lhs = from_sympy(x.args[0])
        rhs = from_sympy(x.args[1])
        return BinaryExpression(BinaryOp.ADD, lhs, rhs)
    elif isinstance(x, sp.Mul):
        lhs = from_sympy(x.args[0])
        rhs = from_sympy(x.args[1])
        if is_minus_one(lhs):
            return UnaryExpression(UnaryOp.NEG, rhs)
        elif is_minus_one(rhs):
            return UnaryExpression(UnaryOp.NEG, lhs)
        else:
            return BinaryExpression(BinaryOp.MUL, lhs, rhs)
    elif isinstance(x, sp.Pow):
        expr = from_sympy(x.args[0])
        expo = from_sympy(x.args[1])
        return BinaryExpression(BinaryOp.POW, expr, expo)
    elif isinstance(x, sp.exp):
        x = from_sympy(x.args[0])
        return UnaryExpression(UnaryOp.EXP, x)
    elif isinstance(x, sp.sin):
        x = from_sympy(x.args[0])
        return UnaryExpression(UnaryOp.SIN, x)
    elif isinstance(x, sp.cos):
        x = from_sympy(x.args[0])
        return UnaryExpression(UnaryOp.COS, x)
    elif isinstance(x, sp.tan):
        x = from_sympy(x.args[0])
        return UnaryExpression(UnaryOp.TAN, x)
    elif isinstance(x, sp.asin):
        x = from_sympy(x.args[0])
        return UnaryExpression(UnaryOp.ASIN, x)
    elif isinstance(x, sp.acos):
        x = from_sympy(x.args[0])
        return UnaryExpression(UnaryOp.ACOS, x)
    elif isinstance(x, sp.atan):
        x = from_sympy(x.args[0])
        return UnaryExpression(UnaryOp.ATAN, x)
    elif isinstance(x, sp.Symbol):
        return VariableExpression(x.name)
    elif isinstance(x, sp.Integer):
        return ConstantExpression(x.numerator)
    else:
        raise NotImplementedError


class ExpressionSampler(Visitor):
    """
    Example:
    ```python`
    sampler = ExpressionSampler()
    # Pick at random (uniform distribution) 3 nodes from the expression tree 'expr'
    nodes = sampler.sample(expr, k=3)
    ```

    This class implements reservoir sampling (algorithm R).
    """

    def __init__(self) -> None:
        super().__init__() 

        self.k = 0
        self.i = 0
        self.reservoir = []

    def visit_expr(self, expr: Expression):
        if len(self.reservoir) < self.k:
            self.reservoir.append(expr)
        else:
            j = np.random.randint(1, self.i + 1)
            if j <= self.k:
                self.reservoir[j] = expr
        
        self.i += 1

    def sample(self, expr: Expression, k: int = 1):
        self.k = k
        self.i = 0
        self.reservoir = []
        self.visit(expr)
        return self.reservoir

class Mutator(Visitor):
    def mutate(self, formula: Formula):
        raise NotImplementedError

class SequentialMutator(Mutator):
    def __init__(self, mutators: List[Mutator]) -> None:
        super().__init__()
        self.mutators = mutators

    def mutate(self, formula: Formula):
        for mutator in self.mutators:
            mutator.mutate(formula)

class RandomMutator(Mutator):
    def __init__(self, mutators: List[Mutator], p = None) -> None:
        super().__init__()
        self.mutators = mutators
        self.p = p

    def mutate(self, formula: Formula):
        selected_mutator = np.random.choice(self.mutators, p=self.p)
        selected_mutator.mutate(formula)

class ConstantNoiseMutator(Mutator):
    """
    Adds a random bias (gaussian noise) to constants (e.g. 4x -> 4.2940039x).
    """

    def __init__(self, std: float = 1.0) -> None:
        super().__init__()
        self.std = std

    def visit_constant_expr(self, expr: ConstantExpression):
        expr.value = expr.value + np.random.normal(0, self.std, np.shape(expr.value))

    def mutate(self, formula: Formula):
        self.visit(formula.expr)

class BinaryOperatorMutator(Mutator):
    """
    Randomly change the operator of a binary expression (e.g. x + y -> x * y).
    """

    def __init__(self, operators = [e for e in BinaryOp], p = None) -> None:
        super().__init__()
        self.operators = operators
        self.p = p

    def visit_binary_expr(self, expr: BinaryExpression):
        new_operator = np.random.choice(self.operators, self.p)
        expr.op = new_operator

    def mutate(self, formula: Formula):
        self.visit(formula.expr)

class UnaryOperatorMutator(Mutator):
    """
    Randomly change the operator of an unary expression (e.g. sin(x) -> cos(x))
    """

    def __init__(self, operators = [e for e in UnaryOp], p = None) -> None:
        super().__init__()
        self.operators = operators
        self.p = p

    def visit_unary_expr(self, expr: UnaryExpression):
        new_operator = np.random.choice(self.operators, self.p)
        expr.op = new_operator

    def mutate(self, formula: Formula):
        self.visit(formula.expr)

class BaseReplacerMutator(Mutator):
    def __init__(self, p: float) -> None:
        assert(p >= 0 and p <= 1)
        super().__init__()
        self.p = p

    def replace(self, expr: Expression) -> Expression:
        if np.random.random() < self.p:
            return self._replace(expr)
        else:
            return expr
    
    def _replace(self, expr: Expression) -> Expression:
        raise NotImplementedError

    def visit_expr(self, expr: Expression):
        return self.replace(expr)
    
    def visit_binary_expr(self, expr: BinaryExpression):
        expr.lhs = self.visit(expr.lhs)
        expr.rhs = self.visit(expr.rhs)
        return self.replace(expr)

    def visit_unary_expr(self, expr: UnaryExpression):
        expr.expr = self.visit(expr.expr)
        return self.replace(expr)
    
    def mutate(self, formula: Formula):
        formula.expr = self.visit(formula.expr)

class UnaryDeleterMutator(BaseReplacerMutator):
    """
    Randomly remove an unary expression (e.g. sin(x) -> x).
    """

    def __init__(self, p: float = 0.05) -> None:
        super().__init__(p)

    def _replace(self, expr: Expression) -> Expression:
        if isinstance(expr, UnaryExpression):
            return expr.expr
        else:
            return expr

class BinaryDeleterMutator(BaseReplacerMutator):
    """
    Randomly remove a binary expression by keeping either the left-hand-side 
    or the right-hand-side (e.g. x + y -> y).
    """

    def __init__(self, p: float = 0.05) -> None:
        super().__init__(p)

    def _replace(self, expr: Expression) -> Expression:
        if isinstance(expr, BinaryExpression):
            if np.random.choice([False, True]):
                return expr.lhs
            else:
                return expr.rhs
        else:
            return expr
