from symreg.formula import *
from symreg.derivative import *
import numpy as np

class SymbolicConstantExpression(VariableExpression):
    """
    A class representing a symbolic constant expression. This is used to replace
    the constants of an expression tree by a variable. This allows to compute the
    derivative of the expression with respect to the constants (for gradient descent).

    Attributes:
        idx (int): The index of the variable.
        value (float): The constant value of the expression. Defaults to a random float.
    """

    def __init__(self, idx: int, value: float = np.random.rand()):
        super().__init__(idx)
        self.value = value

    def __eq__(self, value):
        if isinstance(value, SymbolicConstantExpression):
            return self.idx == value.idx
        return False

    def evaluate(self, ctx):
        return self.value

    def display(self, prec=0):
        return str(self.value)

class ReplaceVisitor(ExpressionVisitor):
    """
    ReplaceVisitor provides a framework for replacing parts of an expression tree with other expressions.

    The subclasses must reimplement the replace() method.
    """

    def replace(self, expr: Expression) -> Expression:
        raise NotImplementedError

    def visit_expr(self, expr: Expression) -> Expression:
        return self.replace(expr)

    def visit_binary_expr(self, expr: BinaryExpression) -> Expression:
        expr.lhs = self.accept(expr.lhs)
        expr.rhs = self.accept(expr.rhs)
        return self.replace(expr)

    def visit_unary_expr(self, expr: UnaryExpression) -> Expression:
        expr.operand = self.accept(expr.operand)
        return self.replace(expr)

class SymbolizeConstantsVisitor(ReplaceVisitor):
    """
    It traverses an expression tree and replaces constant values
    with a variable symbolic representations (see SymbolicConstantExpression).

    The reverse step is done by DesymbolizeConstantsVisitor.
    """

    def __init__(self):
        super().__init__()
        self.constants = []

    def replace(self, expr: Expression) -> Expression:
        if isinstance(expr, ConstantExpression):
            sym_const = SymbolicConstantExpression(len(self.constants), expr.value)
            self.constants.append(sym_const)
            return sym_const
        else:
            return expr


class DesymbolizeConstantsVisitor(ReplaceVisitor):
    """
    It traverses an expression tree and replaces symbolic constant values
    with a constant expression.

    The reverse step is done by SymbolizeConstantsVisitor.
    """

    def replace(self, expr: Expression) -> Expression:
        if isinstance(expr, SymbolicConstantExpression):
            return ConstantExpression(expr.value)
        else:
            return expr

class GDOptimizer:
    """
    GDOptimizer is a class that performs gradient descent optimization on an expression or formula
    to optimize the constants.

    Parameters:
    -----------
    f : Expression | Formula
        The expression or formula to be optimized.
    learning_rate : float, optional (default=0.015)
        The step size for each iteration of gradient descent.
    clip_value : float, optional (default=1.0)
        The maximum value to which gradients are clipped to prevent exploding gradients.
    tolerance : float, optional (default=1e-6)
        The tolerance for the stopping criterion. The optimization stops when the change in the function value is below this threshold.
    max_iters : int, optional (default=1000)
        The maximum number of iterations for the optimization.
    verbose : bool, optional (default=False)
        If True, prints progress messages during optimization (for debugging purposes).

    Example:
    --------
    >>> from symreg.formula import *
    >>> from symreg.optimizer import *
    >>> import numpy as np

    >>> X = np.linspace(0, 5, 100)
    >>> Y = 2 * X ** 2 + 1 * X + 2

    >>> expr = BinaryExpression(...)  # Define the expression 'a * x ** 2 + b * x + c'
    >>> optimizer = GDOptimizer(expr, learning_rate=0.01, max_iters=500, verbose=True)
    >>> optimizer.optimize(X, Y)  # Optimize the constants
    >>> print(expr)
    """

    def __init__(
        self,
        f: Expression | Formula,
        learning_rate: float = 0.015,
        clip_value: float = 1.0,
        tolerance: float = 1e-6,
        max_iters: int = 1000,
        verbose: bool = False
    ):
        if isinstance(f, Expression):
            self.f = f
        elif isinstance(f, Formula):
            self.f = f.expr
        else:
            raise TypeError

        # Extract constants from the formula.
        visitor = SymbolizeConstantsVisitor()
        visitor.accept(self.f)
        self.constants = visitor.constants

        # Derivate f wrt each constant.
        self.f_prime = [derivate(self.f, const) for const in self.constants]
        if verbose or True:
            for i, f_prime in enumerate(self.f_prime):
                print(f"Derivative of f wrt constant {self.constants[i]}: {f_prime}")

        # Optimizer hyper parameters.
        self.learning_rate = learning_rate
        self.clip_value = clip_value
        self.tolerance = tolerance
        self.max_iters = max_iters

        self.verbose = verbose

        self.hook = None

    def _compute_gradients(self, X, Y):
        gradients = np.zeros(len(self.constants))

        # For each constant, compute the partial derivative of the expression.
        for i in range(len(self.constants)):
            Y_pred_deriv = self.f_prime[i](*X)

            # Compute the gradient of the loss with respect to this constant.
            Y_pred = self.f(*X)
            gradients[i] = 2 * np.mean((Y_pred - Y) * Y_pred_deriv)

        return gradients

    def optimize(self, X, Y):
        for epoch in range(self.max_iters):
            gradients = self._compute_gradients(X, Y)

            # Clip gradient if requested (to avoid too large updates).
            if self.clip_value is not None:
                gradients = np.clip(gradients, -self.clip_value, self.clip_value)

            # Update the constants using the gradient.
            for i, const in enumerate(self.constants):
                const.value -= self.learning_rate * gradients[i]

            # Check for convergence (if the update is very small)
            if np.linalg.norm(gradients) < self.tolerance:
                break

            if self.verbose:
                print(f"[Epoch {epoch}] Loss: {np.mean((Y - self.f(*X)) ** 2)}, Formula: {self.f}")

            if self.hook is not None:
                self.hook.step()

        DesymbolizeConstantsVisitor().accept(self.f)
