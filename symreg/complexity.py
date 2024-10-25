from symreg.formula import *


class DepthComplexityVisitor(ExpressionVisitor):
    """
    Visitor that computes the depth of an expression tree.
    """

    def visit_expr(self, expr: Expression) -> int:
        return 1

    def visit_binary_expr(self, expr: BinaryExpression) -> int:
        return 1 + max(self.accept(expr.lhs), self.accept(expr.rhs))

    def visit_unary_expr(self, expr: UnaryExpression) -> int:
        return 1 + self.accept(expr.operand)


def depth(expr: Expression | Formula) -> int:
    """
    Computes the depth of an expression tree (aka depth complexity).
    """
    visitor = DepthComplexityVisitor()
    if isinstance(expr, Expression):
        return visitor.accept(expr)
    elif isinstance(expr, Formula):
        return visitor.accept(expr.expr)
    else:
        raise TypeError(f"Expected an Expression or a Formula but got {type(expr)}.")


class NodeCountComplexityVisitor(ExpressionVisitor):
    """
    Visitor that computes the number of nodes in an expression tree.
    """

    def visit_expr(self, expr: Expression) -> int:
        return 1

    def visit_binary_expr(self, expr: BinaryExpression) -> int:
        return 1 + self.accept(expr.lhs) + self.accept(expr.rhs)

    def visit_unary_expr(self, expr: UnaryExpression) -> int:
        return 1 + self.accept(expr.operand)


def node_count(expr: Expression | Formula) -> int:
    """
    Computes the number of nodes in an expression tree (aka size complexity).
    """
    visitor = NodeCountComplexityVisitor()
    if isinstance(expr, Expression):
        return visitor.accept(expr)
    elif isinstance(expr, Formula):
        return visitor.accept(expr.expr)
    else:
        raise TypeError(f"Expected an Expression or a Formula but got {type(expr)}.")


class LeafCountComplexityVisitor(ExpressionVisitor):
    """
    Visitor that computes the number of leaf nodes (variables and constants) in an expression tree.
    """

    def visit_expr(self, expr: Expression) -> int:
        return 1

    def visit_binary_expr(self, expr: BinaryExpression) -> int:
        return self.accept(expr.lhs) + self.accept(expr.rhs)

    def visit_unary_expr(self, expr: UnaryExpression) -> int:
        return self.accept(expr.operand)


def leaf_count(expr: Expression | Formula) -> int:
    """
    Computes the number of leaf nodes (variables and constants) in an expression tree.
    """
    visitor = LeafCountComplexityVisitor()
    if isinstance(expr, Expression):
        return visitor.accept(expr)
    elif isinstance(expr, Formula):
        return visitor.accept(expr.expr)
    else:
        raise TypeError(f"Expected an Expression or a Formula but got {type(expr)}.")


class OperatorWeightComplexityVisitor(ExpressionVisitor):
    """
    Visitor that computes the sum of weight of operators in an expression tree.
    """

    def __init__(self, weights: dict):
        super().__init__()
        self.weights = weights

    def visit_expr(self, expr: Expression) -> int:
        return 0

    def visit_binary_expr(self, expr: BinaryExpression) -> int:
        w = self.weights[expr.op] if expr.op in self.weights else 1
        return w + self.accept(expr.lhs) + self.accept(expr.rhs)

    def visit_unary_expr(self, expr: UnaryExpression) -> int:
        w = self.weights[expr.op] if expr.op in self.weights else 1
        return w + self.accept(expr.operand)


def operator_weight_sum(expr: Expression | Formula, weights: dict) -> int:
    """
    Computes the sum of weight of operators in an expression tree (aka structural complexity).
    """
    visitor = OperatorWeightComplexityVisitor(weights)
    if isinstance(expr, Expression):
        return visitor.accept(expr)
    elif isinstance(expr, Formula):
        return visitor.accept(expr.expr)
    else:
        raise TypeError(f"Expected an Expression or a Formula but got {type(expr)}.")


def combined_complexity(
    expr: Expression | Formula,
    operator_weights: dict,
    weight_size: float = 1.0,
    weight_depth: float = 1.0,
    weight_structural: float = 1.0,
) -> int:
    """
    Computes the combined complexity of an expression tree (weighted sum of size, depth and structural complexity).
    """
    return (
        weight_depth * depth(expr)
        + weight_size * node_count(expr)
        + weight_structural * operator_weight_sum(expr, operator_weights)
    )
