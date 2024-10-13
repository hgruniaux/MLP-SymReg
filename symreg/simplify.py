"""
The mathematical expression simplifier.
"""

from symreg.formula import *

def _simplify_add(expr: UnaryExpression) -> Expression:
  assert(expr.op == BinaryOp.ADD)

  if expr.lhs == expr.rhs:
    # x + x = 2x
    return BinaryExpression(
      BinaryOp.MUL,
      ConstantExpression(2),
      expr.lhs
    )
  elif isinstance(expr.lhs, ConstantExpression) and expr.lhs.value == 0:
    # 0 + x = x
    return expr.rhs
  elif isinstance(expr.rhs, ConstantExpression) and expr.rhs.value == 0:
    # x + 0 = x
    return expr.lhs

  return expr

def _simplify_sub(expr: BinaryExpression) -> Expression:
  assert(expr.op == BinaryOp.SUB)

  if expr.lhs == expr.rhs:
    # x - x = 0
    return ConstantExpression(0)
  elif isinstance(expr.rhs, ConstantExpression) and expr.rhs.value == 0:
    # x - 0 = x
    return expr.lhs

  return expr

def _simplify_mul(expr: BinaryExpression) -> Expression:
  assert(expr.op == BinaryOp.MUL)

  if isinstance(expr.lhs, ConstantExpression) and expr.lhs.value == 0:
    # 0 * x = 0
    return ConstantExpression(0)
  elif isinstance(expr.rhs, ConstantExpression) and expr.rhs.value == 0:
    # x * 0 = 0
    return ConstantExpression(0)
  elif isinstance(expr.lhs, ConstantExpression) and expr.lhs.value == 1:
    # 1 * x = x
    return expr.rhs
  elif isinstance(expr.rhs, ConstantExpression) and expr.rhs.value == 1:
    # x * 1 = x
    return expr.lhs

  return expr

def _simplify_div(expr: BinaryExpression) -> Expression:
  if expr.lhs == expr.rhs:
    # x / x = 1
    # This simplification is not always true as x may be null. However,
    # for our use case this is enough correct.
    return ConstantExpression(1)
  elif isinstance(expr.rhs, ConstantExpression) and expr.rhs.value == 1:
    # x / 1 = x
    return expr.lhs

  return expr

class Simplifier(ExpressionVisitor):
  def __init__(self, evaluate: bool = True):
    super().__init__()
    self.evaluate = evaluate

  def visit_expr(self, expr: Expression) -> Expression:
    return expr

  def visit_binary_expr(self, expr) -> Expression:# Constant folding
    expr.lhs = self.accept(expr.lhs)
    expr.rhs = self.accept(expr.rhs)

    if self.evaluate and isinstance(expr.lhs, ConstantExpression) and isinstance(expr.rhs, ConstantExpression):
      return ConstantExpression(expr.op.value(expr.lhs.value, expr.rhs.value))

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

    return expr

def simplify(e: Expression) -> Expression | Formula:
  simplifier = Simplifier()
  if isinstance(e, Expression):
    return simplifier.accept(e)
  elif isinstance(e, Formula):
    return Formula(simplifier.accept(e.expr))
  else:
    raise TypeError
