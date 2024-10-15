"""
The mathematical expression simplifier.
"""

from symreg.formula import *

def _is_commutative(op) -> bool:
  match op:
    case BinaryOp.ADD | BinaryOp.MUL:
      return True
    case BinaryOp.SUB | BinaryOp.DIV:
      return False
    case _:
      raise NotImplementedError

def _simplify_add(expr: BinaryExpression) -> Expression:
  assert(expr.op == BinaryOp.ADD)

  if expr.lhs == expr.rhs:
    # x + x = 2x
    return BinaryExpression(
      BinaryOp.MUL,
      ConstantExpression(2),
      expr.lhs
    )
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

  if isinstance(expr.rhs, ConstantExpression) and expr.rhs.value == 0:
    # x * 0 = 0
    return ConstantExpression(0)
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

    if isinstance(expr.lhs, ConstantExpression) and _is_commutative(expr.op):
      # Ensure constants are at right for commutative operators.
      expr.lhs, expr.rhs = expr.Rhs, expr.lhs

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

    if expr.op == UnaryOp.LOG and isinstance(expr.operand, UnaryExpression) and expr.operand.op == UnaryOp.EXP:
      # log(exp(x)) = x
      return expr.operand.operand
    elif expr.op == UnaryOp.EXP and isinstance(expr.operand, UnaryExpression) and expr.operand.op == UnaryOp.LOG:
      # exp(log(x)) = x
      return expr.operand.operand
    elif expr.op == UnaryOp.SIN and isinstance(expr.operand, UnaryExpression) and expr.operand.op == UnaryOp.ASIN:
      # sin(arcsin(x)) = x
      return expr.operand.operand
    elif expr.op == UnaryOp.TAN and isinstance(expr.operand, UnaryExpression) and expr.operand.op == UnaryOp.ATAN:
      # tan(arctan(x)) = x
      return expr.operand.operand

    return expr

def simplify(e: Expression) -> Expression | Formula:
  simplifier = Simplifier()
  if isinstance(e, Expression):
    return simplifier.accept(e)
  elif isinstance(e, Formula):
    return Formula(simplifier.accept(e.expr))
  else:
    raise TypeError
