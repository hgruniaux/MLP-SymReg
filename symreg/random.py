from symreg.formula import *
from symreg.simplify import *
import numpy.random as npr
from dataclasses import dataclass

@dataclass
class RandomOptions:
  prob_constant_expr = 0.2
  prob_variable_expr = 0.8
  prob_binary_expr = 1
  prob_unary_expr = 1
  allowed_variables = ['x']
  allowed_binary_operators = [o for o in BinaryOp]
  prob_binary_operators = [ 0.1, 0.1, 0.7, 0.1 ]
  allowed_unary_operators = [o for o in UnaryOp]
  prob_unary_operators = [ 0.15, 0.2, 0.05, 0.05, 0.05, 0.4, 0.1 ]
  max_depth = 4
  definition_set = None
  must_have_variable = True
  simplify = True

def _normalize_probas(x):
  return x / np.sum(x)

def _random_constant_expr() -> ConstantExpression:
  return ConstantExpression((npr.rand() * 10) - 5)

def _random_variable_expr(variables: list, p = None) -> VariableExpression:
  name = npr.choice(variables, p)
  return VariableExpression(name)

def _random_leaf_expr(options: RandomOptions) -> Expression:
  if npr.choice([True, False], p=_normalize_probas([ options.prob_constant_expr, options.prob_variable_expr ])):
    return _random_constant_expr()
  else:
    return _random_variable_expr(options.allowed_variables)

def _random_binary_expr(max_depth: int, options: RandomOptions) -> BinaryExpression:
  op = np.random.choice(options.allowed_binary_operators, p=options.prob_binary_operators)
  lhs = _random_expr(max_depth - 1, options)
  rhs = _random_expr(max_depth - 1, options)
  return BinaryExpression(op, lhs, rhs)

def _random_unary_expr(max_depth: int, options: RandomOptions) -> UnaryExpression:
  op = np.random.choice(options.allowed_unary_operators, p=options.prob_unary_operators)
  operand = _random_expr(max_depth - 1, options)
  return UnaryExpression(op, operand)

def _random_internal_expr(max_depth: int, options: RandomOptions) -> Expression:
  if npr.choice([True, False], p=_normalize_probas([ options.prob_binary_expr, options.prob_unary_expr ])):
    return _random_binary_expr(max_depth, options)
  else:
    return _random_unary_expr(max_depth, options)

def _random_expr(max_depth: int, options: RandomOptions) -> Expression:
  if max_depth == 0:
    return _random_leaf_expr(options)
  else:
    return _random_internal_expr(max_depth, options)

class HasVariableVisitor(ExpressionVisitor):
  def __init__(self):
    super().__init__()
    self.has_variable = False

  def visit_variable_expr(self, expr):
    self.has_variable = True

def _has_variable(expr: Expression) -> Expression:
  visitor = HasVariableVisitor()
  visitor.accept(expr)
  return visitor.has_variable

def random_expr(options: RandomOptions = RandomOptions()) -> Expression:
  # Suppress warnings about division by zero and invalid math operations (e.g. negative values in log).
  old_err = np.seterr(divide='ignore', invalid='ignore')

  while True:
    expr = _random_expr(npr.randint(1, options.max_depth + 1), options)
    if options.simplify:
      expr = simplify(expr)

    if options.must_have_variable and not _has_variable(expr):
      continue

    if options.definition_set is None:
      break

    # Check if the expression is valid in the given definition set (no division by zero,
    # negative values in logs, etc.).
    y = expr(options.definition_set)
    if np.shape(y) != np.shape(options.definition_set) or np.any(np.isnan(y)):
      # We found a NaN! Try to find a new expr.
      continue


    break

  np.seterr(**old_err) # restore old error settings
  return expr

def random_formula(options: RandomOptions = RandomOptions()) -> Formula:
  return Formula(random_expr(options))
