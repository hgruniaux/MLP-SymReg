from symreg.formula import *
from symreg.simplify import simplify
from copy import deepcopy


class DerivativeVisitor(ExpressionVisitor):
    def visit_constant_expr(self, expr: ConstantExpression) -> Expression:
        return ConstantExpression(0)

    def visit_variable_expr(self, expr: VariableExpression) -> Expression:
        return ConstantExpression(1)

    def visit_binary_expr(self, expr: BinaryExpression) -> Expression:
        match expr.op:
            case BinaryOp.ADD | BinaryOp.SUB:
                # (x + y)' = x' + y'
                return BinaryExpression(
                    expr.op, self.accept(expr.lhs), self.accept(expr.rhs)
                )
            case BinaryOp.MUL:
                # (x * y)' = x' * y + x * y'
                return BinaryExpression(
                    BinaryOp.ADD,
                    BinaryExpression(
                        BinaryOp.MUL, self.accept(expr.lhs), deepcopy(expr.rhs)
                    ),
                    BinaryExpression(
                        BinaryOp.MUL, deepcopy(expr.lhs), self.accept(expr.rhs)
                    ),
                )
            case BinaryOp.DIV:
                # (x / y)' = (x' * y - x * y') / (y * y)
                num = BinaryExpression(
                    BinaryOp.SUB,
                    # x' * y
                    BinaryExpression(
                        BinaryOp.MUL, self.accept(expr.lhs), deepcopy(expr.rhs)
                    ),
                    # x * y'
                    BinaryExpression(
                        BinaryOp.MUL, deepcopy(expr.lhs), self.accept(expr.rhs)
                    ),
                )
                den = BinaryExpression(
                    BinaryOp.MUL, deepcopy(expr.rhs), deepcopy(expr.rhs)
                )
                return BinaryExpression(BinaryOp.DIV, num, den)
            case _:
                raise NotImplementedError

    def visit_unary_expr(self, expr: UnaryExpression) -> Expression:
        match expr.op:
            case UnaryOp.EXP:
                # exp'(x) = x' * exp(x)
                return BinaryExpression(
                    BinaryOp.MUL, self.accept(expr.operand), deepcopy(expr)
                )
            case UnaryOp.SIN:
                # sin'(x) = x' * cos(x) = x' * sin(pi/2 - x)
                return BinaryExpression(
                    BinaryOp.MUL,
                    self.accept(expr.operand),
                    UnaryExpression(
                        UnaryOp.SIN,
                        BinaryExpression(
                            BinaryOp.SUB,
                            ConstantExpression(np.pi / 2),
                            deepcopy(expr.operand),
                        ),
                    ),
                )
            case UnaryOp.TAN:
                # tan'(x) = x' / cos(x)^2 = x' / (sin(pi/2 - x))^2
                cosx = UnaryExpression(
                    UnaryOp.SIN,
                    BinaryExpression(  # pi/2 - x
                        BinaryOp.SUB,
                        ConstantExpression(np.pi / 2),
                        deepcopy(expr.operand),
                    ),
                )
                cosx2 = BinaryExpression(BinaryOp.MUL, cosx, deepcopy(cosx))
                return BinaryExpression(BinaryOp.DIV, self.accept(expr.operand), cosx2)
            case UnaryOp.ASIN:
                # asin'(x) = x' / sqrt(1 - x^2)
                return BinaryExpression(
                    BinaryOp.DIV,
                    self.accept(expr.operand),
                    UnaryExpression(  # sqrt(1 - x^2)
                        UnaryOp.SQRT,
                        BinaryExpression(  # 1 - x^2
                            BinaryOp.SUB,
                            ConstantExpression(1),
                            BinaryExpression(  # x^2
                                BinaryOp.MUL,
                                deepcopy(expr.operand),
                                deepcopy(expr.operand),
                            ),
                        ),
                    ),
                )
            case UnaryOp.ATAN:
                # atan'(x) = x' / (1 + x^2)
                return BinaryExpression(
                    BinaryOp.DIV,
                    self.accept(expr.operand),
                    BinaryExpression(  # 1 + x^2
                        BinaryOp.ADD,
                        ConstantExpression(1),
                        BinaryExpression(  # x^2
                            BinaryOp.MUL, deepcopy(expr.operand), deepcopy(expr.operand)
                        ),
                    ),
                )
            case UnaryOp.SQRT:
                # sqrt'(x) = x' / (2 * sqrt(x))
                return BinaryExpression(
                    BinaryOp.DIV,
                    self.accept(expr.operand),
                    BinaryExpression(
                        BinaryOp.MUL, ConstantExpression(2), deepcopy(expr)
                    ),
                )
            case UnaryOp.LOG:
                # log'(x) = x' / x
                return BinaryExpression(
                    BinaryOp.DIV, self.accept(expr.operand), deepcopy(expr.operand)
                )
            case _:
                raise NotImplementedError


def derivate(expr: Expression | Formula) -> Expression:
    if isinstance(expr, Expression):
        d = DerivativeVisitor().accept(expr)
        d = simplify(d)
        return DerivativeVisitor().accept(expr)
    elif isinstance(expr, Formula):
        d = DerivativeVisitor().accept(expr.expr)
        d = simplify(d)
        return Formula(d)
    else:
        raise TypeError
