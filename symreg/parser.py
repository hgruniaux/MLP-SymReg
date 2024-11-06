from enum import Enum, auto
from symreg.formula import *


class Token(Enum):
    LPAREN = auto()
    RPAREN = auto()
    IDENTIFIER = auto()
    NUMBER = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    COMMA = auto()


class Lexer:
    def __init__(self, input: str):
        self.input = input
        self.pos = 0

    def _skip_whitespace(self):
        while self.pos < len(self.input) and self.input[self.pos].isspace():
            self.pos += 1

    def next_token(self):
        self._skip_whitespace()

        if self.pos >= len(self.input):
            # End of string.
            return None, None

        match self.input[self.pos]:
            case "(":
                self.pos += 1
                return Token.LPAREN, None
            case ")":
                self.pos += 1
                return Token.RPAREN, None
            case ",":
                self.pos += 1
                return Token.COMMA, None
            case "+":
                self.pos += 1
                return Token.PLUS, None
            case "-":
                self.pos += 1
                return Token.MINUS, None
            case "*":
                self.pos += 1
                return Token.STAR, None
            case "/":
                self.pos += 1
                return Token.SLASH, None
            case _ if self.input[self.pos].isalpha():
                start = self.pos
                while self.pos < len(self.input) and self.input[self.pos].isalnum():
                    self.pos += 1
                return Token.IDENTIFIER, self.input[start : self.pos]
            case _ if self.input[self.pos].isdigit():
                start = self.pos
                while self.pos < len(self.input) and self.input[self.pos].isdigit():
                    self.pos += 1
                return Token.NUMBER, float(self.input[start : self.pos])
            case _:
                raise ValueError(f"Unexpected character: {self.input[self.pos]}")


class Parser:
    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self._consume()

    def _consume(self):
        tok, val = self.lexer.next_token()
        self.current_token = tok
        self.current_val = val

    def _expect_token(self, tok):
        if tok != self.current_token:
            raise ValueError(f"Expected token {tok}, got {self.current_token}")

    def _parse_primary(self) -> Expression:
        match self.current_token:
            case Token.IDENTIFIER:
                name = self.current_val

                if name == "x":
                    name = 0
                elif name == "y":
                    name = 1
                elif name == "z":
                    name = 2
                elif name == "w":
                    name = 3

                self._consume()
                return VariableExpression(name)
            case Token.NUMBER:
                value = self.current_val
                self._consume()
                return ConstantExpression(value)
            case Token.LPAREN:
                self._consume()  # eat '('
                expr = self.parse_expr()
                self._expect_token(Token.RPAREN)
                self._consume()  # eat ')'
                return expr
            case _:
                raise ValueError(f"Unexpected token: {self.current_token}")

    def _handle_unary_func(self, func_name: str, arg: Expression) -> Expression:
        if func_name == "sin":
            return sin(arg)
        if func_name == "cos":
            return cos(arg)
        if func_name == "tan":
            return tan(arg)
        if func_name == "asin":
            return asin(arg)
        if func_name == "acos":
            return acos(arg)
        if func_name == "atan":
            return atan(arg)
        if func_name == "exp":
            return exp(arg)
        if func_name == "log":
            return log(arg)
        if func_name == "sqrt":
            return sqrt(arg)
        return None

    def _parse_call_expr(self) -> Expression:
        callee = self._parse_primary()
        if self.current_token != Token.LPAREN:
            return callee

        self._consume()  # eat '('

        args = []
        while self.current_token != Token.RPAREN:
            args.append(self.parse_expr())
            if self.current_token == Token.RPAREN:
                break
            self._expect_token(Token.COMMA)
            self._consume()  # eat ','

        self._expect_token(Token.RPAREN)
        self._consume()  # eat ')'

        # If the callee is a variable, then this can be a function call.
        if isinstance(callee, VariableExpression):
            expr = self._handle_unary_func(callee.name, args[0])
            if expr is not None:
                return expr  # Function call!

        # Otherwise, this is an implicit multiplication.
        if len(args) > 1:
            raise ValueError("Unexpected comma in in implicit multiplication")
        if len(args) == 0:
            return ValueError("Unexpected empty parentheses")
        return BinaryExpression(BinaryOp.MUL, callee, args[0])

    def _parse_unary_expr(self) -> Expression:
        if self.current_token == Token.MINUS:
            self._consume()
            return UnaryExpression(UnaryOp.NEG, self._parse_unary_expr())
        else:
            return self._parse_call_expr()

    def _get_binary_op(self) -> BinaryOp:
        match self.current_token:
            case Token.PLUS:
                return BinaryOp.ADD
            case Token.MINUS:
                return BinaryOp.SUB
            case Token.STAR:
                return BinaryOp.MUL
            case Token.SLASH:
                return BinaryOp.DIV
            case _:
                return None

    def _parse_binary_expr(self, lhs: Expression, expr_prec: int) -> Expression:
        while True:
            binop = self._get_binary_op()
            tok_prec = binop.precedence if binop is not None else -1

            # If this is a binop that binds at least as tightly
            # as the current operator, consume it, otherwise we are done.
            if tok_prec < expr_prec:
                return lhs

            self._consume()  # eat binop

            # Parse the primary expression after the binary operator.
            rhs = self._parse_primary()

            # If the next binary operator binds more tightly with rhs than the
            # operator after rhs, let the pending operator take rhs as its lhs.
            next_op = self._get_binary_op()
            next_prec = next_op.precedence if next_op is not None else -1
            if tok_prec < next_prec:
                rhs = self._parse_binary_expr(rhs, tok_prec + 1)

            # Merge lhs and rhs.
            lhs = BinaryExpression(binop, lhs, rhs)

    def parse_formula(self) -> Formula:
        return Formula(self.parse_expr())

    def parse_expr(self) -> Expression:
        lhs = self._parse_unary_expr()
        return self._parse_binary_expr(lhs, 0)


def parse(input: str) -> Formula:
    lexer = Lexer(input)
    parser = Parser(lexer)
    return parser.parse_formula()
