# NOVA PROGRAMMING LANGUAGE COMPILER DESIGN

# ===== 1. LEXICAL ANALYSIS =====

class Token:
    def __init__(self, type, value, line, column):
        self.type = type
        self.value = value
        self.line = line
        self.column = column
    
    def __repr__(self):
        return f"Token({self.type}, '{self.value}', line={self.line}, col={self.column})"

class Lexer:
    # Token types
    TOKEN_TYPES = {
        # Keywords
        'func': 'FUNC',
        'if': 'IF',
        'else': 'ELSE',
        'while': 'WHILE',
        'return': 'RETURN',
        'var': 'VAR',
        'true': 'TRUE',
        'false': 'FALSE',
        'null': 'NULL',
        
        # Special symbols
        '+': 'PLUS',
        '-': 'MINUS',
        '*': 'MULTIPLY',
        '/': 'DIVIDE',
        '=': 'ASSIGN',
        '==': 'EQUALS',
        '!=': 'NOT_EQUALS',
        '<': 'LESS_THAN',
        '>': 'GREATER_THAN',
        '<=': 'LESS_EQUAL',
        '>=': 'GREATER_EQUAL',
        '(': 'LEFT_PAREN',
        ')': 'RIGHT_PAREN',
        '{': 'LEFT_BRACE',
        '}': 'RIGHT_BRACE',
        ',': 'COMMA',
        ';': 'SEMICOLON',
    }
    
    def __init__(self, source):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.current_char = self.source[0] if self.source else None
    
    def advance(self):
        self.position += 1
        self.column += 1
        
        if self.position >= len(self.source):
            self.current_char = None
        else:
            if self.current_char == '\n':
                self.line += 1
                self.column = 1
            self.current_char = self.source[self.position]
    
    def peek(self):
        peek_pos = self.position + 1
        if peek_pos >= len(self.source):
            return None
        return self.source[peek_pos]
    
    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()
    
    def skip_comment(self):
        if self.current_char == '/' and self.peek() == '/':
            while self.current_char is not None and self.current_char != '\n':
                self.advance()
            self.advance()  # Skip the newline
    
    def identifier(self):
        start_column = self.column
        result = ''
        
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()
        
        token_type = self.TOKEN_TYPES.get(result, 'IDENTIFIER')
        return Token(token_type, result, self.line, start_column)
    
    def number(self):
        start_column = self.column
        result = ''
        is_float = False
        
        while self.current_char is not None and (self.current_char.isdigit() or self.current_char == '.'):
            if self.current_char == '.':
                if is_float:  # Second decimal point is an error
                    raise Exception(f"Invalid number format at line {self.line}, column {self.column}")
                is_float = True
            result += self.current_char
            self.advance()
        
        token_type = 'FLOAT' if is_float else 'INTEGER'
        value = float(result) if is_float else int(result)
        return Token(token_type, value, self.line, start_column)
    
    def string(self):
        start_column = self.column
        self.advance()  # Skip the opening quote
        result = ''
        
        while self.current_char is not None and self.current_char != '"':
            if self.current_char == '\\':
                self.advance()
                if self.current_char == 'n':
                    result += '\n'
                elif self.current_char == 't':
                    result += '\t'
                elif self.current_char == '"':
                    result += '"'
                elif self.current_char == '\\':
                    result += '\\'
                else:
                    raise Exception(f"Invalid escape sequence at line {self.line}, column {self.column}")
            else:
                result += self.current_char
            self.advance()
        
        if self.current_char is None:
            raise Exception(f"Unterminated string at line {self.line}, column {start_column}")
        
        self.advance()  # Skip the closing quote
        return Token('STRING', result, self.line, start_column)
    
    def get_next_token(self):
        while self.current_char is not None:
            # Skip whitespace
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            
            # Skip comments
            if self.current_char == '/' and self.peek() == '/':
                self.skip_comment()
                continue
            
            # Identifier or keyword
            if self.current_char.isalpha() or self.current_char == '_':
                return self.identifier()
            
            # Number
            if self.current_char.isdigit():
                return self.number()
            
            # String
            if self.current_char == '"':
                return self.string()
            
            # Two-character operators
            if self.current_char == '=' and self.peek() == '=':
                token = Token('EQUALS', '==', self.line, self.column)
                self.advance()
                self.advance()
                return token
            
            if self.current_char == '!' and self.peek() == '=':
                token = Token('NOT_EQUALS', '!=', self.line, self.column)
                self.advance()
                self.advance()
                return token
            
            if self.current_char == '<' and self.peek() == '=':
                token = Token('LESS_EQUAL', '<=', self.line, self.column)
                self.advance()
                self.advance()
                return token
            
            if self.current_char == '>' and self.peek() == '=':
                token = Token('GREATER_EQUAL', '>=', self.line, self.column)
                self.advance()
                self.advance()
                return token
            
            # Single character operators and symbols
            char = self.current_char
            if char in self.TOKEN_TYPES:
                token = Token(self.TOKEN_TYPES[char], char, self.line, self.column)
                self.advance()
                return token
            
            # Unknown character
            raise Exception(f"Invalid character '{self.current_char}' at line {self.line}, column {self.column}")
        
        # End of file
        return Token('EOF', None, self.line, self.column)
    
    def tokenize(self):
        tokens = []
        while True:
            token = self.get_next_token()
            tokens.append(token)
            if token.type == 'EOF':
                break
        return tokens


# ===== 2. SYNTAX ANALYSIS =====

class ASTNode:
    pass

class Program(ASTNode):
    def __init__(self, statements):
        self.statements = statements

class FunctionDecl(ASTNode):
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body

class VarDecl(ASTNode):
    def __init__(self, name, init_expr=None):
        self.name = name
        self.init_expr = init_expr

class Block(ASTNode):
    def __init__(self, statements):
        self.statements = statements

class IfStatement(ASTNode):
    def __init__(self, condition, if_body, else_body=None):
        self.condition = condition
        self.if_body = if_body
        self.else_body = else_body

class WhileStatement(ASTNode):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

class ReturnStatement(ASTNode):
    def __init__(self, expr=None):
        self.expr = expr

class ExprStatement(ASTNode):
    def __init__(self, expr):
        self.expr = expr

class BinaryExpr(ASTNode):
    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right

class UnaryExpr(ASTNode):
    def __init__(self, operator, expr):
        self.operator = operator
        self.expr = expr

class CallExpr(ASTNode):
    def __init__(self, callee, arguments):
        self.callee = callee
        self.arguments = arguments

class VariableExpr(ASTNode):
    def __init__(self, name):
        self.name = name

class AssignExpr(ASTNode):
    def __init__(self, name, value):
        self.name = name
        self.value = value

class LiteralExpr(ASTNode):
    def __init__(self, value):
        self.value = value


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current = 0
    
    def parse(self):
        statements = []
        while not self.is_at_end():
            statements.append(self.declaration())
        return Program(statements)
    
    def declaration(self):
        if self.match('FUNC'):
            return self.function_declaration()
        if self.match('VAR'):
            return self.var_declaration()
        return self.statement()
    
    def function_declaration(self):
        name = self.consume('IDENTIFIER', "Expected function name.")
        
        self.consume('LEFT_PAREN', "Expected '(' after function name.")
        parameters = []
        if not self.check('RIGHT_PAREN'):
            parameters.append(self.consume('IDENTIFIER', "Expected parameter name."))
            while self.match('COMMA'):
                parameters.append(self.consume('IDENTIFIER', "Expected parameter name."))
        self.consume('RIGHT_PAREN', "Expected ')' after parameters.")
        
        self.consume('LEFT_BRACE', "Expected '{' before function body.")
        body = self.block_statement()
        
        return FunctionDecl(name, parameters, body)
    
    def var_declaration(self):
        name = self.consume('IDENTIFIER', "Expected variable name.")
        
        initializer = None
        if self.match('ASSIGN'):
            initializer = self.expression()
        
        self.consume('SEMICOLON', "Expected ';' after variable declaration.")
        return VarDecl(name, initializer)
    
    def statement(self):
        if self.match('IF'):
            return self.if_statement()
        if self.match('WHILE'):
            return self.while_statement()
        if self.match('RETURN'):
            return self.return_statement()
        if self.match('LEFT_BRACE'):
            return self.block_statement()
        return self.expression_statement()
    
    def if_statement(self):
        self.consume('LEFT_PAREN', "Expected '(' after 'if'.")
        condition = self.expression()
        self.consume('RIGHT_PAREN', "Expected ')' after condition.")
        
        if_body = self.statement()
        else_body = None
        
        if self.match('ELSE'):
            else_body = self.statement()
        
        return IfStatement(condition, if_body, else_body)
    
    def while_statement(self):
        self.consume('LEFT_PAREN', "Expected '(' after 'while'.")
        condition = self.expression()
        self.consume('RIGHT_PAREN', "Expected ')' after condition.")
        
        body = self.statement()
        
        return WhileStatement(condition, body)
    
    def return_statement(self):
        expr = None
        if not self.check('SEMICOLON'):
            expr = self.expression()
        
        self.consume('SEMICOLON', "Expected ';' after return value.")
        return ReturnStatement(expr)
    
    def block_statement(self):
        statements = []
        
        while not self.check('RIGHT_BRACE') and not self.is_at_end():
            statements.append(self.declaration())
        
        self.consume('RIGHT_BRACE', "Expected '}' after block.")
        return Block(statements)
    
    def expression_statement(self):
        expr = self.expression()
        self.consume('SEMICOLON', "Expected ';' after expression.")
        return ExprStatement(expr)
    
    def expression(self):
        return self.assignment()
    
    def assignment(self):
        expr = self.equality()
        
        if self.match('ASSIGN'):
            value = self.assignment()
            
            if isinstance(expr, VariableExpr):
                return AssignExpr(expr.name, value)
            
            self.error(self.previous(), "Invalid assignment target.")
        
        return expr
    
    def equality(self):
        expr = self.comparison()
        
        while self.match('EQUALS', 'NOT_EQUALS'):
            operator = self.previous()
            right = self.comparison()
            expr = BinaryExpr(expr, operator, right)
        
        return expr
    
    def comparison(self):
        expr = self.term()
        
        while self.match('LESS_THAN', 'GREATER_THAN', 'LESS_EQUAL', 'GREATER_EQUAL'):
            operator = self.previous()
            right = self.term()
            expr = BinaryExpr(expr, operator, right)
        
        return expr
    
    def term(self):
        expr = self.factor()
        
        while self.match('PLUS', 'MINUS'):
            operator = self.previous()
            right = self.factor()
            expr = BinaryExpr(expr, operator, right)
        
        return expr
    
    def factor(self):
        expr = self.unary()
        
        while self.match('MULTIPLY', 'DIVIDE'):
            operator = self.previous()
            right = self.unary()
            expr = BinaryExpr(expr, operator, right)
        
        return expr
    
    def unary(self):
        if self.match('MINUS', 'NOT'):
            operator = self.previous()
            right = self.unary()
            return UnaryExpr(operator, right)
        
        return self.call()
    
    def call(self):
        expr = self.primary()
        
        while True:
            if self.match('LEFT_PAREN'):
                expr = self.finish_call(expr)
            else:
                break
        
        return expr
    
    def finish_call(self, callee):
        arguments = []
        
        if not self.check('RIGHT_PAREN'):
            arguments.append(self.expression())
            while self.match('COMMA'):
                arguments.append(self.expression())
        
        self.consume('RIGHT_PAREN', "Expected ')' after arguments.")
        
        return CallExpr(callee, arguments)
    
    def primary(self):
        if self.match('TRUE'):
            return LiteralExpr(True)
        if self.match('FALSE'):
            return LiteralExpr(False)
        if self.match('NULL'):
            return LiteralExpr(None)
        
        if self.match('INTEGER', 'FLOAT', 'STRING'):
            return LiteralExpr(self.previous().value)
        
        if self.match('IDENTIFIER'):
            return VariableExpr(self.previous())
        
        if self.match('LEFT_PAREN'):
            expr = self.expression()
            self.consume('RIGHT_PAREN', "Expected ')' after expression.")
            return expr
        
        raise self.error(self.peek(), "Expected expression.")
    
    def match(self, *types):
        for type in types:
            if self.check(type):
                self.advance()
                return True
        return False
    
    def check(self, type):
        if self.is_at_end():
            return False
        return self.peek().type == type
    
    def advance(self):
        if not self.is_at_end():
            self.current += 1
        return self.previous()
    
    def is_at_end(self):
        return self.peek().type == 'EOF'
    
    def peek(self):
        return self.tokens[self.current]
    
    def previous(self):
        return self.tokens[self.current - 1]
    
    def consume(self, type, message):
        if self.check(type):
            return self.advance()
        raise self.error(self.peek(), message)
    
    def error(self, token, message):
        if token.type == 'EOF':
            position = f"at end"
        else:
            position = f"at '{token.value}'"
        
        raise Exception(f"[line {token.line}, column {token.column}] Error {position}: {message}")


# ===== 3. SEMANTIC ANALYSIS =====

class Symbol:
    def __init__(self, name, type=None):
        self.name = name
        self.type = type  # Can be expanded for type checking

class VariableSymbol(Symbol):
    def __init__(self, name, type=None):
        super().__init__(name, type)

class FunctionSymbol(Symbol):
    def __init__(self, name, params=None, return_type=None):
        super().__init__(name, return_type)
        self.params = params or []

class SymbolTable:
    def __init__(self, parent=None):
        self.symbols = {}
        self.parent = parent
    
    def define(self, symbol):
        self.symbols[symbol.name.value] = symbol
        return symbol
    
    def resolve(self, name):
        if name.value in self.symbols:
            return self.symbols[name.value]
        
        if self.parent:
            return self.parent.resolve(name)
        
        return None
    
    def create_child_scope(self):
        return SymbolTable(self)

class SemanticAnalyzer:
    def __init__(self):
        self.current_scope = SymbolTable()
        self.errors = []
        
        # Register built-in functions
        self.register_built_ins()
    
    def register_built_ins(self):
        # Create a token for the 'print' function name
        print_token = Token('IDENTIFIER', 'print', 0, 0)
        
        # Create a parameter token for print (representing variable arguments)
        param_token = Token('IDENTIFIER', 'value', 0, 0)
        
        # Register print with one parameter (can be extended for variable args)
        print_symbol = FunctionSymbol(print_token, [param_token])
        self.current_scope.define(print_symbol)
        
        # Add other built-in functions as needed
    
    def analyze(self, ast):
        self.visit(ast)
        return self.errors
    
    def visit(self, node):
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
    
    def generic_visit(self, node):
        # Default implementation for nodes without specific visitors
        pass
    
    def visit_Program(self, node):
        for statement in node.statements:
            self.visit(statement)
    
    def visit_FunctionDecl(self, node):
        function_symbol = FunctionSymbol(node.name, node.params)
        self.current_scope.define(function_symbol)
        
        # Create a new scope for the function body
        previous_scope = self.current_scope
        self.current_scope = self.current_scope.create_child_scope()
        
        # Define parameters in the function scope
        for param in node.params:
            param_symbol = VariableSymbol(param)
            self.current_scope.define(param_symbol)
        
        # Visit the function body
        self.visit(node.body)
        
        # Restore the previous scope
        self.current_scope = previous_scope
    
    def visit_VarDecl(self, node):
        # Check if variable is already defined in the current scope
        if node.name.value in self.current_scope.symbols:
            self.errors.append(f"Variable '{node.name.value}' already defined at line {node.name.line}")
            return
        
        # Initialize variable if there's an initializer
        if node.init_expr:
            self.visit(node.init_expr)
        
        # Define the variable in the current scope
        var_symbol = VariableSymbol(node.name)
        self.current_scope.define(var_symbol)
    
    def visit_Block(self, node):
        # Create a new scope for the block
        previous_scope = self.current_scope
        self.current_scope = self.current_scope.create_child_scope()
        
        # Visit each statement in the block
        for statement in node.statements:
            self.visit(statement)
        
        # Restore the previous scope
        self.current_scope = previous_scope
    
    def visit_IfStatement(self, node):
        self.visit(node.condition)
        self.visit(node.if_body)
        if node.else_body:
            self.visit(node.else_body)
    
    def visit_WhileStatement(self, node):
        self.visit(node.condition)
        self.visit(node.body)
    
    def visit_ReturnStatement(self, node):
        if node.expr:
            self.visit(node.expr)
    
    def visit_ExprStatement(self, node):
        self.visit(node.expr)
    
    def visit_BinaryExpr(self, node):
        self.visit(node.left)
        self.visit(node.right)
        
        # Type checking could be implemented here
    
    def visit_UnaryExpr(self, node):
        self.visit(node.expr)
        
        # Type checking could be implemented here
    
    def visit_CallExpr(self, node):
        self.visit(node.callee)
        
        # Check if callee is a function
        if isinstance(node.callee, VariableExpr):
            symbol = self.current_scope.resolve(node.callee.name)
            if symbol is None:
                self.errors.append(f"Undefined function '{node.callee.name.value}' at line {node.callee.name.line}")
            elif not isinstance(symbol, FunctionSymbol):
                self.errors.append(f"'{node.callee.name.value}' is not a function at line {node.callee.name.line}")
            else:
                # Special case for built-in functions like 'print'
                if node.callee.name.value in ['print']:
                    # Skip argument count check for built-ins
                    pass
                else:
                    # Check argument count for user-defined functions
                    if len(node.arguments) != len(symbol.params):
                        self.errors.append(f"Expected {len(symbol.params)} arguments but got {len(node.arguments)} at line {node.callee.name.line}")
        
        # Visit arguments
        for argument in node.arguments:
            self.visit(argument)
    def visit_VariableExpr(self, node):
        # Check if variable is defined
        symbol = self.current_scope.resolve(node.name)
        if symbol is None:
            self.errors.append(f"Undefined variable '{node.name.value}' at line {node.name.line}")
    
    def visit_AssignExpr(self, node):
        # Check if variable is defined
        symbol = self.current_scope.resolve(node.name)
        if symbol is None:
            self.errors.append(f"Undefined variable '{node.name.value}' at line {node.name.line}")
        
        self.visit(node.value)
    
    def visit_LiteralExpr(self, node):
        # Literals don't need semantic checking
        pass

class Environment:
    def __init__(self, enclosing=None):
        self.values = {}
        self.enclosing = enclosing
    
    def define(self, name, value):
        self.values[name] = value
    
    def get(self, name):
        if name in self.values:
            return self.values[name]
        
        if self.enclosing:
            return self.enclosing.get(name)
        
        raise RuntimeError(f"Undefined variable '{name}'")
    
    def assign(self, name, value):
        if name in self.values:
            self.values[name] = value
            return
        
        if self.enclosing:
            self.enclosing.assign(name, value)
            return
        
        raise RuntimeError(f"Undefined variable '{name}'")


class Function:
    def __init__(self, declaration, closure):
        self.declaration = declaration
        self.closure = closure
    
    def call(self, interpreter, arguments):
        environment = Environment(self.closure)
        
        # Define parameters in function environment
        for i, param in enumerate(self.declaration.params):
            environment.define(param.value, arguments[i])
        
        try:
            interpreter.execute_block(self.declaration.body.statements, environment)
        except ReturnValue as return_value:
            return return_value.value
        
        return None
    
    def __str__(self):
        return f"<function {self.declaration.name.value}>"


class ReturnValue(Exception):
    def __init__(self, value):
        self.value = value
        super().__init__(self)


class Interpreter:
    def __init__(self):
        self.globals = Environment()
        self.environment = self.globals
        self.output = []  # Store output for printing
    
    def interpret(self, program):
        try:
            # First pass to define all functions
            for statement in program.statements:
                if isinstance(statement, FunctionDecl):
                    self.define_function(statement)
            
            # Find and execute the main function
            main_function = self.globals.get("main")
            if callable(main_function):
                result = main_function.call(self, [])
                return result
            else:
                raise RuntimeError("No main function defined")
            
        except Exception as e:
            print(f"Runtime Error: {e}")
            return None
    
    def define_function(self, function_decl):
        function = Function(function_decl, self.environment)
        self.environment.define(function_decl.name.value, function)
    
    def execute_block(self, statements, environment):
        previous = self.environment
        try:
            self.environment = environment
            
            for statement in statements:
                self.execute(statement)
                
        finally:
            self.environment = previous
    
    def execute(self, statement):
        method_name = f"execute_{type(statement).__name__}"
        method = getattr(self, method_name, self.execute_unknown)
        return method(statement)
    
    def execute_unknown(self, statement):
        raise RuntimeError(f"Unknown statement type: {type(statement).__name__}")
    
    def execute_Block(self, block):
        self.execute_block(block.statements, Environment(self.environment))
    
    def execute_ExprStatement(self, stmt):
        value = self.evaluate(stmt.expr)
    
    def execute_FunctionDecl(self, stmt):
        function = Function(stmt, self.environment)
        self.environment.define(stmt.name.value, function)
    
    def execute_IfStatement(self, stmt):
        condition = self.evaluate(stmt.condition)
        
        if self.is_truthy(condition):
            self.execute(stmt.if_body)
        elif stmt.else_body:
            self.execute(stmt.else_body)
    
    def execute_ReturnStatement(self, stmt):
        value = None
        if stmt.expr:
            value = self.evaluate(stmt.expr)
        
        raise ReturnValue(value)
    
    def execute_VarDecl(self, stmt):
        value = None
        if stmt.init_expr:
            value = self.evaluate(stmt.init_expr)
        
        self.environment.define(stmt.name.value, value)
    
    def execute_WhileStatement(self, stmt):
        while self.is_truthy(self.evaluate(stmt.condition)):
            self.execute(stmt.body)
    
    def evaluate(self, expr):
        method_name = f"evaluate_{type(expr).__name__}"
        method = getattr(self, method_name, self.evaluate_unknown)
        return method(expr)
    
    def evaluate_unknown(self, expr):
        raise RuntimeError(f"Unknown expression type: {type(expr).__name__}")
    
    def evaluate_AssignExpr(self, expr):
        value = self.evaluate(expr.value)
        self.environment.assign(expr.name.value, value)
        return value
    
    def evaluate_BinaryExpr(self, expr):
        left = self.evaluate(expr.left)
        right = self.evaluate(expr.right)
        
        operator_type = expr.operator.type
        
        if operator_type == 'PLUS':
            return left + right
        elif operator_type == 'MINUS':
            return left - right
        elif operator_type == 'MULTIPLY':
            return left * right
        elif operator_type == 'DIVIDE':
            return left / right
        elif operator_type == 'EQUALS':
            return left == right
        elif operator_type == 'NOT_EQUALS':
            return left != right
        elif operator_type == 'LESS_THAN':
            return left < right
        elif operator_type == 'GREATER_THAN':
            return left > right
        elif operator_type == 'LESS_EQUAL':
            return left <= right
        elif operator_type == 'GREATER_EQUAL':
            return left >= right
        
        return None
    
    def evaluate_CallExpr(self, expr):
        callee = self.evaluate(expr.callee)
        
        arguments = []
        for argument in expr.arguments:
            arguments.append(self.evaluate(argument))
        
        if not callable(callee):
            raise RuntimeError(f"Can only call functions")
        
        return callee.call(self, arguments)
    
    def evaluate_LiteralExpr(self, expr):
        return expr.value
    
    def evaluate_UnaryExpr(self, expr):
        right = self.evaluate(expr.expr)
        
        if expr.operator.type == 'MINUS':
            return -right
        
        return None
    
    def evaluate_VariableExpr(self, expr):
        return self.environment.get(expr.name.value)
    
    def is_truthy(self, value):
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        return True
    
    # Add built-in print function
    def add_native_functions(self):
        class PrintFunction:
            def call(self, interpreter, arguments):
                for arg in arguments:
                    interpreter.output.append(str(arg))
                    print(arg)  # Also print to console
                return None
        
        self.globals.define("print", PrintFunction())


# Extend the existing compile_nova function to execute the program
def run_nova(source_code):
    # 1. Lexical Analysis
    lexer = Lexer(source_code)
    tokens = lexer.tokenize()
    print("Lexical analysis completed")
    
    # 2. Syntax Analysis
    parser = Parser(tokens)
    ast = parser.parse()
    print("Syntax analysis completed")
    
    # 3. Semantic Analysis
    analyzer = SemanticAnalyzer()
    errors = analyzer.analyze(ast)
    
    if errors:
        print("Semantic errors found:")
        for error in errors:
            print(f"  - {error}")
        return None
    
    print("Semantic analysis completed")
    
    # 4. Interpretation/Execution
    interpreter = Interpreter()
    interpreter.add_native_functions()  # Add built-in functions
    
    print("\n=== Program Output ===")
    result = interpreter.interpret(ast)
    print("======================")
    
    if result is not None:
        print(f"\nProgram returned: {result}")
    
    return result

# Example usage
def compile_nova(source_code):
    # 1. Lexical Analysis
    lexer = Lexer(source_code)
    tokens = lexer.tokenize()
    print("Tokens:", tokens)
    
    # 2. Syntax Analysis
    parser = Parser(tokens)
    ast = parser.parse()
    print("AST created successfully")
    
    # 3. Semantic Analysis
    analyzer = SemanticAnalyzer()
    errors = analyzer.analyze(ast)
    
    if errors:
        print("Semantic errors found:")
        for error in errors:
            print(f"  - {error}")
        return None
    
    print("Semantic analysis passed")
    return ast


# Sample Nova program with print statements
sample_code = """
// Enhanced Nova program example with printing
func fibonacci(n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

func main() {
    var i = 0;
    while (i <= 10) {
        print(fibonacci(i));
        i = i + 1;
    }
    return fibonacci(10);
}
"""

# Run the program
result = run_nova(sample_code)