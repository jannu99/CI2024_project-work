import re
#USED TO GENERATE A PYTHON CODE FORMULA FROM A GP_Individual OR FROM A STRING

# Mapping of tree language operators to Python functions
OPERATOR_MAP = {
    "+": "({} + {})",
    "-": "({} - {})",
    "*": "({} * {})",
    "/": "safe_div({}, {})",
    "sin": "np.sin({})",
    "cos": "np.cos({})",
    "tan": "safe_tan({})",
    "asin": "safe_arcsin({})",
    "acos": "safe_arccos({})",
    "atan": "np.arctan({})",
    "sinh": "np.sinh({})",
    "cosh": "np.cosh({})",
    "tanh": "np.tanh({})",
    "exp": "safe_exp({})",
    "log": "safe_log({})",
    "sqrt": "safe_sqrt({})",
    "pow": "safe_pow({}, {})"
}

def parse_expression(expression):
    """ Parses the GP expression and converts it into a nested list """
    expression = expression.replace("(", " ( ").replace(")", " ) ").split()
    stack = []
    current = []

    for token in expression:
        if token == "(":
            stack.append(current)
            current = []
        elif token == ")":
            last = current
            current = stack.pop()
            current.append(last)
        else:
            current.append(token)

    return current[0] if current else []

def convert_to_python(parsed_expr):
    """ Converts the nested list into Python code """
    if isinstance(parsed_expr, str):  # Variable or number
        if re.match(r"^[-+]?\d+(\.\d+)?$", parsed_expr):  # Numeric constant
            return parsed_expr
        elif re.match(r"^x\d+$", parsed_expr):  # Variable (x0, x1, ...)
            return f"x[{parsed_expr[1:]}, :]"
        else:
            raise ValueError(f"Unknown token: {parsed_expr}")

    if not isinstance(parsed_expr, list) or len(parsed_expr) < 2:
        raise ValueError(f"Invalid expression: {parsed_expr}")

    operator = parsed_expr[0]
    operands = parsed_expr[1:]

    converted_args = [convert_to_python(arg) for arg in operands]

    if operator in OPERATOR_MAP:
        return OPERATOR_MAP[operator].format(*converted_args)
    else:
        raise ValueError(f"Unknown operator: {operator}")

def tree_to_function(tree_str):
    """ Converts an ExpressionTree string into executable Python code """
    parsed_expr = parse_expression(tree_str)
    function_body = convert_to_python(parsed_expr)

    function_code = f"""{function_body}"""
    return function_code