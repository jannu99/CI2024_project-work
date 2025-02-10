import numpy as np

def safe_div(a, b):
    return np.divide(a, b, out=np.ones_like(a), where=np.abs(b) > 1e-6)

def safe_log(x):
    return np.log1p(np.abs(x))  

def safe_tan(x):
    return np.tan(np.clip(x, -np.pi/3, np.pi/3))

def safe_exp(x):
    return np.exp(np.clip(x, -100, 100))

def safe_pow(a, b):
    return np.power(np.abs(a), np.clip(b, -5, 5))

def safe_sqrt(x):
    return np.sqrt(np.abs(x))

def safe_arcsin(x):
    return np.arcsin(np.clip(x, -1, 1))

def safe_arccos(x):
    return np.arccos(np.clip(x, -1, 1))

def safe_arctanh(x):
    return np.arctanh(np.clip(x, -0.9999, 0.9999))

def safe_arccosh(x):
    return np.where(x >= 1, np.arccosh(x), 0)

OPERATORS = {
    "+": (np.add, 2),
    "-": (np.subtract, 2),
    "*": (np.multiply, 2),
    "/": (safe_div, 2),
    "sin": (np.sin, 1),
    "cos": (np.cos, 1),
    "tan": (safe_tan, 1),
    "asin": (safe_arcsin, 1),
    "acos": (safe_arccos, 1),
    "atan": (np.arctan, 1),
    "sinh": (np.sinh, 1),
    "cosh": (np.cosh, 1),
    "tanh": (np.tanh, 1),
    # "asinh": (np.arcsinh, 1),
    # "acosh": (safe_arccosh, 1),
    # "atanh": (safe_arctanh, 1),
    "exp": (safe_exp, 1),
    "log": (safe_log, 1),
    "sqrt": (safe_sqrt, 1),
    "pow": (safe_pow, 2)
}
