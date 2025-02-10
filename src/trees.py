import random
import numpy as np
from src.operators import OPERATORS  # Imports safe operators
import re

class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children else []

    def is_operator(self):
        return self.value in OPERATORS

    def copy(self):
        """ Creates a deep copy of the node and its children """
        return Node(self.value, [child.copy() for child in self.children])

    def evaluate(self, inputs):
        """ Recursively evaluates the node on an input array (num_variables, num_samples) """
        if self.is_operator():
            func, arity = OPERATORS[self.value]
            args = [child.evaluate(inputs) for child in self.children]
            return func(*args)
        elif isinstance(self.value, str):  # Variable (x0, x1, ...)
            var_index = int(self.value[1:])
            return inputs[var_index, :]
        else:
            return np.full_like(inputs[0, :], self.value)  # Constant

    def get_random_subtree(self):
        """ Selects a random subtree (node). """
        all_nodes = self._collect_nodes()
        return random.choice(all_nodes)

    def replace_with(self, new_subtree):
        """ Replaces the current node with `new_subtree`. """
        self.value = new_subtree.value
        self.children = new_subtree.children

    def _collect_nodes(self):
        """ Collects all nodes in a list for random selection. """
        nodes = [self]
        for child in self.children:
            nodes.extend(child._collect_nodes())
        return nodes

    def simplify(self):
        """ Recursively simplifies the tree by removing redundant operations """

        # Simplify children before simplifying the current node
        for i in range(len(self.children)):
            self.children[i] = self.children[i].simplify()

        # If the node is an operator and all its children are constants, compute the result directly
        if self.is_operator():
            func, arity = OPERATORS[self.value]
            if all(isinstance(child.value, (int, float, np.float64)) for child in self.children):
                values = [float(child.value) for child in self.children]  # Convert everything to float for safety
                try:
                    result = func(*values)
                    return Node(round(float(result), 5))  # Convert to normal float and round
                except (OverflowError, ZeroDivisionError, ValueError, FloatingPointError):
                    return self  # If an error occurs, retain the operation

        # 🔹 Specific simplifications for mathematical operations
        if self.value == "*":
            if self.children[0].value == 1:
                return self.children[1]  # x * 1 = x
            if self.children[1].value == 1:
                return self.children[0]  # 1 * x = x
            if self.children[0].value == 0 or self.children[1].value == 0:
                return Node(0.0)  # x * 0 = 0

        elif self.value == "+":
            if self.children[0].value == 0:
                return self.children[1]  # 0 + x = x
            if self.children[1].value == 0:
                return self.children[0]  # x + 0 = x

        elif self.value == "-":
            if self.children[1].value == 0:
                return self.children[0]  # x - 0 = x

        elif self.value == "/":
            if self.children[1].value == 1:
                return self.children[0]  # x / 1 = x
            if self.children[0].value == 0:
                return Node(0.0)  # 0 / x = 0

        elif self.value == "pow":
            if self.children[1].value == 1:
                return self.children[0]  # x^1 = x
            if self.children[1].value == 0:
                return Node(1.0)  # x^0 = 1 (if x ≠ 0)

        elif self.value == "log":
            if self.children[0].value == 1:
                return Node(0.0)  # log(1) = 0

        elif self.value == "exp":
            if self.children[0].value == 0:
                return Node(1.0)  # e^0 = 1

        elif self.value in ["sin", "cos", "tan"]:
            if self.children[0].value == 0:
                return Node({"sin": 0.0, "cos": 1.0, "tan": 0.0}[self.value])  # sin(0)=0, cos(0)=1, tan(0)=0

        return self  # If no simplification applies, return the original node
    
    def calculate_complexity(self):
        """ 
        Computes the tree complexity in terms of:
        - Total number of nodes
        - Maximum depth
        """
        num_nodes = 1  # Count this node
        max_depth = 1  # Minimum depth

        for child in self.children:
            child_nodes, child_depth = child.calculate_complexity()
            num_nodes += child_nodes  # Sum child nodes
            max_depth = max(max_depth, child_depth + 1)  # Find max depth

        return num_nodes, max_depth
    
    def __str__(self):
        if self.is_operator():
            return f"({self.value} " + " ".join(str(child) for child in self.children) + ")"
        return str(self.value)


class ExpressionTree:
    def __init__(self, num_vars, depth=3, method="full"):
        """
        Initializes a random tree using the Full or Grow method.
        """
        self.num_vars = num_vars
        self.root = self._generate_tree(depth, method)  # Generate root node
        self.num_nodes = self._count_nodes()  # Count nodes initially

    @classmethod
    def from_string(cls, expression, num_vars):
        """
        Converts a string into an ExpressionTree.
        
        :param expression: String in prefix notation.
        :param num_vars: Number of variables.
        :return: ExpressionTree instance.
        """
        tokens = re.findall(r'[\w\.\+\-\*/]+|\(|\)', expression)

        def build_tree(tokens):
            """ Recursively constructs a Node from tokens """
            if not tokens:
                return None

            token = tokens.pop(0)

            if token == "(":
                operator = tokens.pop(0)  # Extract operator
                children = []

                while tokens[0] != ")":
                    children.append(build_tree(tokens))

                tokens.pop(0)  # Remove closing parenthesis
                return Node(operator, children)

            elif token.replace('.', '', 1).isdigit() or re.match(r'^\-?\d+(\.\d+)?$', token):
                return Node(float(token))

            return Node(token)  

        tree_root = build_tree(tokens)
        expr_tree = cls(num_vars)  # Create new ExpressionTree
        expr_tree.root = tree_root  # Assign root
        expr_tree.num_nodes = expr_tree._count_nodes()  # Update node count

        return expr_tree

    def _count_nodes(self):
        """ Counts the total number of nodes in the tree """
        return len(self.root._collect_nodes())

    def _generate_tree(self, depth, method, is_root=True):
        """ Generates a random tree using the specified method """
        if depth == 0:
            return self._random_terminal()
        if is_root or method == "full":
            return self._random_function(depth, method)
        if method == "grow" and random.random() < 0.5:
            return self._random_terminal()
        return self._random_function(depth, method)

    def _random_function(self, depth, method):
        """ Generates an operator node with children """
        op = random.choice(list(OPERATORS.keys()))
        arity = OPERATORS[op][1]
        return Node(op, [self._generate_tree(depth - 1, method, is_root=False) for _ in range(arity)])

    def _random_terminal(self):
        """ Generates a terminal node (constant or variable) """
        SPECIAL_CONSTANTS = [np.pi, np.e, np.sqrt(2), np.log(2), -np.pi, -np.e, -np.sqrt(2), -np.log(2)]

        if random.random() < 0.5:
            return Node(f"x{random.randint(0, self.num_vars - 1)}")
        else:
            return Node(round(random.choice(SPECIAL_CONSTANTS + [random.uniform(-10, 10)]), 3))

    def copy(self):
        """ Creates a deep copy of the tree. """
        new_tree = ExpressionTree(self.num_vars)
        new_tree.root = self.root.copy()
        new_tree.num_nodes = new_tree._count_nodes()
        return new_tree

    def evaluate(self, inputs):
        return self.root.evaluate(inputs)

    def __str__(self):
        return str(self.root)