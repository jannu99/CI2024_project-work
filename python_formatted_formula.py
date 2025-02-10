import numpy as np
from parse_formula import tree_to_function

# Expression saved as String
#Example
formula_string ="((- 5.22051 (sinh x1))"
generated=tree_to_function(formula_string)
print(generated)

