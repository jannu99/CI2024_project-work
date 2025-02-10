import numpy as np
from src.evolution import run_gp
from src.utils_mio import plot_3d_function, plot_multiple_3d, plot_expression_tree


file_path = './data/problem_2.npz'
problem = np.load(file_path)
x = problem['x'] 
y = problem['y'] 
num_variables = x.shape[0]
num_samples = x.shape[1]
print(x.shape)

# Execute to get the formula mutpb=mutation probability p_size fitenss hole probability
best_formula = run_gp(x, y, pop_size=1000, ngen=100, mutpb=0.10, p_size=0.10)

print("\nBest formula:")
print(best_formula)
plot_3d_function(x, y, best_formula, num_samples)
plot_multiple_3d(x, y, best_formula, num_samples)
plot_expression_tree(best_formula.tree)