import random
import numpy as np
from src.trees import ExpressionTree, Node
from src.operators import OPERATORS

class GPIndividual:
    """ Genetic Programming individual containing an expression tree and its fitness """
    def __init__(self, tree):
        self.tree = tree
        self.fitness = None  # Initially, fitness is not computed
        self.num_nodes = tree.num_nodes  # Stores the number of nodes

    def evaluate(self, x_train):
        """ Evaluates the individual's expression tree on the input data """
        return self.tree.evaluate(x_train)

    def update_nodes(self):
        """ Updates the node count after mutation or crossover """
        self.num_nodes = self.tree._count_nodes()

    def __str__(self):
        return f"{self.tree} (Nodes: {self.num_nodes})"
    
def generate_population(pop_size, num_vars, min_depth=1, max_depth=3):
    """ Generates an initial population of GP trees """
    return [GPIndividual(ExpressionTree(num_vars, random.randint(min_depth, max_depth), random.choice(["full", "grow"]))) 
            for _ in range(pop_size)]  # List of individuals, each containing a randomly generated expression tree

def evaluate_fitness(ind, x_train, y_train):
    """ Evaluates the individual's fitness using Mean Squared Error (MSE) """
    try:
        y_pred = ind.evaluate(x_train)  # Apply the individual's function to input data
        mse = np.mean((y_train - y_pred) ** 2)  # Compute MSE
        ind.fitness = mse  # Assign fitness directly to the individual
        return mse
    except (OverflowError, ValueError, FloatingPointError):  # Assign infinite fitness in case of errors
        ind.fitness = float('inf')
        return float('inf')

def tournament_selection(population, tournsize=3, p_size=0.0):
    """ 
    Selects the best individual from a randomly chosen subset.
    With probability `p_size`, selects the smallest tree instead of the best in fitness.
    """
    selected = random.sample(population, tournsize)
    
    if random.random() < p_size:
        return min(selected, key=lambda ind: ind.num_nodes)  # Selects the smallest tree
    else:
        return min(selected, key=lambda ind: ind.fitness)  # Selects the best in fitness

def diverse_tournament_selection(population, top_x, tournsize, p_size=0.0):
    """ 
    Diversity-enhanced tournament selection: 80% elite, 20% non-elite.
    With probability `p_size`, selects the smallest tree instead of the best in fitness.
    """
    pop_size = len(population)
    elite_size = int(top_x * pop_size)

    sorted_population = sorted(population, key=lambda ind: ind.fitness)
    elite = sorted_population[:elite_size]
    non_elite = sorted_population[elite_size:]

    selected_pool = elite if random.random() < 0.8 else non_elite
    return tournament_selection(selected_pool, tournsize, p_size)

def crossover(parent1, parent2):
    """ Crossover: swaps subtrees between two individuals while preserving tree structure """
    child1, child2 = parent1.tree.copy(), parent2.tree.copy()

    node1 = child1.root.get_random_subtree()
    node2 = child2.root.get_random_subtree()

    subtree_copy = node1.copy()  

    node1.replace_with(node2)  
    node2.replace_with(subtree_copy)  

    # Create new individuals and update their node counts
    new_child1 = GPIndividual(child1)
    new_child2 = GPIndividual(child2)

    # Simplify the trees after crossover
    new_child1.tree.root = new_child1.tree.root.simplify()
    new_child2.tree.root = new_child2.tree.root.simplify()

    new_child1.update_nodes()  # Update node count for offspring 1
    new_child2.update_nodes()  # Update node count for offspring 2

    return new_child1, new_child2

def mutate(ind):
    """ Mutation: replaces a random subtree with a newly generated random subtree """
    mutated_tree = ind.tree.copy()
    node_to_replace = mutated_tree.root.get_random_subtree()
    new_subtree = ExpressionTree(ind.tree.num_vars, depth=random.randint(1, 3), method="grow").root

    node_to_replace.replace_with(new_subtree)
    mutated_ind = GPIndividual(mutated_tree)
    mutated_ind.update_nodes()  # Update node count
    return mutated_ind

def run_gp(x, y, pop_size=1000, ngen=50, top_x=0.32, tournsize=3, cxpb=0.7, mutpb=0.2, p_size=0.0):
    num_variables = x.shape[0]
    population = generate_population(pop_size, num_variables)

    # Evaluate initial fitness
    for ind in population:
        evaluate_fitness(ind, x, y)

    hof = None  # Hall of Fame for the best individual found

    for gen in range(ngen):
        new_population = []

        while len(new_population) < pop_size:
            parent1 = diverse_tournament_selection(population, top_x, tournsize, p_size)
            parent2 = diverse_tournament_selection(population, top_x, tournsize, p_size)

            if random.random() < cxpb:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2  # Direct copy

            if random.random() < mutpb:
                child1 = mutate(child1)
            if random.random() < mutpb:
                child2 = mutate(child2)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        # Evaluate the new population
        for ind in new_population:
            evaluate_fitness(ind, x, y)

        # Replace the old population
        population = new_population

        # Find the best individual in this generation
        best_ind = min(population, key=lambda ind: ind.fitness)

        if hof is None or best_ind.fitness < hof.fitness:
            hof = best_ind  # Update Hall of Fame with the best found so far

        # Print the best result of this generation
        best_fitness = best_ind.fitness
        fitness_str = f"{best_fitness:.4e}" if best_fitness > 1e3 or best_fitness < 1e-4 else f"{best_fitness:.4f}"
        print(f"Gen {gen+1}: Best formula: {best_ind} → Fitness: {fitness_str}")

    return hof  # Returns the best individual found
