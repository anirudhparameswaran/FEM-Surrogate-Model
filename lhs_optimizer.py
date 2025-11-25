import numpy as np
from typing import List, Tuple, Dict
import pandas as pd
from scipy.stats import qmc, lognorm, gumbel_r, uniform
import random
from scipy.spatial.distance import pdist

n = 200      # number of design points
d = 7        # number of variables

# --- GA Configuration Parameters (Based on Paper Section 2.4 and 4.2) ---
# Population size (must be even). N_pop = 50 is recommended.
POPULATION_SIZE = 50 
# Mutation probability. P_mut = 0.1 is recommended.
MUTATION_PROBABILITY = 0.1 
# Number of generations to check for accumulated improvement (n). 100 is used here.
GENERATIONS_FOR_CHECK = 10
# Stopping criterion tolerance (epsilon). 1e-7 is used.
STOPPING_TOLERANCE = 1e-7 
# Safety break to prevent infinite loops (arbitrary high value)
MAX_GENERATIONS = 5000 

sampler = qmc.LatinHypercube(d)
sample = sampler.random(n)

def lhs_dist_matrix(L: np.ndarray) -> np.ndarray:
    N, p = L.shape
    sum_sq = np.sum(L**2, axis=1)
    dot_product_matrix = L @ L.T
    squared_distance_matrix = sum_sq[:, None] + sum_sq[None, :] - 2 * dot_product_matrix
    np.fill_diagonal(squared_distance_matrix, 10000000.00)
    return squared_distance_matrix

def lhs_min_dist(L: np.ndarray):
    D = lhs_dist_matrix(L)
    min_distance_squared = D.min()
    min_distance_count = np.sum(D <= min_distance_squared*1.1) // 2
    return min_distance_squared, min_distance_count

def lhs_force_criterion(L: np.ndarray):
    D = lhs_dist_matrix(L)
    inverse_squared_distances = 1 / D
    force_criterion_G = np.sum(np.tril(inverse_squared_distances, k=-1))
    return force_criterion_G

def evaluate_lhs(L: np.ndarray) -> Dict[str, float]:
    min_distance_squared, min_distance_count = lhs_min_dist(L)
    force_criterion_G = lhs_force_criterion(L)
    return {
        "min_distance_squared_d_L": min_distance_squared,
        "min_distance_count_n_L": min_distance_count,
        "force_criterion_G_L": force_criterion_G
    }

def _crossover(best_parent: np.ndarray, current_parent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs the two-child crossover operation (Paper Fig. 1).
    
    - Child 1: Best parent with a random column replaced by the column from current_parent.
    - Child 2: Current parent with a random column replaced by the column from best_parent.
    """
    p = best_parent.shape[1]
    # Select a random column index to swap
    col_index = random.randint(0, p - 1)
    
    # Child 1 (Best -> Current)
    child1 = best_parent.copy()
    child1[:, col_index] = current_parent[:, col_index]
    
    # Child 2 (Current -> Best)
    child2 = current_parent.copy()
    child2[:, col_index] = best_parent[:, col_index]
    
    return child1, child2

def _mutate(L: np.ndarray, p_mut: float):
    """
    Performs swap mutation on each column of the LHS with probability p_mut.
    If mutation occurs, two random elements in that column are swapped.
    (Paper Section 2.3).
    """
    N, p = L.shape
    for j in range(p):
        # Check if mutation occurs for this column
        if random.random() < p_mut:
            # Swap two random elements in the column
            idx1, idx2 = random.sample(range(N), 2)
            L[[idx1, idx2], j] = L[[idx2, idx1], j]

def optimize_lhs_ga(initial_LHS: np.ndarray) -> np.ndarray:
    """
    Optimizes a Latin Hypercube Sample (LHS) using a Genetic Algorithm based 
    on the force criterion (G(L)).

    Args:
        initial_LHS: The N x p NumPy array (RLH) to start the optimization from. 
                     It is assumed to be an integer matrix (permutations of 1 to N).

    Returns:
        The optimized N x p LHS matrix (OLH).
    """
    N, p = initial_LHS.shape
    num_survivors = POPULATION_SIZE // 2
    
    # 1. Initialize Population: 
    # The first individual is the user-provided RLH, the rest are new RLHs.
    population: List[np.ndarray] = [initial_LHS.copy()]
    while len(population) < POPULATION_SIZE:
        population.append(sampler.random(N))
    
    # Calculate initial fitness and find the overall best individual
    fitness_list = [(lhs_force_criterion(L), L) for L in population]
    fitness_list.sort(key=lambda x: x[0]) # Sort by G(L), minimizing fitness
    
    best_lhs_history = [fitness_list[0][0]] # G(L) of the best LHS in each generation
    
    current_best_lhs = fitness_list[0][1]
    
    print(f"Starting GA optimization (N={N}, p={p}). Initial G(L): {best_lhs_history[0]:.6f}")

    # 2. Main Optimization Loop
    for generation in range(1, MAX_GENERATIONS + 1):
        
        # --- Selection ---
        # Select the top N_pop/2 best LHs as survivors (parents)
        survivors = [item[1] for item in fitness_list[:num_survivors]]
        
        # The best LHS (fittest) from the entire current population
        best_of_generation = survivors[0]
        best_fitness_of_generation = fitness_list[0][0]
        
        # --- Stopping Criterion Check (Paper Eq. 6) ---
        if generation > GENERATIONS_FOR_CHECK and generation % GENERATIONS_FOR_CHECK == 0:
            # G(L_k) - G(L_{k-n}) < epsilon * (G(L_n) - G(L_0))
            
            # G(L_k) is best_fitness_of_generation
            G_k_minus_n = best_lhs_history[generation - GENERATIONS_FOR_CHECK]
            
            delta_G_n_initial = best_lhs_history[GENERATIONS_FOR_CHECK] - best_lhs_history[0]
            delta_G_k_n = best_fitness_of_generation - G_k_minus_n
            
            # Check for convergence based on accumulated improvement
            if delta_G_k_n > -STOPPING_TOLERANCE * abs(delta_G_n_initial):
                print(f"\n--- Stopping Criterion Met at Generation {generation} ---")
                print(f"Final G(L): {best_fitness_of_generation:.6f}")
                return current_best_lhs.copy()

        # --- Crossover ---
        new_population: List[np.ndarray] = []
        
        # 1. Preserve the absolute best individual (Paper Fig. 1, L1)
        # This prevents the best solution from being lost.
        new_population.append(best_of_generation.copy())
        
        # 2. Mating (Best parent with all other survivors)
        for k in range(1, num_survivors):
            parent2 = survivors[k]
            
            # Generate two children: Crossover operation ensures the LH property is kept
            child1, child2 = _crossover(best_of_generation, parent2)
            
            # Child 1 is k-th child, Child 2 is (N_pop/2 + k)-th child
            new_population.append(child1)
            new_population.append(child2)

        # Handle the last child to reach POPULATION_SIZE (since we started with 1)
        if len(new_population) < POPULATION_SIZE:
             new_population.append(survivors[num_survivors-1].copy()) # Should not happen if POPULATION_SIZE is even
        
        # --- Mutation ---
        # Mutate all children except the first one (the absolute best copy)
        for i in range(1, POPULATION_SIZE):
            _mutate(new_population[i], MUTATION_PROBABILITY)

        # --- Update Population and Fitness ---
        population = new_population
        fitness_list = [(lhs_force_criterion(L), L) for L in population]
        fitness_list.sort(key=lambda x: x[0]) # Sort for selection in the next generation
        
        current_best_lhs = fitness_list[0][1]
        best_lhs_history.append(fitness_list[0][0])
        
        # Optional: Print progress
        # if generation % 100 == 0:
            #  print(f"Gen {generation:4d}: Best G(L) = {best_lhs_history[-1]:.1f}")

    print(f"\n--- Reached Max Generations ({MAX_GENERATIONS}) ---")
    return current_best_lhs.copy()

def lhs_maximin(n, d, tries=50):
    best = None
    best_score = -np.inf
    for _ in range(tries):
        sampler = qmc.LatinHypercube(d)  # LHS sampler from SciPy
        s = sampler.random(n)  # Generate n points in [0,1]^d
        dist = pdist(s)  # Calculate pairwise distance
        min_dist = dist.min()  # Find the smallest distance
        if min_dist > best_score:
            best_score = min_dist  # Keep track of the best maximin score
            best = s  # Store the best LHS sample
    return best

LHS_example = sample

results = evaluate_lhs(LHS_example)
print("Out of the box LHS:")
print(f"Minimum Distance Squared (d(L)): {results['min_distance_squared_d_L']:.3f}")
print(f"Min Distance Occurrences (n(L)): {results['min_distance_count_n_L']}")
print(f"Force Criterion (G(L)): {results['force_criterion_G_L']:.4f}")
print("-" * 30)

optimized_lhs = optimize_lhs_ga(LHS_example)
results = evaluate_lhs(optimized_lhs)
print("GA Optimized LHS:")
print(f"Minimum Distance Squared (d(L)): {results['min_distance_squared_d_L']:.3f}")
print(f"Min Distance Occurrences (n(L)): {results['min_distance_count_n_L']}")
print(f"Force Criterion (G(L)): {results['force_criterion_G_L']:.3f}")
print("-" * 30)

maximin_lhs = lhs_maximin(n, d, tries=100)
results = evaluate_lhs(maximin_lhs)
print("Maximin out of 50 tries LHS:")
print(f"Minimum Distance Squared (d(L)): {results['min_distance_squared_d_L']:.3f}")
print(f"Min Distance Occurrences (n(L)): {results['min_distance_count_n_L']}")
print(f"Force Criterion (G(L)): {results['force_criterion_G_L']:.3f}")

def get_lognormal_params(mean, std):
    sigma = np.sqrt(np.log(np.sqrt(1 + (std / mean)**2)))
    mu = np.log(mean) - 0.5 * sigma**2
    return sigma, np.exp(mu)

def lognormal_params(mean, std):
    mu = np.log((mean**2) / np.sqrt(std**2 + mean**2))
    sigma_sq = np.log(1 + (std**2) / (mean**2))
    return np.sqrt(sigma_sq), np.exp(mu)

# Gumbel: mean, std → loc, scale
def get_gumbel_params(mean, std):
    beta = std * np.sqrt(6) / np.pi
    loc = mean - 0.5772 * beta
    return loc, beta

# Uniform: mean ± sqrt(3)*std
def get_uniform_bounds(mean, std):
    delta = np.sqrt(3) * std
    return mean - delta, mean + delta

# Column order
col_order = ['sigma_mem_y', 'f_mem', 'sigma_mem', 'E_mem', 'nu_mem', 'sigma_edg', 'sigma_sup']

# -----------------------------
# Define Lognormal Parameters
# -----------------------------
lognorm_vars = {
    'sigma_mem_y': get_lognormal_params(11000, 1650),
    'sigma_mem': get_lognormal_params(4000, 800),
    'E_mem': get_lognormal_params(600000, 90000),
    'sigma_edg': get_lognormal_params(353677.6513, 70735.53026),
    'sigma_sup': get_lognormal_params(400834.6715, 80166.9343),
    'f_mem': get_gumbel_params(0.4, 0.12),
    'nu_mem': get_uniform_bounds(0.4, 0.0115)
}

olhs_df = pd.DataFrame(optimized_lhs, columns=col_order)
olhs_df['sigma_mem_y'] = lognorm.ppf(optimized_lhs[:, 0], s=lognorm_vars['sigma_mem_y'][0], scale=lognorm_vars['sigma_mem_y'][1])
olhs_df['f_mem'] = gumbel_r.ppf(optimized_lhs[:, 1], loc=lognorm_vars['f_mem'][0], scale=lognorm_vars['f_mem'][1])
olhs_df['sigma_mem'] = lognorm.ppf(optimized_lhs[:, 2], s=lognorm_vars['sigma_mem'][0], scale=lognorm_vars['sigma_mem'][1])
olhs_df['E_mem'] = lognorm.ppf(optimized_lhs[:, 3], s=lognorm_vars['E_mem'][0], scale=lognorm_vars['E_mem'][1])
olhs_df['nu_mem'] = uniform.ppf(optimized_lhs[:, 4], loc=lognorm_vars['nu_mem'][0], scale=lognorm_vars['nu_mem'][1] - lognorm_vars['nu_mem'][0])
olhs_df['sigma_edg'] = lognorm.ppf(optimized_lhs[:, 5], s=lognorm_vars['sigma_edg'][0], scale=lognorm_vars['sigma_edg'][1])
olhs_df['sigma_sup'] = lognorm.ppf(optimized_lhs[:, 6], s=lognorm_vars['sigma_sup'][0], scale=lognorm_vars['sigma_sup'][1])

olhs_df.to_csv("design.csv", index=False)