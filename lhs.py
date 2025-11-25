import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import qmc, lognorm, gumbel_r, uniform

# -----------------------------
# CONFIG
# -----------------------------
n = 150      # number of design points
d = 7        # number of variables
tries = 50   # number of LHS samples to evaluate

# -----------------------------
# INPUT DISTRIBUTIONS
# -----------------------------

# Lognormal: mean, std → shape (sigma), scale (exp(mu))
def get_lognormal_params(mean, std):
    sigma = np.log(np.sqrt(1 + (std / mean)**2))
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

# Gumbel: Mean and std for f_mem
gumbel_mean, gumbel_std = 0.4, 0.12
beta = gumbel_std * np.sqrt(6) / np.pi
loc_gumbel = gumbel_mean - 0.5772 * beta

# Uniform: Mean and std for nu_mem
nu_mean, nu_std = 0.4, 0.0115
a_uni = nu_mean - np.sqrt(3) * nu_std
b_uni = nu_mean + np.sqrt(3) * nu_std

# -----------------------------
# LHS + MAXIMIN SEARCH
# -----------------------------

def lhs_maximin(n, d, tries=50):
    best = None
    best_score = -np.inf
    for _ in range(tries):
        sampler = qmc.LatinHypercube(d)  # LHS sampler from SciPy
        sample = sampler.random(n)  # Generate n points in [0,1]^d
        dist = pdist(sample)  # Calculate pairwise distance
        min_dist = dist.min()  # Find the smallest distance
        if min_dist > best_score:
            best_score = min_dist  # Keep track of the best maximin score
            best = sample  # Store the best LHS sample
    return best

X_prob = lhs_maximin(n, d, tries)

# -----------------------------
# INVERSE TRANSFORM TO PHYSICAL VALUES
# -----------------------------
X_phys = pd.DataFrame(columns=col_order)

# Applying transformations based on the new formulas
X_phys['sigma_mem_y'] = lognorm.ppf(X_prob[:, 0], s=lognorm_vars['sigma_mem_y'][0], scale=lognorm_vars['sigma_mem_y'][1])
X_phys['f_mem'] = gumbel_r.ppf(X_prob[:, 1], loc=loc_gumbel, scale=beta)
X_phys['sigma_mem'] = lognorm.ppf(X_prob[:, 2], s=lognorm_vars['sigma_mem'][0], scale=lognorm_vars['sigma_mem'][1])
X_phys['E_mem'] = lognorm.ppf(X_prob[:, 3], s=lognorm_vars['E_mem'][0], scale=lognorm_vars['E_mem'][1])
X_phys['nu_mem'] = uniform.ppf(X_prob[:, 4], loc=a_uni, scale=b_uni - a_uni)
X_phys['sigma_edg'] = lognorm.ppf(X_prob[:, 5], s=lognorm_vars['sigma_edg'][0], scale=lognorm_vars['sigma_edg'][1])
X_phys['sigma_sup'] = lognorm.ppf(X_prob[:, 6], s=lognorm_vars['sigma_sup'][0], scale=lognorm_vars['sigma_sup'][1])

# -----------------------------
# MINIMAX - FILLING GAPS

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# Function to add multiple points using Minimax (50 iterations)
def add_minimax_points(X_phys, n_new_points=5, min_distance=1e-4):
    added_points = 0  # Track how many new points have been added
    
    while added_points < n_new_points:
        # Calculate pairwise distances
        dist_matrix = squareform(pdist(X_phys))
        # Set diagonal to a large number to avoid self-comparison
        np.fill_diagonal(dist_matrix, np.inf)

        # Find largest gap (max distance between any two points)
        max_gap_idx = np.unravel_index(np.argmax(dist_matrix, axis=None), dist_matrix.shape)

        # Get midpoint between the points with the largest gap
        point_a = X_phys.iloc[max_gap_idx[0]]  # Access the first index
        point_b = X_phys.iloc[max_gap_idx[1]]  # Access the second index

        # Average the values (or other strategies, e.g., interpolation, etc.)
        midpoint = (point_a + point_b) / 2
        
        # Check if the midpoint is unique (not too close to existing points)
        # Concatenate the midpoint to the existing dataframe
        X_phys_temp = pd.concat([X_phys, midpoint.to_frame().T], ignore_index=True)
        dist_to_existing = pdist(X_phys_temp)  # Calculate distance to all other points

        # Ensure new point is sufficiently far from existing points
        if np.all(dist_to_existing >= min_distance):  # Ensure the new point is not too close
            X_phys = X_phys_temp  # Accept the new point
            added_points += 1  # Successfully added a new point
        else:
            # If the midpoint is too close to any existing point, retry with the next largest gap
            continue
    
    return X_phys

# Adding 50 minimax points (to go from 150 to 200 points)
# X_phys = add_minimax_points(X_phys, n_new_points=50)

# -----------------------------
# SAVE OR VIEW
# -----------------------------
X_phys = X_phys.round(2)
print(X_phys.head())

# To save the updated design with 200 points:
X_phys.to_csv("lhs_maximin_minimax_sunsail_200.csv", index=False)