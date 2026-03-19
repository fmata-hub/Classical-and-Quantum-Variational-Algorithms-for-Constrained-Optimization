

import numpy as np
import itertools
import yfinance as yf
import pandas as pd
import random
from collections import Counter
import matplotlib.pyplot as plt


def show_results_brute_force(results_sorted, tickers, K):
    """
    Displays the top portfolio configurations.
    """
    print("--- PORTFOLIO RANKING ---")
    
    for i in range(len(results_sorted)):
        combo, cost = results_sorted[i]
        
        # Check constraint
        if sum(combo) == K:

            status = "VALID"
            # Map to tickers
            chosen_tickers = [tickers[j] for j, val in enumerate(combo) if val == 1]
            
            print(f"Top {i+1}: {combo} | Cost: {cost:.4f} | {status} | Assets: {chosen_tickers}")

    # Best solution
    best_classic_x = results_sorted[0][0]
    print(f"\n>>> BEST CLASSICAL SOLUTION: {best_classic_x}")
    
    return best_classic_x


def brute_force_solver(get_cost,tickers, mu, sigma, q, K, lmbda):
    """
    Solves the portfolio optimization problem via brute force.
    Returns sorted results and the best solution.
    """
    
    n_assets = len(tickers)
    all_combinations = list(itertools.product([0, 1], repeat=n_assets))

    print(f"We have {len(all_combinations)} possible portfolios\n")

    results = []
    for combo in all_combinations:
        x = np.array(combo)
        cost = get_cost(x, mu, sigma, q, K, lmbda)
        results.append((combo, cost))

    # Sort portfolios by cost (ascending: best first)
    results_sorted = sorted(results, key=lambda x: x[1])

    return results_sorted

def brute_force_solver_smart(get_cost, tickers, mu, sigma, q, K, lmbda):
    n_assets = len(tickers)
    # Genera combinaciones de índices, ej: (0, 1, 2), (0, 1, 3)...
    possible_indices = list(itertools.combinations(range(n_assets), K))
    
    print(f"Checking only {len(possible_indices)} valid portfolios (K={K})")

    results = []
    for indices in possible_indices:
        # Convertimos los índices (0,2,4) a vector binario [1,0,1,0,1,0...]
        x = np.zeros(n_assets)
        x[list(indices)] = 1
        
        cost = get_cost(x, mu, sigma, q, K, lmbda)
        results.append((tuple(x.astype(int)), cost))

    return sorted(results, key=lambda x: x[1])