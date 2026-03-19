

import numpy as np
import itertools
import yfinance as yf
import pandas as pd
import random
from collections import Counter
import matplotlib.pyplot as plt


def simulated_annealing_stats(get_cost, tickers, mu, sigma, q, K, lmbda,
                              T_start=10.0, T_end=0.001, cooling_rate=0.99, iterations=1000):
    """
    Simulated Annealing for portfolio optimization with statistics.

    Returns:
        best_state: best portfolio found
        best_cost: cost of best portfolio
        history: list of costs over iterations
        ticker_counts: how often each ticker was selected
        combo_counter: Counter of portfolio combinations
    """
    n_assets = len(tickers)
    
    current_state = np.random.randint(2, size=n_assets)
    current_cost = get_cost(current_state, mu, sigma, q, K, lmbda)

    best_state = np.copy(current_state)
    best_cost = current_cost

    history = []
    state_history = []
    ticker_counts = np.zeros(n_assets)
    combo_counter = Counter()

    T = T_start

    while T > T_end:
        for i in range(iterations):
            #n--- Generate next state intelligently ---
            next_state = np.copy(current_state)

            if np.sum(current_state) != K:
                idx = np.random.randint(n_assets)
                next_state[idx] = 1 - next_state[idx]
            else:
                ones_idx = np.where(current_state == 1)[0]
                zeros_idx = np.where(current_state == 0)[0]

                if len(ones_idx) > 0 and len(zeros_idx) > 0:
                    idx1 = np.random.choice(ones_idx)
                    idx0 = np.random.choice(zeros_idx)
                    next_state[idx1], next_state[idx0] = next_state[idx0], next_state[idx1]

            # Compute cost
            next_cost = get_cost(next_state, mu, sigma, q, K, lmbda)
            delta_e = next_cost - current_cost

            # Metropolis acceptance
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
                current_state = next_state
                current_cost = next_cost

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_state = np.copy(current_state)

            # Record statistics
            state_history.append(tuple(current_state))
            ticker_counts += current_state
            combo_counter[tuple(current_state)] += 1

        history.append(current_cost)
        T *= cooling_rate

    return best_state, best_cost, history, ticker_counts, combo_counter


def plot_sa_results(best_state, best_cost, tickers, K, history, ticker_counts, combo_counter):
    """
    Plots statistics from Simulated Annealing:
    - Cost evolution
    - Ticker selection frequency
    - Top 5 portfolio combinations
    """
    # --- Portfolio summary ---
    chosen_tickers = [tickers[i] for i, val in enumerate(best_state) if val == 1]
    print("\n--- SIMULATED ANNEALING RESULTS ---")
    print(f"Best portfolio found: {best_state}")
    print(f"Cost: {best_cost:.4f}")
    print(f"Selected tickers: {chosen_tickers}")
    print(f"Meets K={K} constraint?: {'YES' if sum(best_state) == K else 'NO'}")

    # --- Plot cost evolution ---
    plt.figure(figsize=(10,4))
    plt.plot(history)
    plt.title("Cost Evolution Over Iterations")
    plt.xlabel("Cooling Steps")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.show()

    # --- Plot ticker selection frequency ---
    plt.figure(figsize=(8,4))
    plt.bar(tickers, ticker_counts / sum(ticker_counts) * 100)
    plt.title("Ticker Selection Frequency (%)")
    plt.ylabel("Percentage")
    plt.xlabel("Tickers")
    plt.grid(axis='y')
    plt.show()

    # --- Top portfolio combinations (fixed top 5) ---
    print("\nTop portfolio combinations:")
    for combo, count in combo_counter.most_common(5):
        pct = count / sum(combo_counter.values()) * 100
        print(f"{combo} -> {count} times ({pct:.2f}%)")