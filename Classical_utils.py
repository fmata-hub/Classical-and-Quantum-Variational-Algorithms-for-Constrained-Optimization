
import numpy as np
import itertools
import yfinance as yf
import pandas as pd
import random

# --- COST FUNCTION (Classical Hamiltonian) ---
def get_cost(x, mu, sigma, q, K, lmbda):
    """
    Computes the cost of a portfolio configuration x (binary vector).
    """
    # Risk term: x^T * Sigma * x
    risk = q * np.dot(x.T, np.dot(sigma, x))
    
    # Return term: mu * x
    expected_return = np.dot(mu, x)
    
    # Constraint penalty: lambda * (sum(x) - K)^2
    constraint = lmbda * (np.sum(x) - K)**2
    
    return risk - expected_return + constraint

def show_data(tickers, mu, sigma):
    """
    Displays expected annual returns and covariance matrix.
    """
    print("--- REAL-WORLD DATA EXTRACTED ---")
    
    for i, ticker in enumerate(tickers):
        print(f"{ticker}: Expected Annual Return = {mu[i]:.4f}")
    
    print("\nCovariance Matrix (Sigma):")
    print(sigma)
