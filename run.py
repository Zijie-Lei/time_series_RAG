import numpy as np
import pandas as pd
import ast
from time_series_RAG import TimeSeriesRetriever, apply_kernel_with_noise
import matplotlib.pyplot as plt
import torch

def plot_top_k_time_series(hist, top_k_indices):
    """
    Plots the top-k most similar time series clips.
    
    Parameters:
        hist (np.array): np array containing time series data.
        top_k_indices (list of tuples): List of (index, similarity score) pairs.
        k (int): Number of top similar time series to plot (default is 5).
    """
    
    plt.figure(figsize=(10, 6))
    for idx, sim in top_k_indices:
        print(f"Index: {idx}, Similarity: {sim:.4f}")
        plt.plot(hist[idx], label=f'Array {idx} (Sim: {sim:.4f})')
    
    # Labels and legend
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.title(f"Plot of Retrieved Time Series Clips")
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()

def percentage_in_range(index: int, fetch_indices: list, k:int) -> float:
    if not fetch_indices:
        return 0.0  # Avoid division by zero
    
    count_in_range = sum(index * 8 <= x <= index * 8 + 8 for x in fetch_indices)
    percentage = (count_in_range / len(fetch_indices)) * 100
    
    return percentage

def compute_mean_percentage_in_range(hist, retriever, k=8):
    """
    Computes the mean percentage of retrieved top-k indices that fall within the valid range.

    Args:
        hist (list): List of time series history.
        retriever: An object that has a `retrieve_top_k` method.
        k (int): The number of top-k indices to retrieve.

    Returns:
        float: Mean percentage of retrieved indices in range.
    """
    percentages = []
    num_queries = len(hist)

    for index in range(num_queries):
        query_ts = hist[index]
        top_k_indices = retriever.retrieve_top_k(query_ts, k=k)
        percentage = percentage_in_range(index, [x[0] for x in top_k_indices], k=k)
        percentages.append(percentage)

        # Display progress
        progress = (index + 1) / num_queries * 100
        print(f"Progress: {progress:.2f}% completed", end="\r")

    mean_percentage = np.mean(percentages)
    print(f"\nMean Percentage in range: {mean_percentage:.2f}%")

    return mean_percentage

df = pd.read_csv('data/MSPG.csv')
hist = np.vstack(df['Hist'].apply(lambda x: np.array(ast.literal_eval(x))).to_numpy())
hist = hist[~np.all(hist == 0, axis=1)]

kernel_types = ["original", "gaussian", "uniform", "laplace", "salt_and_pepper",
                "multiplicative", "exponential", "poisson"]

# Stack the noisy arrays
original_data = hist
noisy_data_list = []

for row in original_data:
    noisy_data = np.array([apply_kernel_with_noise(row, kernel_type=kernel) for kernel in kernel_types])
    noisy_data_list.append(noisy_data)

# Convert to final shape (8, m, n)
final_noisy_data = np.array(noisy_data_list)
final_noisy_data_reshaped = final_noisy_data.reshape(8 * original_data.shape[0], original_data.shape[1])


alpha = [0.0, 0.2, 0.5, 0.8, 1.0]
### 0.0 alpha = Full DTW
### 1.0 alpha = Full Cosine Similarity
embedding_model = ['catch22', "chronos"]
for a in alpha:
    for embedding in embedding_model:
        retriever = TimeSeriesRetriever(
            knowledge_source=final_noisy_data_reshaped, 
            # knowledge_output="data/MSPG_knowledge", 
            # feature_output="data/MSPG_features_chronos",
            # feature_source="data/MSPG_features_chronos.npy",
            alpha=a, 
            M=200,
            embedding_model=embedding
        )
        print('Performance for Alpha:', a, 'Embedding:', embedding)
        compute_mean_percentage_in_range(original_data, retriever, k=8)

### accuracy goes down when sample size goes up