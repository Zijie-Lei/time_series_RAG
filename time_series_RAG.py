import pycatch22
from fastdtw import fastdtw
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import ast
import os
from chronos import ChronosPipeline
import torch
import random
from typing import List, Union
import matplotlib.pyplot as plt

# class TimeSeriesRetriever:
#     def __init__(self, knowledge_source=None, feature_source=None, feature_output=None, knowledge_output=None, alpha=0.5, M=20, embedding_model=None):
#         """
#         knowledge_source: Path to CSV containing time series data or a list of time series clips (NumPy arrays)
#         feature_source: Path to a precomputed feature embedding (NumPy array)
#         knowledge_output: Path to store the processed knowledge base (as a NumPy array)
#         alpha: Weight parameter to balance Catch-22 and DTW similarities
#         M: Number of top candidates to keep after Catch-22 filtering
#         """
#         self.alpha = alpha
#         self.M = M

#         # Load knowledge base from CSV if a path is provided
#         if isinstance(knowledge_source, str) and os.path.exists(knowledge_source):
#             df = pd.read_csv(knowledge_source)
#             df['Hist'] = df['Hist'].fillna('[]')  # Impute missing values
#             self.knowledge_base = df['Hist'].apply(lambda x: np.array(ast.literal_eval(x))).tolist()
#         elif isinstance(knowledge_source, list) or isinstance(knowledge_source, np.ndarray):
#             self.knowledge_base = knowledge_source
#         else:
#             raise ValueError("Invalid knowledge_source provided. Must be a valid path or a list of time series clips.")

#         # Store knowledge base to disk if an output path is provided
#         if knowledge_output:
#             np.save(knowledge_output, self.knowledge_base)

#         # Load precomputed features if a path is provided
#         if feature_source and os.path.exists(feature_source):
#             self.feature_vectors = np.load(feature_source)
#         else:
#             if embedding_model == 'catch22':

#                 self.feature_vectors = np.array([pycatch22.catch22_all(ts)['values'] for ts in self.knowledge_base])
#             elif embedding_model == 'chronos':
#                 pipeline = ChronosPipeline.from_pretrained(
#                     "amazon/chronos-t5-tiny",
#                     device_map="cpu",
#                     torch_dtype=torch.bfloat16,
#                 )
#                 self.feature_vectors = np.array([pipeline.embed(torch.tensor(ts)) for ts in self.knowledge_base])
#             else:
#                 raise ValueError("Invalid embedding_model provided. Must be 'catch22' or 'chronos'.")
            
#             # Handle NaN values by replacing with column means
#             nan_mask = np.isnan(self.feature_vectors)
#             col_means = np.nanmean(self.feature_vectors, axis=0)
#             self.feature_vectors[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
            
#             # Save computed features for future use
#             if feature_output:
#                 np.save(feature_output, self.feature_vectors)

#         # Normalize features
#         self.scaler = StandardScaler()
#         self.feature_vectors = self.scaler.fit_transform(self.feature_vectors)

#     def retrieve_top_k(self, query_ts, k=5):
#         """
#         Retrieve the top-K most similar time series clips to the query time series.
#         """
#         # Compute Catch-22 features for the query and replace NaNs
#         query_features = pycatch22.catch22_all(query_ts)['values']
#         query_features = np.nan_to_num(query_features, nan=np.nanmean(query_features))  # Replace NaNs

#         # Normalize query features
#         query_features = self.scaler.transform([query_features])[0]

#         # Compute Catch-22 similarity using Cosine Similarity
#         catch22_similarities = cosine_similarity([query_features], self.feature_vectors)[0]

#         # Retrieve the top-M most similar based on Catch-22 features
#         top_m_indices = np.argsort(catch22_similarities)[-self.M:][::-1]

#         # Compute DTW only for the top-M candidates
#         dtw_distances = np.array([fastdtw(query_ts, self.knowledge_base[i])[0] for i in top_m_indices])
#         dtw_similarities = 1 / (1 + dtw_distances)  

#         # Compute the final combined similarity score
#         combined_similarity = self.alpha * catch22_similarities[top_m_indices] + (1 - self.alpha) * dtw_similarities

#         # Retrieve top-K indices within the M candidates
#         final_top_k_indices = top_m_indices[np.argsort(combined_similarity)[-k:][::-1]]

#         return [(i, combined_similarity[np.where(top_m_indices == i)[0][0]]) for i in final_top_k_indices]
    
#     def compute_similarity(self, ts1, ts2):
#         """
#         Compute similarity between two time series based on Catch-22 and DTW metrics.
#         """
#         # Compute Catch-22 features for both time series
#         features1 = pycatch22.catch22_all(ts1)['values']
#         features2 = pycatch22.catch22_all(ts2)['values']
        
#         # Handle NaN values
#         features1 = np.nan_to_num(features1, nan=np.nanmean(features1))
#         features2 = np.nan_to_num(features2, nan=np.nanmean(features2))

#         # Normalize features
#         features1 = self.scaler.transform([features1])[0]
#         features2 = self.scaler.transform([features2])[0]

#         # Compute Catch-22 similarity using Cosine Similarity
#         catch22_similarity = cosine_similarity([features1], [features2])[0][0]

#         # Compute DTW distance and similarity
#         dtw_distance, _ = fastdtw(ts1, ts2)
#         dtw_similarity = 1 / (1 + dtw_distance)

#         # Compute the final combined similarity score
#         similarity_score = self.alpha * catch22_similarity + (1 - self.alpha) * dtw_similarity

#         return similarity_score

class TimeSeriesRetriever:
    def __init__(self, knowledge_source=None, feature_source=None, feature_output=None, knowledge_output=None, alpha=0.5, M=20, embedding_model=None):
        """
        knowledge_source: Path to CSV containing time series data or a list of time series clips (NumPy arrays)
        feature_source: Path to a precomputed feature embedding (NumPy array)
        knowledge_output: Path to store the processed knowledge base (as a NumPy array)
        alpha: Weight parameter to balance feature similarity and DTW similarities
        M: Number of top candidates to keep after feature filtering
        embedding_model: Feature extraction model ('catch22', 'chronos')
        """
        self.alpha = alpha
        self.M = M
        self.embedding_model = embedding_model

        # Load knowledge base from CSV if a path is provided
        if isinstance(knowledge_source, str) and os.path.exists(knowledge_source):
            df = pd.read_csv(knowledge_source)
            df['Hist'] = df['Hist'].fillna('[]')  # Impute missing values
            self.knowledge_base = df['Hist'].apply(lambda x: np.array(ast.literal_eval(x))).tolist()
        elif isinstance(knowledge_source, list) or isinstance(knowledge_source, np.ndarray):
            self.knowledge_base = knowledge_source
        else:
            raise ValueError("Invalid knowledge_source provided. Must be a valid path or a list of time series clips.")

        # Store knowledge base to disk if an output path is provided
        if knowledge_output:
            np.save(knowledge_output, self.knowledge_base)

        # Load precomputed features if a path is provided
        if feature_source and os.path.exists(feature_source):
            self.feature_vectors = np.load(feature_source)
        else:
            self.feature_vectors = self._compute_features(self.knowledge_base)
            if feature_output:
                np.save(feature_output, self.feature_vectors)

        # Normalize features
        # self.scaler = StandardScaler()
        # self.feature_vectors = self.scaler.fit_transform(self.feature_vectors)

    def _compute_features(self, time_series_list):
        if self.embedding_model == 'catch22':
            features = np.array([pycatch22.catch22_all(ts)['values'] for ts in time_series_list])
        elif self.embedding_model == 'chronos':
            pipeline = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-tiny",
                device_map="cpu",
                torch_dtype=torch.float16,
            )
            features = np.vstack([pipeline.embed(torch.tensor(ts))[0].flatten(start_dim=1, end_dim=-1).float().numpy() for ts in time_series_list])
            # features = features.flatten(start_dim=1, end_dim=-1)
        else:
            raise ValueError("Invalid embedding_model provided. Must be 'catch22' or 'chronos'.")
        
        # Handle NaN values by replacing with column means
        nan_mask = np.isnan(features)
        col_means = np.nanmean(features, axis=0)
        features[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        
        return features

    def retrieve_top_k(self, query_ts, k=5):
        """
        Retrieve the top-K most similar time series clips to the query time series.
        """
        query_features = self._compute_features([query_ts])[0]
        # query_features = self.scaler.transform([query_features])[0]

        feature_similarities = cosine_similarity([query_features], self.feature_vectors)[0]
        top_m_indices = np.argsort(feature_similarities)[-self.M:][::-1]

        dtw_distances = np.array([fastdtw(query_ts, self.knowledge_base[i])[0] for i in top_m_indices])
        dtw_similarities = 1 / (1 + dtw_distances)  

        combined_similarity = self.alpha * feature_similarities[top_m_indices] + (1 - self.alpha) * dtw_similarities
        final_top_k_indices = top_m_indices[np.argsort(combined_similarity)[-k:][::-1]]

        return [(i, combined_similarity[np.where(top_m_indices == i)[0][0]]) for i in final_top_k_indices]

    def compute_similarity(self, ts1, ts2):
        """
        Compute similarity between two time series based on selected feature embedding and DTW metrics.
        """
        features1 = self._compute_features([ts1])[0]
        features2 = self._compute_features([ts2])[0]

        features1 = self.scaler.transform([features1])[0]
        features2 = self.scaler.transform([features2])[0]

        feature_similarity = cosine_similarity([features1], [features2])[0][0]
        
        dtw_distance, _ = fastdtw(ts1, ts2)
        dtw_similarity = 1 / (1 + dtw_distance)

        similarity_score = self.alpha * feature_similarity + (1 - self.alpha) * dtw_similarity
        
        return similarity_score


def apply_kernel_with_noise(
    data: Union[List[float], np.ndarray], 
    kernel_type: str = "gaussian",
    apply_to_all: bool = True, 
    sample_fraction: float = 0.5,
    **kwargs
) -> np.ndarray:
    """
    Applies a specified kernel function to add noise to a NumPy array of floats.
    
    Args:
        data (List[float] | np.ndarray): The original list or NumPy array of floats.
        kernel_type (str): Type of noise kernel to apply. Options:
                           "gaussian", "uniform", "laplace", "salt_and_pepper",
                           "multiplicative", "exponential", "poisson".
        apply_to_all (bool): If True, applies the kernel to all data points. 
                             If False, applies the kernel to a random sample.
        sample_fraction (float): Fraction of points to apply the kernel to if apply_to_all is False.
                                 Should be between 0 and 1.
        **kwargs: Additional parameters for the noise functions (e.g., std, range).
    
    Returns:
        np.ndarray: The modified array with noise applied.
    """
    
    # Convert input data to a NumPy array if it is not already
    data = np.array(data, dtype=np.float64)

    # Define available kernel functions
    def gaussian_noise(x, mean=0, std=0.1):
        return x + np.random.normal(mean, std)

    def uniform_noise(x, lower=-0.1, upper=0.1):
        return x + np.random.uniform(lower, upper)

    def laplace_noise(x, scale=0.1):
        return x + np.random.laplace(0, scale)

    def salt_and_pepper_noise(x, prob=0.1, low=-1.0, high=1.0):
        mask = np.random.rand(*x.shape) < prob
        salt_or_pepper = np.random.choice([low, high], size=x.shape)
        return np.where(mask, salt_or_pepper, x)

    def multiplicative_noise(x, factor=0.1):
        return x * (1 + np.random.uniform(-factor, factor, size=x.shape))

    def exponential_noise(x, scale=0.1):
        return x + np.random.exponential(scale, size=x.shape)

    def poisson_noise(x, lam=1):
        return x + np.random.poisson(lam, size=x.shape)
    
    def no_noise(x):
        return x
    
    # Kernel function mapping
    kernel_functions = {
        "original": no_noise,
        "gaussian": gaussian_noise,
        "uniform": uniform_noise,
        "laplace": laplace_noise,
        "salt_and_pepper": salt_and_pepper_noise,
        "multiplicative": multiplicative_noise,
        "exponential": exponential_noise,
        "poisson": poisson_noise
    }
    
    # Select kernel function
    kernel = kernel_functions.get(kernel_type)
    
    if not kernel:
        raise ValueError(f"Invalid kernel type '{kernel_type}'. Choose from {list(kernel_functions.keys())}.")

    # Apply noise function
    if apply_to_all:
        return kernel(data, **kwargs)
    else:
        sample_size = max(1, int(len(data) * sample_fraction))
        sampled_indices = np.random.choice(len(data), size=sample_size, replace=False)
        noisy_data = data.copy()
        noisy_data[sampled_indices] = kernel(noisy_data[sampled_indices], **kwargs)
        return noisy_data
    
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
