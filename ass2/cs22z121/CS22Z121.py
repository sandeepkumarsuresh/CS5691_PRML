
"""

Contains all the necessary functions for finding number of PCA Components and Finding the Reconstruction Error

To Run the Code (In Tereminal):

                python3 cs22z121.py



"""



# Importing all the necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eigh # for finding the eigenvalue and eigenvector

# Defining the Function to Compute Mean , Covariance and Standard Deviation

def ComputeMean(data):
    """ Function to get data mean
    
    Arguments:

    data: Dataset

    Returns:
    
    mean of the data
    
    
    """
    return np.sum(data,axis=0)/len(data)

def Covariance(data):
    """ Function to get the Covariance Matrix
    
    Arguments:

    data: Dataset

    Returns:
    cov: The covariance Matrix
    
    
    """
    N, M = data.shape
    cov = np.zeros((M, M))
    for i in range(M):
        mean_i = np.sum(data[:, i]) / N
        for j in range(M):
            mean_j = np.sum(data[:, j]) / N
            cov[i, j] = np.sum((data[:, i] - mean_i) * (data[:, j] - mean_j)) / (N - 1)
    return cov  




def get_EV_and_Evectors(Covariance_Matrix):
    """Function to get the eigenvalue and Eigenvectors
    
    Arguments:

    Data: Mean-Centered Data

    Returns:
    EV: The Sorted EigenValue
    Evectors: The Sorted EigenVectors
    
    """

    eigenvalue , eigenvector =  eigh(Covariance_Matrix)
    sorted_index = np.argsort(eigenvalue)[::-1]
    EV = eigenvalue[sorted_index]
    EVectors = eigenvector[:,sorted_index]

    return EV , EVectors


def PCA(data):
    """ Computes the likelihood score of a data point with respect to a given class
    given the class' mean and covariance matrix

    Arguments:
    Data: Dataset



    Returns:
    numPCs: number of principal components that contribute to 90% of the varaince in the dataset.

    """
    data_mean = ComputeMean(data)
    data_centered = data_raw - data_mean
    data_centered_cov = Covariance(data_centered)



    sorted_eigenvalue,sorted_eigenvectors = get_EV_and_Evectors(data_centered_cov)



    total_variance = np.sum(sorted_eigenvalue)
    explained_variance_ratio = sorted_eigenvalue / total_variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
    print(n_components_90)
    return n_components_90 

def Reconstruction_Error(data , n_components):

    """
    Perform reconstruction of data using the dimensionality-reduced data considering the number of dimensions [2,4,8,16]. 
    
    Report the Mean Square Error (MSE) between the original data and reconstructed data,
    
    and interpret the optimal dimensions d^ based on the MSE values.
    
    """
    dimensions = [2, 4, 8, 16,21,64]
    mse_values = {}

    data_mean = ComputeMean(data)
    data_centered = data_raw - data_mean
    data_centered_cov = Covariance(data_centered)

    sorted_eigenvalue,sorted_eigenvectors = get_EV_and_Evectors(data_centered_cov)





    for n_components in dimensions:
        selected_eigenvectors = sorted_eigenvectors[:, :n_components]        
        reduced_data = np.dot(data_centered, selected_eigenvectors)        
        reconstructed_data = np.dot(reduced_data, selected_eigenvectors.T)        
        reconstructed_data = (reconstructed_data) + data_mean        
        mse = np.mean(np.square(data_raw - reconstructed_data))        
        mse_values[n_components] = mse
    for n_components, mse in mse_values.items():
        print(f"Dimension {n_components}: MSE = {mse:.4f}")

    # For Optimal Decision

    optimal_dimension = min(mse_values, key=mse_values.get)
    print(f"Optimal dimension based on MSE: {optimal_dimension}")



if __name__ == '__main__':
    data_path = 'Data/arr_0.npy'
    data_raw = np.load(data_path)
    data_raw.shape
    n_PCA_Componenets = PCA(data_raw)
    Reconstruction_Error(data_raw,n_PCA_Componenets)