import numpy as np

def sortedEigSym(data: np.ndarray, ascending: bool):
    """
    Performs an eigenvalue decomposition,
    but the result is sorted ascending from smallest to biggest (the breeze eig function sorts also ascending but the absolute value!)

    @param data The matrix has to be symmetric!
    """
    eigenvalues, eigenvectors = np.linalg.eigh(data)
    sortedRes = np.argsort(eigenvalues) if ascending else np.argsort(-eigenvalues)
    
    eigvals = eigenvalues[sortedRes]
    eigvecs = eigenvectors[sortedRes]
    return eigvals, eigvecs
