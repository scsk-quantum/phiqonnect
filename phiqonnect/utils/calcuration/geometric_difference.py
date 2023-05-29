import numpy as np

def geometric_diference(K1, K2):
    """行列1, 行列2のジオメトリックディファレンスを計算する
        
	Parameters
	----------
	K1: array
		行列1
	K2: array
		行列2

	Returns
	-------
	result: float
		行列1, 行列2のジオメトリックディファレンス
	"""
    u, s, v = np.linalg.svd(K2, hermitian=True)
    K2_sqrt = u @ np.diag(np.sqrt(s)) @ v
    g = np.sqrt(np.linalg.norm(K2_sqrt @ np.linalg.inv(K1) @ K2_sqrt, np.inf))
    
    return g