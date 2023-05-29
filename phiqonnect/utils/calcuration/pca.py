from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np

def pca_data(sample_train, sample_test, n, only_normalize=False):
    """データにpcaをかける

    Parameters
    ----------
    sample_train: array
        学習データ
    sample_test: array
        テストデータ
    n: int
        次元削減後の次元
    only_normalize: boolean
        ノーマライズのみ行うかどうか

    Returns
    -------
    result: float
        行列1, 行列2のジオメトリックディファレンス
	"""
    # data = np.concatenate([sample_train, sample_test])
    data = sample_train
    
    std_scale = StandardScaler().fit(data)
    data = std_scale.transform(data)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)
    
    if only_normalize:
        pca=None
    else:
        pca = PCA(n_components=n, svd_solver="full").fit(data)
        sample_train = pca.transform(sample_train)
        sample_test = pca.transform(sample_test)
    
    samples = np.append(sample_train, sample_test, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    sample_train = minmax_scale.transform(sample_train)
    sample_test = minmax_scale.transform(sample_test)
    
    return sample_train, sample_test, (pca, std_scale, minmax_scale)
    

def reverse_pca(pca_data, pca) :
    """(pcaをかけた)データを元の次元に射影する
        
	Parameters
	----------
	pca_data: array
		データ
	pca: sklearn.decomposition.PCA
		pcaモデル

	Returns
	-------
	result: array
		元の次元に射影したデータ
	"""
    if pca[0] == None:
        std_scale = pca[1]
        minmax_scale = pca[2]
        data = pca_data
        data = minmax_scale.inverse_transform(data)
        data = std_scale.inverse_transform(data)
    else:
        std_scale = pca[1]
        minmax_scale = pca[2]
        pca = pca[0]
        data = pca_data
        data = minmax_scale.inverse_transform(data)
        data = pca.inverse_transform(data)
        data = std_scale.inverse_transform(data)
        
    return data