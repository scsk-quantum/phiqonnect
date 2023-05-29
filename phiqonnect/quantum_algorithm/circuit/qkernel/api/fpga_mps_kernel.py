import numpy as np
import requests
import json

class FPGA_MPS_Kernel():
    """
    API経由でFPGA_MPS_Kernelを使用する

    Parameters
	----------
    endpoint: str
        エンドポイントのURL
    api_key: str
        API KEY
    feature_dimension: int
        特徴量の次元
    block_size: int
        ブロックサイズ
    save_kernel_per_block: boolean
        ブロックごとにカーネル行列を保存するか
    product_flag: boolean
        カーネルの積を取るかどうか
    """
    def __init__(self, endpoint="", api_key="", feature_dimension=2, block_size=2, save_kernel_per_block = False, product_flag=True) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.feature_dimension = feature_dimension
        self.block_size = block_size
        self.save_kernel_per_block = save_kernel_per_block
        self.product_flag = product_flag
        
        response = requests.get(self.endpoint + 'health', params={'api_key': self.api_key})

        if(response.status_code != 200):
            raise Exception(response)
            
        print("[Health Check Complete]", "dim:", feature_dimension, "nblock:", block_size)
    
    def get_kernel_matrix(self, x1_vec, x2_vec,instanse=None):
        mat = []
        
        if x2_vec is None:
            x2_vec = x1_vec.copy()
            sym = True
        else:
            sym = False
        
        print("[Get Kernel Matrix started]")
        response = requests.get(self.endpoint + 'get_kernel_matrix', params={'api_key': self.api_key, 
                                                                             'x1_vec': json.dumps(x1_vec, cls=MyEncoder), 
                                                                             'x2_vec': json.dumps(x2_vec, cls=MyEncoder),
                                                                             'dim': self.feature_dimension,
                                                                             'kernel': 'fpga_mps',
                                                                             'option': json.dumps({
                                                                                 'block_size': self.block_size,
                                                                                 'save_kernel_per_block': self.save_kernel_per_block,
                                                                                 'product_flag': self.product_flag
                                                                             })
                                                                            })
        
        if(response.status_code != 200):
            raise Exception(response)
        
        responseJson = response.json()
        if responseJson['kernel_matrix']:
            mat = np.array(json.loads(responseJson['kernel_matrix']))
         
        return mat
    
    def draw():
        # 回路等の描画用メソッド(必要であればFPGA回路図など)
        return False
    
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)