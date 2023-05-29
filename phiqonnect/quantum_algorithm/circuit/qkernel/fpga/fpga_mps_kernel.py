#コンストラクタでfeature_dimension, block_num(ブロック分割数)
#qkernel = FPGA_MPS_Kernel(feature_dimension=2, block_size=2)
# phiqonnect2/quantum_algorithm/circuit//qkernel/fpga_mps_kernel.pyで保存を想定

import pynq
import numpy as np
import gc
#import time
#from phiqonnect2.qalgorithm.qai.classification.qsvm.QSVM import QSVM
#ol = pynq.Overlay('../6qbit_awsxclbin/binary_container_1.awsxclbin')

class FPGA_MPS_Kernel():
    """
    FPGA_MPS_Kernelクラス

    Parameters
	----------
    feature_dimension: int
        特徴量の次元
    block_size: int
        ブロックサイズ
    save_kernel_per_block: boolean
        ブロックごとにカーネル行列を保存するか
    product_flag: boolean
        カーネルの積を取るかどうか
    """
    def __init__(self, feature_dimension=2, block_size=2,save_kernel_per_block = False,product_flag=True) -> None:
        #コンストラクタ
        #ここで必要なパラメータ等の設定を行ってください
        self.feature_dimension = feature_dimension
        self.block_size = block_size
        self.kernel_list=[]
        self.save_kernel_per_block = save_kernel_per_block
        self.product_flag = product_flag
        self.in1 = None
        self.out = None
        if self.block_size == 2:
            self.ol = pynq.Overlay('./2qubit_circuit.awsxclbin')
        elif self.block_size == 3: 
            self.ol = pynq.Overlay('./3qubit_circuit.awsxclbin')
        elif self.block_size == 6:
            self.ol = pynq.Overlay('./6qubit_circuit.awsxclbin')
    
    def get_kernel_matrix(self, x1_vec, x2_vec,instanse=None):
        #必須メソッド
        #ここでカーネル行列を返してください
        if x2_vec is None:
            x2_vec = x1_vec
          
        if len(x2_vec) > 2000 and len(x1_vec) > 2000:
            self.in1 = pynq.allocate((4000,self.block_size),'i4') 
            self.out = pynq.allocate((2000,2000),'u4')
            mat1 = self.fpga_calc_by_block(x1_vec[0:2000], x2_vec[0:2000], self.feature_dimension, self.block_size)
            
            self.in1 = pynq.allocate((2000+len(x2_vec[2000:]),self.block_size),'i4') 
            self.out = pynq.allocate((len(x2_vec[2000:]),2000),'u4')
            mat2 = self.fpga_calc_by_block(x1_vec[0:2000], x2_vec[2000:], self.feature_dimension, self.block_size)
            matA = np.hstack((mat1,mat2))
                                     
            self.in1 = pynq.allocate((len(x1_vec[2000:])+len(x2_vec[2000:]),self.block_size),'i4') 
            self.out = pynq.allocate((len(x2_vec[2000:]),len(x1_vec[2000:])),'u4')                         
            mat3 = self.fpga_calc_by_block(x1_vec[2000:], x2_vec[2000:], self.feature_dimension, self.block_size)
            
            matB = np.hstack((mat2.T,mat3))
            mat = np.vstack((matA,matB))
            
        elif len(x2_vec) > 2000:
            self.in1 = pynq.allocate((len(x1_vec)+2000,self.block_size),'i4') 
            self.out = pynq.allocate((2000,len(x1_vec)),'u4')
            mat1 = self.fpga_calc_by_block(x1_vec, x2_vec[0:2000], self.feature_dimension, self.block_size)
            
            self.in1 = pynq.allocate((len(x1_vec)+len(x2_vec[2000:]),self.block_size),'i4') 
            self.out = pynq.allocate((len(x2_vec[2000:]),len(x1_vec)),'u4')            
            mat2 = self.fpga_calc_by_block(x1_vec, x2_vec[2000:], self.feature_dimension, self.block_size)
            mat = np.hstack((mat1,mat2))
            
        else:
            
            self.in1 = pynq.allocate((len(x1_vec)+len(x2_vec),self.block_size),'i4') 
            self.out = pynq.allocate((len(x2_vec),len(x1_vec)),'u4')   
            mat = self.fpga_calc_by_block(x1_vec, x2_vec, self.feature_dimension, self.block_size)
         
        return mat
        
        #return self.fpga_calc_by_block(x1_vec, x2_vec, self.feature_dimension, self.block_size)
    
    def fpga_divide_for_block(self,x1, x2, feature_dim, block_size):
        _x1 = np.zeros((len(x1), feature_dim // block_size, block_size))
        _x2 = np.zeros((len(x2), feature_dim // block_size, block_size))

        for i in range(feature_dim // block_size):
            idx = range(i * block_size, (i+1) * block_size)
            _x1[:, i, :] = x1[:, idx]
            _x2[:, i, :] = x2[:, idx]
        
        return np.array(_x1), np.array(_x2)
    
    def fpga_calc_by_block(self,x1, x2, feature_dim, block_size):
        if x2 is None:
            x2 = x1

        _x1, _x2 = self.fpga_divide_for_block(x1, x2, feature_dim, block_size)
        
        kernel_matrix = np.ones((len(x1), len(x2)))
        for i in range(feature_dim // block_size):
            idx = range(i * block_size, (i+1) * block_size)
            if self.product_flag:
                kernel_matrix *= self._get_kernel_matrix(_x1[:, i], _x2[:, i])
            else:
                kernel_matrix += self._get_kernel_matrix(_x1[:, i], _x2[:, i])

        return kernel_matrix
        
    def _get_kernel_matrix(self,x1_vec, x2_vec):
        if x2_vec is None:
            x2_vec = x1_vec  
        AA = np.floor(x1_vec*(2**12))
        AA2 = np.floor(x2_vec*(2**12))
        AA = AA.astype('int32')
        AA2 = AA2.astype('int32')
        #print(AA2)
        datasize1=AA.shape[0]
        datasize2=AA2.shape[0]
        #print(datasize1)
        #print(datasize2)
        AA = np.concatenate([AA, AA2])
        kernel = self.ol.rtl_kernel_Qmain_1
        
        #in1 = pynq.allocate((datasize1+datasize2,2),'i4')#20,2 'u4'
            
        #out = pynq.allocate((datasize2,datasize1),'u4')#50,2 'u4'

        dataLenXtemp=datasize1+datasize2
        Rsize = (datasize1+datasize2)*self.block_size#16#40
        Wsize = datasize1*datasize2#16#100


        datas1 = datasize1
        datas2 = datasize2

        self.in1[:] = AA
        self.out[:] = 0 #for DDR mem init

        ReadSize = Rsize*4
        WriteSize = Wsize*4
        #print("input",in1)
        #start = time.time()
        self.out.sync_to_device()
        self.in1.sync_to_device()
        #print(datas1)
        #print(datas2)
        kernel.call(ReadSize,WriteSize,dataLenXtemp,datas1,datas2,self.in1,self.out)

        self.in1.sync_from_device()
        self.out.sync_from_device()
        #end = time.time()
        #print(end - start)
        #fpga_only_time = end-start
        mat = self.out/(2**28)
        mat = mat.T
        
        self.in1.freebuffer()
        self.out.freebuffer()
        #del self.in1
        #del self.out
        #gc.collect()
        
        if self.save_kernel_per_block:
            self.kernel_list.append(mat)
        
        return mat
    
    def draw():
        # 回路等の描画用メソッド(必要であればFPGA回路図など)
        return False