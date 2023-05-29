# QSVRのテスト

from phiqonnect.quantum_algorithm.quantum_ai.regression.qsvr import QSVR
from phiqonnect.quantum_algorithm.circuit.qkernel.mps_qkernel import MPS_QKernel
from phiqonnect.utils.dataset_utils.load_regression_datasets import *
from phiqonnect.utils.visualization.regression_result import *


train_data, test_data, train_target, test_target, data, target, target_scaler = load_diabetes(feature_dim=4, train_size=20, test_size=10)

qkernel = MPS_QKernel(feature_dimension=4)
qsvr = QSVR(qkernel, instance="qasm_simulator", shots=400)

train_result = qsvr.train(train_data, train_target, True)
print(f'train_r2: {train_result["r2"]}')
print(f'train_mae: {train_result["mae"]}')
print(f'train_rmse: {train_result["rmse"]}')

test_result = qsvr.test(test_data, test_target, use_support_vector_index=False)

print(f'test_r2: {test_result["r2"]}')
print(f'test_mae: {test_result["mae"]}')
print(f'test_rmse: {test_result["rmse"]}')

predict = qsvr.predict(test_data, use_support_vector_index=False)
print(f'predict: {predict}')