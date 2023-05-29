# QSVCのテスト

from phiqonnect.quantum_algorithm.quantum_ai.classification.qsvc import QSVC
from phiqonnect.quantum_algorithm.circuit.qkernel.mps_qkernel import MPS_QKernel
from phiqonnect.utils.dataset_utils.load_classification_datasets import *
from phiqonnect.utils.visualization.classification_result import *

training_input, test_input, label_train, label_test, class_label, *_ = load_wine(feature_dim=4, train_size=20, test_size=10)

qkernel = MPS_QKernel(feature_dimension=4)
qsvc = QSVC(qkernel, instance="qasm_simulator", shots=400)

train_result = qsvc.train(training_input, label_train, result=True, class_label=class_label)
print(f'train_acc: {train_result["accuracy"]}')

test_result = qsvc.test(test_input, label_test, class_label=class_label)
print(f'test_acc: {test_result["accuracy"]}')

predict = qsvc.predict(test_input, class_label=class_label)
print(f'predict: {predict}')