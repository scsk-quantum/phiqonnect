# PhiQonnect
[![License](https://img.shields.io/github/license/Qiskit/qiskit-experiments.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)

**PhiQonnect** は、量子カーネルによるQSVC、QSVRを実装したライブラリです。

## インストール
```bash
curl curl -OL httpd://github.com/scsk-r-and-d/phiqonnect/phiqonnect-1.0.1-py3-none-any.whl
```
```bash
pip install ./phiqonnect-1.0.1-py3-none-any.whl
```

## 実行例

```python
from phiqonnect.quantum_algorithm.quantum_ai.classification.qsvc import QSVC
from phiqonnect.quantum_algorithm.circuit.qkernel.mps_qkernel import MPS_QKernel
from phiqonnect.utils.dataset_utils.load_classification_datasets import *
from phiqonnect.utils.visualization.classification_result import *

training_input, test_input, label_train, label_test, _ = load_wine(feature_dim=3, train_size=20, test_size=10)

qkernel = MPS_QKernel(feature_dimension=3)
qsvc = QSVC(qkernel, instance="qasm_simulator", shots=400)

train_result = qsvc.train(training_input, label_train, result=True, class_label=class_label)
print(f'train_acc: {train_result["accuracy"]}')

test_result = qsvc.test(test_input, label_test, class_label=class_label)
print(f'test_acc: {test_result["accuracy"]}')

predict = qsvc.predict(test_input, class_label=class_label)
print(f'predict: {predict}')

```
詳細は[tutorial.ipynb](tutorial.ipynb)を参照


## ライセンス
[Apache License 2.0](LICENSE.txt)

## 注意
BraketQuantumInstanceを利用する場合は、Amazon Braketで実行可能なIAMロールなどの必要な設定を行なってください。 \
また、BraketQuantumInstanceでの実機の利用に伴いを利用する場合に発生する利用コストや実行中断・実行エラーに関しては、いかなる責任も負いません。

## 問い合わせ
お問い合わせは下記にお願いいたします。 \
[quantum-tech@scsk.jp](quantum-tech@scsk.jp)