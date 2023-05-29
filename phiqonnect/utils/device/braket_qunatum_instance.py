"braket_qunatum_instance.py"
from braket.devices import LocalSimulator
from braket.aws import AwsDevice

class BraketQuantumInstance():

    """
    phiqonnectからAmazon Braketを使用するためのクラス
    (デバイスを使用するにはAWS SDK for Python3(Boto3)の設定が必要です)

    Parameters
    ----------
    qkernel: QKernel
        QKernelを継承したクラスのオブジェクト
    device: AwsDevice|string
        AwsDeviceあるいは使用するバックエンド("simulator", "ionq", "rigetti")
    backet: string
        backet
    prefix: string
        prefix

    """
    def __init__(self, device="simulator", shots=1000, backet=None, prefix="Task"):
        """Initialize the BraketQuantumInstance object.

        Args:
            device: The device to run on.
            shots: The number of shots to run.
            backet: s3 backet.
            prefix: s3 backet prefix.
        """
        if backet and prefix:
            s3_folder = (backet, prefix)

        if device == "simulator":
            self.device = LocalSimulator("default")
        elif device == "ionq":
            self.device = AwsDevice("arn:aws:braket:::device/qpu/ionq/ionQdevice")
        elif device == "rigetti":
            self.device = AwsDevice("arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-2")
        else:
            self.device = device
        self.shots = shots
