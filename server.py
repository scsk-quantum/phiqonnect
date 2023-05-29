api_key_list = [
    "0831hf21f89nfpkljlwoo4e2908u",
]

from phiqonnect.quantum_algorithm.circuit.qkernel.mps_qkernel import MPS_QKernel
# from phiqonnect.quantum_algorithm.circuit.qkernel.fpga.fpga_mps_kernel import FPGA_MPS_Kernel

from flask import Flask, jsonify, request, make_response, send_file
from flask_cors import CORS
import logging
from io import BytesIO
import numpy as np
import json
import string
import random
import time
from PIL import Image
from io import BytesIO
from queue import Queue
import functools
import hashlib

from qiskit.providers.aer import Aer
from qiskit.utils import QuantumInstance

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

multipleQueue = Queue(maxsize=1)
singleQueue = Queue(maxsize=1)

def multiple_control(q):
    def _multiple_control(func):
        @functools.wraps(func)
        def wrapper(*args,**kwargs):
            q.put(time.time())
            result = func(*args,**kwargs)
            q.get()
            q.task_done()
            return result

        return wrapper
    return _multiple_control

app = Flask(__name__)

@app.route('/health', methods=["GET", "POST"])
def health():
    ret = jsonify({'code': '200', 'message': '200 OK'})
    response = make_response(ret)
    response.headers['Content-Type'] = 'application/json'
    response.status_code = 200
    
    return response

@app.route('/get_kernel_matrix', methods=["GET", "POST"])
def get_kernel_matrix():
    if request.method == 'GET':
        dim = request.args.get('dim', '')
        option = request.args.get('option', '')
        kernel = request.args.get('kernel', '')
        x1_vec = request.args.get('x1_vec', '')
        x2_vec = request.args.get('x2_vec', '')
    elif request.method == 'POST':
        dim = request.form["dim"]
        option = request.form["option"]
        kernel = request.form["kernel"]
        x1_vec = request.form["x1_vec"]
        x2_vec = request.form["x2_vec"]
        
    if dim and kernel and x1_vec and x2_vec:
        if kernel == 'mps':
            backend = Aer.get_backend('qasm_simulator')
            instance = QuantumInstance(backend, shots=500, seed_simulator=1234, seed_transpiler=1234)
            qkernel = MPS_QKernel(feature_dimension=int(dim))
            kernel_matrix = qkernel.get_kernel_matrix(np.array(json.loads(x1_vec)).astype('float'), np.array(json.loads(x2_vec)).astype('float'), instance)
        if kernel == 'fpga_mps':
            params = json.loads(option)
            print(params)
            backend = Aer.get_backend('qasm_simulator')
            instance = QuantumInstance(backend, shots=500, seed_simulator=1234, seed_transpiler=1234)
            qkernel = FPGA_MPS_QKernel(feature_dimension=int(dim), block_size=params["block_size"], save_kernel_per_block=params["save_kernel_per_block"], product_flag=params["product_flag"])
            kernel_matrix = qkernel.get_kernel_matrix(np.array(json.loads(x1_vec)).astype('float'), np.array(json.loads(x2_vec)).astype('float'), instance)    
            
        ret = jsonify({'code': '200', 'message': '200 OK', 'kernel_matrix': json.dumps(kernel_matrix,cls = MyEncoder)})
        response = make_response(ret)
        response.headers['Content-Type'] = 'application/json'
        response.status_code = 200
    else:
        ret = jsonify({'code': '400', 'message': '400 Bad Request'})
        response = make_response(ret)
        response.headers['Content-Type'] = 'application/json'
        response.status_code = 400
        
    return response
        
@app.before_request
def before_request():
    if request.method == 'GET':
        api_key = request.args.get('api_key', '')
    elif request.method == 'POST':
        api_key = request.form["api_key"]
        
    if api_key not in api_key_list:
        ret = jsonify({'code': '401', 'message': '401 Unauthorized'})
        response = make_response(ret)
        response.headers['Content-Type'] = 'application/json'
        response.status_code = 403
        
        return response

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == "__main__":
    app.run(threaded=True)