from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate

"""
def Block_encoding(n, qa, qr1, qr2):
    # Setting up the circuit
    qr1 = QuantumRegister(n, name='Q1')
    qr2 = QuantumRegister(n, name='Q2')
    qc = QuantumCircuit(qr1,qr2, name = 'U_diff')
    return qc
"""

def generate_advection_matrix(size, a):
    matrix = np.zeros((size, size), dtype=float) 

    np.fill_diagonal(matrix[1:], a) 
    np.fill_diagonal(matrix[:, 1:], -a)  

    matrix[0, -1] = a  
    matrix[-1, 0] = -a  

    return matrix

def Block_encoding(n, d, T=1, c=1):
    N = 2**n
    beta_M =  c * T * N / (2 * d)
    print("beta*M : ", beta_M)
    
    dev = qml.device('default.qubit', wires=2*n)
    @qml.qnode(dev)
    def example_circuit(matrix):
        qml.BlockEncode(matrix, wires=range(2*n))
        return qml.state()
    
    A = generate_advection_matrix(N, beta_M) * -1j
    print(A)

    B = qml.matrix(example_circuit)(A)
    #print(B)

    gate = UnitaryGate(B)

    qr1 = QuantumRegister(n, name='Q1')
    qr2 = QuantumRegister(n, name='Q2')
    qc = QuantumCircuit(qr1, qr2, name = 'U_block')
    qc.append(gate, qr1[:] + qr2[:])

    return qc

def is_unitary(matrix, tol=1e-10):
    """
    Checks if a matrix is unitary.

    Parameters:
    - matrix (np.ndarray): The matrix to check.
    - tol (float): Numerical tolerance for equality check.

    Returns:
    - bool: True if the matrix is unitary, False otherwise.
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False  # Must be square

    identity = np.eye(matrix.shape[0])
    conjugate_transpose = np.conjugate(matrix.T)

    return np.allclose(conjugate_transpose @ matrix, identity, atol=tol) and \
           np.allclose(matrix @ conjugate_transpose, identity, atol=tol)
