# Import packages
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT, RYGate, RZGate, MCXGate, ModularAdderGate
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

class AdvectionQSVT:
    def __init__(self, n=6, dt=0.05, nu=0.02, shots=10**6, show_gate_count=True, draw=True):
        """
        Initializes the parameters for the Advection simulation using QSVT.

        deg: Degree of the polynomial in QSVT.
        n: Number of qubits in the data registers.
        dt: Time step of the evolution.
        nu: Viscosity coefficient.
        shots: Number of runs in the simulation.
        show_gate_count: If True, displays the quantum gate count.
        draw: If True, draws the quantum circuit.
        """
        self.n = n
        self.dt = dt
        self.nu = nu
        self.shots = shots
        self.show_gate_count = show_gate_count
        self.draw = draw
        self.qc = None  # Se inicializará en build_circuit()

        # Configurar condiciones iniciales
        self.N = 2**self.n
        self.d = 4  # Dominio espacial [0, d]
        self.dx = self.d / self.N
        self.x = np.linspace(0, self.d, self.N, endpoint=False)
        self.y = np.exp(-20 * (self.x - self.d / 3) ** 2)  # Condiciones iniciales gaussianas
        self.y = self.y / np.linalg.norm(self.y)  # Normalizar para que sea un vector unitario
        
    def build_circuit(self):

        # Registros cuánticos
        self.qra = QuantumRegister(2, name="QA")  # Ancilla register en 2 qubits para QSVT
        self.qr1 = QuantumRegister(self.n, name="Q1")  # Registros de datos
        self.qr2 = QuantumRegister(self.n, name="Q2")

        # Registros clásicos
        self.cra = ClassicalRegister(2, name="CA")
        self.cr1 = ClassicalRegister(self.n, name="C1")
        self.cr2 = ClassicalRegister(self.n, name="C2")

        # Crear circuito cuántico
        self.qc = QuantumCircuit(self.qra, self.qr1, self.qr2, self.cra, self.cr1, self.cr2)

        # Preparar el estado inicial en el registro qr2
        self.qc.prepare_state(Statevector(self.y), self.qr2)

    def get_circuit(self):
        
        self.qc.draw("mpl")
        return self.qc
    
    def QSVT(self, U, Phi, Phi2):
        
        self.U = U
        self.Phi = Phi
        self.Phi2 = Phi2
        
        # Applying the QSVT circuit
        self.qc.h(self.qra[1])
        self.qc.barrier()
        self.qc.s(self.qra[0])
        self.qc.h(self.qra[0])
        self.qc.s(self.qra[0]).inverse()
        
        s = 0
        for k in range(len(self.Phi)-1,-1,-1):
            self.qc.barrier()
            if s==0:
                self.qc.append(self.U, self.qr1[:]+self.qr2[:])
                s = 1
            else:
                self.qc.append(self.U.inverse(), self.qr1[:]+self.qr2[:])
                s = 0
            self.qc.mcx(self.qr1[:],self.qra[1],ctrl_state = self.n*'0')
            self.qc.crz(2*self.Phi[k],self.qra[0], self.qra[1], ctrl_state = '0') # Cos angle
            self.qc.crz(2*self.Phi2[k],self.qra[0], self.qra[1], ctrl_state = '1') # Sin angle
            self.qc.mcx(self.qr1[:],self.qra[1],ctrl_state = self.n*'0')

        self.qc.barrier()
        
        self.qc.s(self.qra[1])
        self.qc.h(self.qra[1])

        self.qc.barrier()

        self.qc.s(self.qra[0]).inverse()
        self.qc.h(self.qra[0])
        self.qc.s(self.qra[0])
        self.qc.barrier()
        
        # Measurements
        self.qc.measure(self.qra,self.cra)
        self.qc.measure(self.qr1,self.cr1)
        self.qc.measure(self.qr2,self.cr2)
        
        return self.qc
    
    def run_circuit(self, shots=10**6):
        self.shots = shots
        sim = AerSimulator()
        qc_comp = transpile(self.qc, sim)
        res = sim.run(qc_comp, shots = shots).result()
        self.counts = res.get_counts(0)
        
        # Printing gate counts
        if self.show_gate_count:
            dict = qc_comp.count_ops()
            gate_1q = 0
            gate_2q = 0
            for key in dict:
                if key[0] == 'c':
                    gate_2q += dict[key]
                elif key != 'measure':
                    gate_1q += dict[key]

            print("1 qubit gates:", gate_1q)
            print("2 qubit gates:", gate_2q)
            print("Total:", gate_1q+gate_2q)

            print('Circuit depth after transpiling:', qc_comp.depth())
            
    def post_selection(self):
        # Postselection
        selected_counts = {}
        
        select = (self.n+2)*'0'
        total = 0                      # Tracks the number of successfull outcomes
        z = np.zeros(int(2**self.n))                # The results are encoded in z
        for key in self.counts:
            L = key.split()
            if L[1]+L[2] == select:
                selected_counts[L[0]] = self.counts[key]
                z[int(L[0],2)] = np.sqrt(self.counts[key]/self.shots)    # By construction all amplitudes are positive real numbers
                total += self.counts[key]                           # so this actually recovers them!
        self.success_rate = total/self.shots
        print('Success rate =', self.success_rate)
        self.selected_counts = selected_counts