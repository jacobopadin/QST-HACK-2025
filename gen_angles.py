import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pyqsp
from scipy.linalg import expm
from pyqsp import angle_sequence, response
from pyqsp.poly import (polynomial_generators, PolyTaylorSeries)
from numpy.polynomial.chebyshev import chebfit

output_file = "angles_advection.txt"

def QSVT_format(phiset):
	n = len(phiset)-1
	Phi = np.zeros(n)
	Phi[1:n] = phiset[1:n]-np.pi/2
	Phi[0] = phiset[0]+phiset[-1]+((n-2)%4)*np.pi/2
	return Phi

M = 1
func1 = lambda x: np.cos(M*x)
func2 = lambda x: np.sin(M*x)
polydeg = 5 # Desired QSP protocol length.
max_scale = 0.9 # Maximum norm (<1) for rescaling.
true_func1 = lambda x: max_scale * func1(x) # For error, include scale.
true_func2 = lambda x: max_scale * func2(x)

poly1 = PolyTaylorSeries().taylor_series(
    func=func1,
    degree=polydeg,
    max_scale=max_scale,
    chebyshev_basis=True,
    cheb_samples=2*polydeg)

poly2 = PolyTaylorSeries().taylor_series(
    func=func2,
    degree=polydeg,
    max_scale=max_scale,
    chebyshev_basis=True,
    cheb_samples=2*polydeg)

# Compute full phases (and reduced phases, parity) using symmetric QSP.
(phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
    poly1,
    method='sym_qsp',
    chebyshev_basis=True)

(phiset2, red_phiset2, parity2) = angle_sequence.QuantumSignalProcessingPhases(
    poly2,
    method='sym_qsp',
    chebyshev_basis=True)

pQ = QSVT_format(phiset)
pQ2 = QSVT_format(phiset2)

if (len(pQ) == polydeg):
    pQ = pQ[:-1]
elif(len(pQ2) == polydeg):
    pQ2 = pQ2[:-1]

print(pQ)
print(pQ2)

with open(output_file, "a") as f:
    for item1, item2 in zip(pQ, pQ2):
        f.write(f"{item1:.14f} {item2:.14f} ")  
    f.write("\n") 