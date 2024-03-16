from tools import *
import matplotlib.pyplot as plt
import qiskit_nature.second_q.hamiltonians as h
from qiskit_nature.second_q.hamiltonians.lattices import LineLattice, BoundaryCondition
from qiskit_nature.second_q.mappers import JordanWignerMapper
import qiskit_algorithms as qalgs


# %% Try FHM with 4 sites just for demonstration
DELTA_TAU = 0.1

line_FHM_lattice = LineLattice(4, boundary_condition=BoundaryCondition.PERIODIC)
fermi_hubbard = h.FermiHubbardModel(line_FHM_lattice.uniform_parameters(uniform_interaction=-0.1,
                                                                        uniform_onsite_potential=0.0),
                                    onsite_interaction=0.1)
fermionic_op = fermi_hubbard.second_q_op()
FHM_op = JordanWignerMapper().map(fermionic_op)
print(FHM_op)
FHM_eigenvalues, FHM_eigenvectors = np.linalg.eigh(FHM_op.to_matrix())
FHM_diagonal = np.diag(FHM_eigenvalues)

# %%
theta = np.pi
FHM_init = QuantumCircuit(QuantumRegister(8))
FHM_init.h(0)
FHM_init.ry(theta, 0)
FHM_init.x(1)
FHM_init.x(7)
for i in range(7):
    FHM_init.cx(i, i+1)

statevector_FHM = get_statevector(FHM_init)

# %%
MAX_TROTTER_STEPS_FHM = 150
solver = qalgs.SciPyImaginaryEvolver(MAX_TROTTER_STEPS_FHM)

trotter_arr_FHM = np.arange(0, MAX_TROTTER_STEPS_FHM, 1)
energy_FHM, probability_FHM = run_experiment(trotter_arr_FHM, statevector_FHM, FHM_op)

# %%
FHM_gs_energy = qalgs.NumPyMinimumEigensolver().compute_minimum_eigenvalue(FHM_op).eigenvalue
print(FHM_gs_energy)
# plt.plot(trotter_arr_FHM, analytical_energy_FHM, label="Exact")
plt.plot(trotter_arr_FHM, energy_FHM, label="PITE")
plt.xlabel("Trotter step $r$")
plt.ylabel("Energy $\langle E \\rangle$")
plt.legend()
# plt.tight_layout()
# plt.savefig("plots/FHM_E_vs_r.pdf", format="pdf", bbox_inches="tight")
plt.show()


# %%
plt.plot(trotter_arr_FHM, probability_FHM, label="PITE")
# plt.plot(trotter_arr_FHM, np.exp(-4 * lamb_FHM * DELTA_TAU * trotter_arr_FHM), label="Exact")
plt.legend()
plt.xlabel("Trotter step $r$")
plt.ylabel("Probability of success $p_s$")
plt.yscale("log")
plt.tight_layout()
# plt.savefig("plots/FHM_p_vs_r.pdf", format="pdf", bbox_inches="tight")
plt.show()
