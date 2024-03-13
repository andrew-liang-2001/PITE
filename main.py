from tools import *
import matplotlib.pyplot as plt
import qiskit_nature.second_q.hamiltonians as h
from qiskit_nature.second_q.hamiltonians.lattices import LineLattice, BoundaryCondition
from qiskit_nature.second_q.mappers import JordanWignerMapper
from scipy.linalg import expm
plt.style.use('science')
# %% 4 site 1D Ising model

line_TIM_lattice = LineLattice(4, boundary_condition=BoundaryCondition.PERIODIC)

ising_model = h.IsingModel(line_TIM_lattice.uniform_parameters(0.5, 0.1))
TIM_op = SparsePauliOp(data=["ZZII", "IZZI", "IIZZ", "XIII", "IXII", "IIXI", "IIIX", "ZIIZ"],
                       coeffs=[0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.5])

# print(TIM_op.to_matrix())
TIM_eigenvalues, TIM_eigenvectors = np.linalg.eig(TIM_op.to_matrix())
TIM_diagonal = np.diag(TIM_eigenvalues)
# P_inv = np.linalg.inv(TIM_eigenvectors)
# print(TIM_eigenvectors @ TIM_diagonal @ P_inv)

# %% 2 site 1D Hubbard model
DELTA_TAU = 0.1

line_FHM_lattice = LineLattice(2, boundary_condition=BoundaryCondition.PERIODIC)
fermi_hubbard = h.FermiHubbardModel(line_FHM_lattice.uniform_parameters(uniform_interaction=-0.1,
                                                                        uniform_onsite_potential=0.0),
                                    onsite_interaction=0.1)
fermionic_op = fermi_hubbard.second_q_op()
FHM_op = JordanWignerMapper().map(fermionic_op)
FHM_eigenvalues, FHM_eigenvectors = np.linalg.eigh(FHM_op.to_matrix())
FHM_diagonal = np.diag(FHM_eigenvalues)

# print(f"Number of Pauli strings: {len(FHM_op)}")

# %% initial state preparation
theta = np.pi  # produces singlet state 1/sqrt(2) |0110> - |1001>
FHM_init = QuantumCircuit(QuantumRegister(4))
FHM_init.h(0)
FHM_init.ry(theta, 0)
FHM_init.x(1)
FHM_init.x(3)
FHM_init.cx(0, 1)
FHM_init.cx(1, 2)
FHM_init.cx(2, 3)
statevector_FHM = get_statevector(FHM_init)

#  initial state preparation for 4 site 1D Ising model
TFIM_init = QuantumCircuit(QuantumRegister(4))
TFIM_init.h([0, 1, 2, 3])
statevector_TIM = get_statevector(TFIM_init)

# %%
trotter_arr_TIM = np.arange(0, 40, 1)
energy_TIM, probability_TIM = run_experiment(trotter_arr_TIM, statevector_TIM, TIM_op)
analytical_energy_TIM = []
lamb_TIM = np.linalg.norm(TIM_op.coeffs, 1)
analytical_probability_TIM = np.exp(-4 * lamb_TIM * DELTA_TAU * trotter_arr_TIM)
for r in trotter_arr_TIM:
    ITE_TIM = Operator(np.exp(-r * DELTA_TAU * TIM_op))
    evolved_statevector_TIM = statevector_TIM.evolve(ITE_TIM)
    analytical_energy_TIM.append(expectation_value(TIM_op, evolved_statevector_TIM))

# %%
trotter_arr_FHM = np.arange(0, 60, 1)
energy_FHM, probability_FHM = run_experiment(trotter_arr_FHM, statevector_FHM, FHM_op)
analytical_energy_FHM = []
lamb_FHM = np.linalg.norm(FHM_op.coeffs, 1)
analytical_probability_FHM = np.exp(-4 * lamb_FHM * DELTA_TAU * trotter_arr_FHM)
for r in trotter_arr_FHM:
    ITE_FHM = Operator(np.exp(-r * DELTA_TAU * FHM_op))
    evolved_statevector_FHM = statevector_FHM.evolve(ITE_FHM)
    analytical_energy_FHM.append(expectation_value(FHM_op, evolved_statevector_FHM))

# %%
# lamb = np.linalg.norm(qubit_jw_op.coeffs, 1)

# %% Plot energy vs Trotter steps TIM
plt.plot(trotter_arr_TIM, energy_TIM, label="PITE")
plt.plot(trotter_arr_TIM, analytical_energy_TIM, label="Analytical")
plt.xlabel("Trotter step $r$")
plt.ylabel("Energy $\langle E \\rangle$")
plt.show()
# plt.savefig("plots/TFIM_E_vs_r.pdf", format="pdf", bbox_inches="tight")

# %% Plot energy difference to ground state vs Trotter steps TIM
TIM_gs_energy = min(TIM_eigenvalues)
plt.plot(trotter_arr_TIM, energy_TIM - min(TIM_eigenvalues), label="PITE")
plt.yscale("log")
plt.xlabel("Trotter step $r$")
plt.ylabel(r"Energy difference to ground state $\langle E \rangle - E_0$")
plt.show()

# %% Plot probability vs Trotter steps TIM
plt.plot(trotter_arr_TIM, probability_TIM, label="PITE")
plt.plot(trotter_arr_TIM, np.exp(-4 * lamb_TIM * DELTA_TAU * trotter_arr_TIM), label="Analytical")
plt.xlabel("Trotter step $r$")

plt.ylabel("Probability of success $p_s$")
plt.yscale("log")
plt.show()

# %% Plot energy vs Trotter steps FHM
plt.plot(trotter_arr_FHM, energy_FHM, label="PITE")
plt.plot(trotter_arr_FHM, analytical_energy_FHM, label="Analytical")
print(min(energy_FHM))
plt.xlabel("Trotter step $r$")
plt.ylabel("Energy $\langle E \\rangle$")
plt.show()

# %% Plot energy difference to ground state vs Trotter steps FHM
FHM_gs_energy = min(FHM_eigenvalues)
plt.plot(trotter_arr_FHM, energy_FHM - min(FHM_eigenvalues), label="PITE")
plt.yscale("log")
plt.xlabel("Trotter step $r$")
plt.ylabel(r"Energy difference to ground state $\langle E \rangle - E_0$")
plt.show()

# %% Plot probability vs Trotter steps FHM
plt.plot(trotter_arr_FHM, probability_FHM, label="PITE")
plt.plot(trotter_arr_FHM, np.exp(-4 * lamb_FHM * DELTA_TAU * trotter_arr_FHM), label="Analytical")
plt.legend()
plt.xlabel("Trotter step $r$")
plt.ylabel("Probability of success $p_s$")
plt.yscale("log")
plt.show()
