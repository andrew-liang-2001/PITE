from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np

# %%
delta_tau = 0.1
gamma = 1
phi = 2 * np.arccos(np.exp(-2 * np.abs(gamma) * delta_tau))


# %% IYZY
IYZY = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1))
IYZY.sdg(2)
IYZY.h(2)
IYZY.sdg(4)
IYZY.h(4)
IYZY.cx(2, 3)
IYZY.cx(3, 4)
IYZY.crx(phi, 4, 0)
IYZY.measure(0, 0)
IYZY.cx(3, 4)
IYZY.cx(2, 3)
IYZY.h(4)
IYZY.s(4)
IYZY.h(2)
IYZY.s(2)
print(IYZY.draw())

# %% IXZX
IXZX = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1))
IXZX.h(2)
IXZX.h(4)
IXZX.cx(2, 3)
IXZX.cx(3, 4)
IXZX.crx(phi, 4, 0)
IXZX.measure(0, 0)
IXZX.cx(3, 4)
IXZX.cx(2, 3)
IXZX.h(4)
IXZX.h(2)
print(IXZX.draw())

# %% YZYI
YZYI = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1))
YZYI.sdg(1)
YZYI.h(1)
YZYI.sdg(3)
YZYI.h(3)
YZYI.cx(1, 2)
YZYI.cx(2, 3)
YZYI.crx(phi, 3, 0)
YZYI.measure(0, 0)
YZYI.cx(2, 3)
YZYI.cx(1, 2)
YZYI.h(3)
YZYI.s(3)
YZYI.h(1)
YZYI.s(1)
print(YZYI.draw())

# %% XZXI
XZXI = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1))
XZXI.h(1)
XZXI.h(3)
XZXI.cx(1, 2)
XZXI.cx(2, 3)
XZXI.crx(phi, 3, 0)
XZXI.measure(0, 0)
XZXI.cx(2, 3)
XZXI.cx(1, 2)
XZXI.h(3)
XZXI.h(1)

# %% IIZI
IIZI = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1))
IIZI.crx(phi, 3, 0)
IIZI.measure(0, 0)
print(IIZI.draw())

# %% IIIZ
IIIZ = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1))
IIIZ.crx(phi, 4, 0)
IIIZ.measure(0, 0)
print(IIIZ.draw())

# %% IIZZ
IIZZ = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1))
IIZZ.cx(3, 4)
IIZZ.crx(phi, 4, 0)
IIZZ.measure(0, 0)
IIZZ.cx(3, 4)

# %% ZIII
ZIII = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1))
ZIII.crx(phi, 1, 0)
ZIII.measure(0, 0)

# %% IZII
IZII = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1))
IZII.crx(phi, 2, 0)
IZII.measure(0, 0)

# %% ZZII
ZZII = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1))
ZZII.cx(1, 2)
ZZII.crx(phi, 2, 0)
ZZII.measure(0, 0)
ZZII.cx(1, 2)

# %%
IIII = QuantumCircuit(QuantumRegister(1, "anc"), QuantumRegister(4), ClassicalRegister(1))

# %%
