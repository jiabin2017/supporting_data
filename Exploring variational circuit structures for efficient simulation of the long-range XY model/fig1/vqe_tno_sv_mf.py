import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import timeit
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# device = 'cpu'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.complex64


I = torch.eye(2, dtype=dtype, device=device)
X = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)

def XX():
    matrix = torch.tensor([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0]
    ], dtype=dtype, device=device)
    return matrix

def YY():
    matrix = torch.tensor([
        [0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0]
    ], dtype=dtype, device=device)
    return matrix

def SWAP():
    matrix = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=dtype, device=device)
    return matrix

def RZ(theta):
    return torch.diag(torch.stack([
        torch.exp(-1j * theta / 2),
        torch.exp(1j * theta / 2)
    ])).to(device=device, dtype=dtype)

def RX(theta):
    return torch.cos(theta / 2) * I - 1j * torch.sin(theta / 2) * X

def RY(theta):
    return torch.cos(theta / 2) * I - 1j * torch.sin(theta / 2) * Y

def RXX(theta):
    return torch.matrix_exp(-1j * theta / 2 * XX())

def RYY(theta):
    return torch.matrix_exp(-1j * theta / 2 * YY())

def kron_n(ops):
    result = ops[0]
    for op in ops[1:]:
        result = torch.kron(result, op)
    return result

def build_hamiltonian(N, J, h):
    H = torch.zeros(2**N, 2**N, dtype=dtype, device=device)
    for i, j in itertools.combinations(range(N), 2):
        XI = [I]*N
        YI = [I]*N
        XI[i] = X
        XI[j] = X
        YI[i] = Y
        YI[j] = Y
        H += -J * (kron_n(XI) + kron_n(YI))
    
    for i in range(N):
        ZI = [I]*N
        ZI[i] = Z
        H += -h * kron_n(ZI)
    return H

class VQEModel(nn.Module):
    def __init__(self, N, num_layer, device='cpu', dtype=torch.complex64):
        super().__init__()
        self.N = N
        self.num_layer = num_layer
        self.device = device
        self.dtype = dtype
        
        # Initialize parameters
        self.theta = nn.Parameter(torch.randn(num_layer, device=self.device))  # RY gates
        self.phi = nn.Parameter(torch.randn(num_layer, device=self.device))   # RZ gates

    def init_state(self):
        # Initialize the state |000...0>
        psi = torch.tensor([1, 0], dtype=self.dtype, device=self.device)
        state = kron_n([psi] * self.N).reshape(-1, 1)
        return state

    def apply_ansatz(self):
        
        state = self.init_state()

        for i in range(self.num_layer):

            # Apply RY rotations
            U_ry = kron_n([RX(self.theta[i]) for j in range(self.N)])
            state = U_ry @ state

            # Apply RZ rotations
            U_rz = kron_n([RZ(self.phi[i]) for j in range(self.N)])
            state = U_rz @ state

        return state

    def forward(self, H):
        final_state = self.apply_ansatz()
        energy = torch.real((final_state.conj().T @ H @ final_state).squeeze())
        return energy


start = timeit.default_timer()

h_vals = torch.arange(-2, 2.1, 0.1)
energies = []
output_dir = "sv/2025-07-08/mean_field/vqe_tno_sv_mf"
os.makedirs(output_dir, exist_ok=True)

N = 8
J = 1.0
num_layer = 1
epochs = 500
lr = 0.01

for h in h_vals:

    # Build Hamiltonian
    H = build_hamiltonian(N, J, h)

    # Initialize model
    model = VQEModel(N=N, num_layer=num_layer, device=device, dtype=dtype)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for step in range(epochs):
        optimizer.zero_grad()
        loss = model(H)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        print(f"Step {step}: Energy = {loss.item():.6f}")

    print(f"h = {h:.2f}: Optimized Energy = {loss.item():.6f}")
    energies.append([h, loss.item()])

    state = model.apply_ansatz()

    # save state
    state_np = state.detach().cpu().numpy().flatten()
    state_real_imag = np.stack([state_np.real, state_np.imag], axis=1)
    np.savetxt(f"{output_dir}/ground_state_h_{h.item():.2f}.txt", state_real_imag, fmt="%.6f", header="Real Imag")

# save energy
np.savetxt(f"{output_dir}/ground_energies.txt", np.array(energies), fmt="%.6f", header="h energy")





stop = timeit.default_timer()
print("Execution Time: ", stop - start)

