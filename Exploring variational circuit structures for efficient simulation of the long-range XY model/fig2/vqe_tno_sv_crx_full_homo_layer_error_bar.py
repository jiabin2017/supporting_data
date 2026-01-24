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
Hd = (1 / torch.sqrt(torch.tensor(2.0))) * torch.tensor([[1, 1], [1, -1]], dtype=dtype, device=device)

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

def CNOT():
    matrix = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
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

def CRX(theta):
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    matrix = torch.eye(4, dtype=dtype)
    matrix[2, 2] = c
    matrix[2, 3] = -1j * s
    matrix[3, 2] = -1j * s
    matrix[3, 3] = c
    return matrix.to(device)

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
    def __init__(self, N, num_layer, h, device='cpu', dtype=torch.complex64):
        super().__init__()
        self.N = N
        self.num_layer = num_layer
        self.h = h
        self.device = device
        self.dtype = dtype
        
        # Initialize parameters
        amp = 2 * torch.pi
        self.theta = nn.Parameter(amp * torch.rand(num_layer, device=self.device))  # RY gates
        self.phi = nn.Parameter(amp * torch.rand(num_layer, device=self.device))   # RZ gates
        self.alpha = nn.Parameter(amp * torch.rand(num_layer, device=self.device))   # RXX gates

    def init_state(self):
        # Initialize the state |000...0>
        psi = torch.tensor([1, 0], dtype=self.dtype, device=self.device)
        state = kron_n([psi] * self.N).reshape(-1, 1)
        return state

    def apply_ansatz(self):
        state = self.init_state()

        for i in range(self.num_layer):
            # Apply RX rotations
            U_rx = kron_n([RX(self.theta[i]) for j in range(self.N)])
            state = U_rx @ state

            # Apply RZ rotations
            U_rz = kron_n([RZ(self.phi[i]) for j in range(self.N)])
            state = U_rz @ state

            # Add full CRX gates
            for j in range(self.N - 1):
                for k in range(self.N - j - 1):
                    CRXI = [I] * (self.N-1)
                    CRXI[k] = CRX(self.alpha[i])
                    U_crx = kron_n(CRXI)
                    state = U_crx @ state

                    # Apply SWAP
                    SWAPI = [I] * (self.N-1)
                    SWAPI[k] = SWAP()
                    U_swap = kron_n(SWAPI)
                    state = U_swap @ state


            for j in range(self.N - 1):
                for k in range(self.N - j - 1):
                    # Apply SWAP
                    SWAPI = [I] * (self.N-1)
                    SWAPI[k] = SWAP()
                    U_swap = kron_n(SWAPI)
                    state = U_swap @ state

        return state

    def forward(self, H):
        final_state = self.apply_ansatz()
        if abs(self.h) == 1:
            energy = torch.real((final_state.conj().T @ H @ final_state).squeeze())
            for i in range(N):
                ZI = [I]*N
                ZI[i] = Z
                energy -= 0.1 * torch.real((final_state.conj().T @ kron_n(ZI) @ final_state).squeeze())
        else:
            energy = torch.real((final_state.conj().T @ H @ final_state).squeeze())
        return energy

start = timeit.default_timer()

h_vals = torch.arange(-2, 2.1, 0.1)
num_layers = [6]  # Different numbers of layers to try
N = 8
J = 1.0
epochs = 20
num_repeats = 20  # Number of repeats for each configuration

output_dir = "vqe_tno_sv_crx_full_homo_layer_error_bar"
os.makedirs(output_dir, exist_ok=True)

layer_energies = {}
layer_states = {}

for num_layer in num_layers:
    layer_energies[num_layer] = []
    layer_states[num_layer] = []

for num_layer in num_layers:
    print(f"\nRunning for num_layer = {num_layer}")
        
    for h in h_vals:
        # Build Hamiltonian
        H = build_hamiltonian(N, J, h)

        for repeat in range(num_repeats):
            print(f"\nRepeat {repeat+1}/{num_repeats} for h = {h:.2f}, num_layer = {num_layer}")
            
            # Initialize model
            model = VQEModel(N=N, num_layer=num_layer, h=h, device=device, dtype=dtype)
            optimizer = optim.LBFGS(model.parameters(), 
                                    lr=1, 
                                    max_iter=20, 
                                    history_size=10, 
                                    line_search_fn='strong_wolfe',
                                    tolerance_grad=1e-10,
                                    tolerance_change=1e-12
                                    )

            best_energy = float('inf')
            best_state = None
            best_params = None

            for step in range(epochs):
                def closure():
                    optimizer.zero_grad()
                    loss = model(H)
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)

                if loss.item() < best_energy:
                    best_energy = loss.item()
                    best_state = model.apply_ansatz()
                    best_params = {name: param.detach().clone() for name, param in model.named_parameters()}

                print(f"Step {step}: Energy = {loss.item():.6f}")
            
            # Optional post-processing if |h| == 1
            if abs(h) == 1:
                state_detach = best_state.detach()
                for i in range(N):
                    ZI = [I]*N
                    ZI[i] = Z
                    best_energy += 0.1 * torch.real((state_detach.conj().T @ kron_n(ZI) @ state_detach).squeeze()).item()
                print(f"h = {h:.2f}, num_layer = {num_layer}, repeat = {repeat+1}: Best Energy = {best_energy:.6f}")
                layer_energies[num_layer].append([h, repeat, best_energy])
            else:
                print(f"h = {h:.2f}, num_layer = {num_layer}, repeat = {repeat+1}: Best Energy = {best_energy:.6f}")
                layer_energies[num_layer].append([h, repeat, best_energy])

            # Save this state
            state_np = best_state.detach().cpu().numpy().flatten()
            state_real_imag = np.stack([state_np.real, state_np.imag], axis=1)
            layer_states[num_layer].append([h, repeat, state_real_imag])

    # Save all results for this num_layer
    layer_output_dir = f"{output_dir}/num_layer_{num_layer}"
    os.makedirs(layer_output_dir, exist_ok=True)

    # Save energies: h, repeat, energy
    np.savetxt(f"{layer_output_dir}/ground_energies.txt", 
               np.array(layer_energies[num_layer]), 
               fmt="%.6f %.0f %.12f", 
               header="h repeat energy")

    # Save each state to its own file
    for h_val, repeat_idx, state in layer_states[num_layer]:
        state_filename = f"{layer_output_dir}/ground_state_h_{h_val:.2f}_rep_{int(repeat_idx)}.txt"
        np.savetxt(state_filename, state, fmt="%.6e", header="Re Im")


stop = timeit.default_timer()
print("Execution Time: ", stop - start)

