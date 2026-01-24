import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class MeanFieldModel(nn.Module):
    def __init__(self, N):
        super().__init__()
        # theta ∈ [0, π], varphi ∈ [0, 2π]
        self.theta = nn.Parameter(torch.rand(N) * np.pi)     # θ_p
        self.varphi = nn.Parameter(torch.rand(N) * 2 * np.pi) # φ_p

    def forward(self, J, h):
        theta = self.theta
        varphi = self.varphi

        # ∑_{p<q} sinθ_p sinθ_q cos(φ_p - φ_q)
        sin_theta = torch.sin(theta)
        cos_diff = torch.cos(varphi.unsqueeze(0) - varphi.unsqueeze(1))
        sin_outer = torch.outer(sin_theta, sin_theta)
        interaction = torch.triu(sin_outer * cos_diff, diagonal=1).sum()

        # ∑_p cosθ_p
        field = torch.cos(theta).sum()

        energy = -J * interaction - h * field
        return energy

output_dir = "mean_field_theory/energy_mean_field_theory"
os.makedirs(output_dir, exist_ok=True)

# 参数
N = 8
J = 1.0
hs = np.arange(-2.0, 2.1, 0.1)

# 能量
energies = []

# 实例化模型
model = MeanFieldModel(N)
optimizer = optim.Adam(model.parameters(), lr=0.01)

for h in hs:

    print(h)

    # 优化循环
    for step in range(500):
        optimizer.zero_grad()
        energy = model(J, h)
        energy.backward()
        optimizer.step()
        print(f"Step {step}: E_MF = {energy.item():.6f}")

    energies.append(energy.item())


# save energy
np.savetxt(f"{output_dir}/mean_field_theory_energy_N=8.txt", np.array(energies), fmt="%.6f", header="h energy")



