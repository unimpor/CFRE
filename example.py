import torch
import torch.nn as nn
import torch.optim as optim


# Define a simple MLP
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


# Define another MLP for post-processing
class SecondMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SecondMLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


def create_hook(lengths):
    def custom_backward_hook(grad):
        normalized_grad = grad / lengths  # Normalize by expansion lengths
        return normalized_grad

    return custom_backward_hook


mlp1 = SimpleMLP(input_dim=3, output_dim=1)
mlp2 = SecondMLP(input_dim=10, output_dim=2)

optimizer1 = optim.SGD(mlp1.parameters(), lr=0.01)
optimizer2 = optim.SGD(mlp2.parameters(), lr=0.01)

input_tensor = torch.randn((5, 3))

output1 = mlp1(input_tensor).squeeze()
output1.register_hook(create_hook(lengths=torch.tensor([1., 2., 3., 4., 1.])))
expanded_output = torch.cat([output1[i].expand(i + 1) for i in range(len(output1) - 1)])
print(expanded_output.dtype)
print(expanded_output)

mult_vector = torch.randn(expanded_output.shape, requires_grad=False)
b = expanded_output * mult_vector

# for leaf-tensor
output1.retain_grad()
expanded_output.retain_grad()

output2 = mlp2(b)

target = torch.tensor([1.0, 1.0])
loss_fn = nn.MSELoss()
loss = loss_fn(output2[0], target[0])

# Backpropagation
optimizer1.zero_grad()
optimizer2.zero_grad()
loss.backward()

optimizer1.step()
optimizer2.step()
# hook_handle.remove()
print(expanded_output.grad)
print(output1.grad)  # output1.grad[2] -> 0.00. See Line 36, output1[2] does not contribute to latter calculation

print((expanded_output.grad[1] + expanded_output.grad[2]) / 2 == output1.grad[1])
print((expanded_output.grad[3] + expanded_output.grad[4] + expanded_output.grad[5]) / 3 == output1.grad[2])
print((expanded_output.grad[6] + expanded_output.grad[7] + expanded_output.grad[8] + expanded_output.grad[9]) / 4 ==
      output1.grad[3])
# Check the gradients of the original MLP's parameters
# print("Gradients of MLP1 parameters (before expansion):")
# for name, param in mlp1.named_parameters():
#     if param.requires_grad:
#         print(f"{name}: {param.grad}")
#
# # Check the gradients of the second MLP as well
# print("\nGradients of MLP2 parameters (after element-wise multiplication):")
# for name, param in mlp2.named_parameters():
#     if param.requires_grad:
#         print(f"{name}: {param.grad}")
