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


mlp1 = SimpleMLP(input_dim=5, output_dim=3)
mlp2 = SecondMLP(input_dim=4, output_dim=2)

optimizer1 = optim.SGD(mlp1.parameters(), lr=0.01)
optimizer2 = optim.SGD(mlp2.parameters(), lr=0.01)

input_tensor = torch.randn(5)

output1 = mlp1(input_tensor)

expanded_output = torch.cat([output1[i].expand(2) for i in range(len(output1)-1)])
mult_vector = torch.ones_like(expanded_output, requires_grad=False)

b = expanded_output * mult_vector

# for leaf-tensor
expanded_output.retain_grad()
output1.retain_grad()

output2 = mlp2(b)
output2.retain_grad()

target = torch.tensor([1.0, 1.0])
loss_fn = nn.MSELoss()
loss = loss_fn(output2[0], target[0])

# Backpropagation
optimizer1.zero_grad()
optimizer2.zero_grad()
loss.backward()

optimizer1.step()
optimizer2.step()

print(output2.grad)  # output[1].grad[0] -> 0.00, as output[1] does not contribute to the loss.
print(expanded_output.grad)
print(output1.grad)  # output1.grad[2] -> 0.00. See Line 36, output1[2] does not contribute to latter calculation

for i in range(2):
    print(expanded_output.grad[2*i] + expanded_output.grad[2*i+1] == output1.grad[i])  # expand <--> grad sum

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
