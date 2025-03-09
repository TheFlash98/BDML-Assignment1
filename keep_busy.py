import torch

# Ensure the GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a large tensor
dummy_tensor = torch.randn(10000, 10000, device=device)

# Infinite loop to keep the GPU busy
while True:
    dummy_tensor = torch.matmul(dummy_tensor, dummy_tensor)  # Matrix multiplication

