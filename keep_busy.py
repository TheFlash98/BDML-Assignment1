import torch

# Ensure GPUs are available
if not torch.cuda.is_available():
    raise RuntimeError("No GPUs are available!")

# Get all available GPUs
device_count = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(device_count)]

# Create a large tensor on each GPU
dummy_tensors = [torch.randn(10000, 10000, device=device) for device in devices]

# Infinite loop to keep all GPUs busy
while True:
    for i, device in enumerate(devices):
        dummy_tensors[i] = torch.matmul(dummy_tensors[i], dummy_tensors[i])  # Matrix multiplication