import torch
import torch.nn as nn
import numpy as np

# Ensure reproducibility
torch.manual_seed(42)

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return x
    
# === 1. Instantiate and eval mode
model = LeNet()
model.eval()

# === 2. Create one input image and save it
input_image = np.random.rand(1, 1, 28, 28).astype(np.float32)
input_tensor = torch.from_numpy(input_image)

input_image.tofile("test_input.bin")
print("✅ Saved test_input.bin")

# === 3. Run inference and save output
with torch.no_grad():
    output_tensor = model(input_tensor)

output_np = output_tensor.numpy()
output_np.astype(np.float32).tofile("expected_output.bin")
print("✅ Saved expected_output.bin")

# === 4. Save weights and biases
params = {k: v.cpu().detach().numpy() for k, v in model.state_dict().items()}
for name, array in params.items():
    array.astype(np.float32).tofile(f"{name}.bin")
    print(f"✅ Saved {name}.bin")

print("✅ Done")