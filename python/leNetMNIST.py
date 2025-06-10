import torch
import torch.nn as nn
import numpy as np

from torchvision import datasets, transforms

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

# === 2. Load one MNIST image (e.g., digit at index 0)
transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

N = len(mnist)
images = np.zeros((N, 1, 28, 28), dtype=np.float32)
labels = np.zeros((N,), dtype=np.int32)

for i in range(N):
    img, label = mnist[i]
    images[i] = img.numpy()
    labels[i] = label

images.tofile("mnist_images.bin")
labels.tofile("mnist_labels.bin")

print("✅ Exported mnist_images.bin and mnist_labels.bin")

# === 3. Run inference and save output
with torch.no_grad():
    output_tensor = model(torch.from_numpy(images))

output = output_tensor.detach().numpy()
output.tofile("mnist_output.bin")
print("✅ Exported mnist_output.bin")

# === 4. Save weights and biases
params = {k: v.cpu().detach().numpy() for k, v in model.state_dict().items()}
for name, array in params.items():
    array.astype(np.float32).tofile(f"{name}.bin")
    print(f"✅ Saved {name}.bin")

print("✅ Done")