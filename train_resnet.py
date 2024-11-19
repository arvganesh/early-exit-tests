import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import matplotlib.pyplot as plt
from torchvision import models

# Load a pre-trained ResNet18 model
resnet = models.resnet34(weights='DEFAULT')
resnet.eval()

# Freeze all layers in the model
for param in resnet.parameters():
    param.requires_grad = False

summary(resnet, input_size=(1, 3, 224, 224))

# Extract features from an intermediate layer (e.g., layer3)
class IntermediateLayerModel(nn.Module):
    def __init__(self, original_model):
        super(IntermediateLayerModel, self).__init__()
        # Keep layers up to layer3 (for ResNet18)
        self.features = nn.Sequential(*list(original_model.children())[:7])
        # Add a new prediction head (e.g., a fully connected layer)
        self.prediction_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(256, 1000)  # Adjusted input size to 256
        )

    def forward(self, x):
        # Forward pass through frozen layers
        x = self.features(x)
        # Forward pass through the new prediction head
        x = self.prediction_head(x)
        return x

# Instantiate the new model with the frozen backbone and prediction head
model = IntermediateLayerModel(resnet)
print("Printing intermediate model summary")
summary(model, input_size=(1, 3, 224, 224))

# Define loss function and optimizer (only for the new prediction head)
criterion = nn.CrossEntropyLoss()  # Assuming a classification task
optimizer = optim.Adam(model.prediction_head.parameters(), lr=0.001)

# Dummy input tensor (batch_size=8, channels=3, height=224, width=224)
input_tensor = torch.randn(8, 3, 224, 224)

# Forward pass through original ResNet for comparison
with torch.no_grad():
    original_resnet_output = resnet(input_tensor)

# # Training loop for fine-tuning the new head
num_epochs = 200
for epoch in range(num_epochs):
    model.train()

    # Forward pass through the modified model
    output = model(input_tensor)

    # Calculate loss between original ResNet output and new head output
    loss = criterion(output, original_resnet_output.argmax(dim=1))

    # Backpropagation and optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# print("Fine-tuning complete!")

# Load a batch of real images for comparison
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop

# Define transformations
preprocess = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
model.eval()
dataset = ImageFolder(root='./images/', transform=preprocess)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

# Get a batch of images
real_images, _ = next(iter(dataloader))

# Forward pass through the original ResNet model and the modified model
with torch.no_grad():
    original_outputs = resnet(real_images)
    modified_outputs = model(real_images)

# Convert outputs to probabilities using softmax
probabilities_original = torch.softmax(original_outputs, dim=1)
probabilities_modified = torch.softmax(modified_outputs, dim=1)

with open("./images/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Display images being classified
def show_images(images, titles=None):
    plt.figure(figsize=(10, 5))
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image.permute(1, 2, 0))
        plt.axis('off')
        if titles:
            plt.title(titles[i])
    plt.show()

# Convert tensor images to displayable format
display_images = real_images * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
display_images = display_images.clamp(0, 1)

# Show images with predictions from the original model
show_images(display_images, titles=None)

# Show top categories per image
for img_prob in probabilities_original:
    print("Predicted classes by the original model:")
    top5_prob, top5_catid = torch.topk(img_prob, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())
    print("")

for img_prob in probabilities_modified:
    print("Predicted classes by the modified model:")
    top5_prob, top5_catid = torch.topk(img_prob, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())
    print("")
