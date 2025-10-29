
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Load pre-trained ResNet-18 model
resnet18 = models.resnet18(pretrained=True)

# import torchinfo
# torchinfo.summary(resnet18, input_size=(1, 3, 224, 224))   # Total params: 11,689,512

# Hazem: THE RESNET18 CAN TAKE INPUT IMAGE OF 256X256 AND NOT NECESSARILY 224X224,
# BUT THE OUTPUT FEATURE MAPS WILL BE OF SIZE 32X32 INSTEAD OF 28X28 --> DEPENDING ON THE INPUT IMAGE SIZE.

"""
Hazem: To extract the output feature maps from stage 2 in a ResNet-18 model, 
you can modify the model's architecture to access intermediate layers directly. 
One way to do this is by creating a custom model that uses the pre-trained ResNet-18 
as a backbone and defines the forward pass to output the intermediate feature maps.
"""

 # Create a custom model class
class ResNetStage2(nn.Module):
    def __init__(self, original_model):
        super(ResNetStage2, self).__init__()
        # Extract layers up to layer2 (stage 2)
        self.features = nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool,
            original_model.layer1,
            original_model.layer2,
            original_model.layer3
        )
    
    def forward(self, x):
        x = self.features(x)
        return x

# Instantiate the custom model
custom_model = ResNetStage2(resnet18)

# Set the model to evaluation mode
custom_model.eval()

# Define the image preprocessing steps
preprocess = transforms.Compose([
    #transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor()
])   


# Load and preprocess the image
image = Image.open("E://VS Projects//test_15-2-2025_BiT_CD//test_1_0_A.png")    # 1, 3, 256, 256
image = preprocess(image)
image = image.unsqueeze(0)  # Add a batch dimension
print ("image.shape:", image.shape)   # torch.Size([1, 3, 224, 224])   OR  torch.Size([1, 3, 256, 256])

# Perform a forward pass to obtain the feature maps from stage 2
with torch.no_grad():
    feature_maps_stage2 = custom_model(image)

print("Feature maps from stage 2:", feature_maps_stage2.shape)   # torch.Size([1, 128, 28, 28])  OR  torch.Size([1, 128, 32, 32])

