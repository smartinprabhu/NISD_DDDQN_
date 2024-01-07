import torch
import torch.nn as nn
from torchvision.models.detection import backbone_utils
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection import FasterRCNN

# Define the number of classes for instance segmentation
num_classes = 21  # REPLACE_WITH_ACTUAL_VALUE

# Load a pre-trained YOLOv8x model
yolov8x_model = yolov5.yolov5x()  # Replace with actual YOLOv8x model instantiation

# Extract the feature extractor from the YOLOv8x model
class FeatureExtractor(nn.Module):
    def __init__(self, yolov8x_model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(yolov8x_model.children())[:-1])

    def forward(self, x):
        return self.features(x)

# Define the Nano Instance Segmentation model
class NanoInstanceSegmentationModel(nn.Module):
    def __init__(self, feature_extractor, num_classes):
        super(NanoInstanceSegmentationModel, self).__init__()
        self.feature_extractor = feature_extractor
        in_channels = backbone_utils.get_channels(yolov8x_model.features[-1].out_channels)
        self.instance_segmentation_head = NanoInstanceSegmentationHead(in_channels, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.instance_segmentation_head(x)

# Define Nano Instance Segmentation Head
class NanoInstanceSegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(NanoInstanceSegmentationHead, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.upsample(x)
        x = nn.functional.relu(self.conv1(x))
        x = self.conv2(x)
        return x

# Create the Nano Instance Segmentation model
feature_extractor = FeatureExtractor(yolov8x_model)
nano_instance_segmentation_model = NanoInstanceSegmentationModel(feature_extractor, num_classes)

# Print the model architecture for verification
print(nano_instance_segmentation_model)
