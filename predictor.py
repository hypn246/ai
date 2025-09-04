import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms

CLASS_LABEL={
    0: "Anger",
    1: "Contempt",
    2: "Disgust",
    3: "Fear",
    4: "Happy",
    5: "Neutral",
    6: "Sad",
    7: "Surprised"
}
BATCH_SIZE=128
IMG_SIZE=(224,224)


def get_resnet34(num_classes=len(CLASS_LABEL), in_channels=3, pretrained=True):
    # Load pretrained ResNet34
    model = models.resnet34(pretrained=pretrained)

    # Modify the first conv layer to accept 1 channel instead of 3
    if in_channels == 1:
        # Old conv1 expects 3 channels: [64, 3, 7, 7]
        weight = model.conv1.weight.data
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Initialize with mean of pretrained weights
        model.conv1.weight.data = weight.mean(dim=1, keepdim=True)

    # Modify the fully connected layer for your number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def predictor(img):
    data_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    model = get_resnet34(num_classes=len(CLASS_LABEL), in_channels=3, pretrained=False)
    checkpoint = torch.load('./res224colored_transfer_b2_best_20250903_092840.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state'])


    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    img=data_transforms(img)
    img=img.unsqueeze(0) 

    output = model(img)
    pred = torch.argmax(output, dim=1)
    return CLASS_LABEL[pred.item()] 