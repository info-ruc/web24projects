import torch.nn as nn
import torch


class FCN8s(nn.Module):
    def __init__(self, n_class):
        super(FCN8s, self).__init__()
        self.n_class = n_class
        # Existing layers
        self.conv6 = nn.Conv2d(2048, 1024, kernel_size=1)
        self.conv7 = nn.Conv2d(1024, 512, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        # Decoder layers
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Final classifier
        self.classifier = nn.Conv2d(128, n_class, kernel_size=1)

        # Additional layers for skip connections (assuming the channel sizes)
        self.skip_conv1 = nn.Conv2d(256, 128, kernel_size=1,stride=2)
        self.skip_conv2 = nn.Conv2d(512, 256, kernel_size=1)
        self.skip_conv3 = nn.Conv2d(1024, 512, kernel_size=1)

    def forward(self, x1):
        # Assuming x is a list of your feature maps
        f1, f2, f3, f4 = x1

        # Your existing forward code
        x = self.conv6(f4)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.relu(x)

        x = self.relu(self.deconv1(x))
        x = x + self.skip_conv3(f3)  # Skip connection
        x = self.bn1(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = x + self.skip_conv2(f2)  # Skip connection
        x = self.relu(x)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = x + self.skip_conv1(f1)  # Skip connection
        x = self.relu(x)

        x = self.classifier(x)

        return x
