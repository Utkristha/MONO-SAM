class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        # If input and output channels differ, we need a 1x1 conv to match them
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch)
            )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))

class DepthEncoder(nn.Module):
    def __init__(self, in_channel=2, embeddings_channel=256):
        super().__init__()
        # Using blocks makes it much more robust
        self.layer1 = ResidualBlock(in_channel, 64)
        self.layer2 = ResidualBlock(64, 128)
        self.layer3 = ResidualBlock(128, 256)
        self.final_projection = nn.Conv2d(256, embeddings_channel, kernel_size=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.final_projection(x)