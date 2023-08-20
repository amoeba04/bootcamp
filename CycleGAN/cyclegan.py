import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, n_residual_blocks=9):
        """
        Generator 클래스
        Args:
        - input_nc: 입력 이미지 채널 수
        - n_residual_blocks: 중간에 사용될 ResidualBlock의 개수
        """
        super(Generator, self).__init__()

        # 초기 콘볼루션 블록(Convolution Block)
        out_channels = 64
        model = [nn.ReflectionPad2d(input_nc)]
        model.append(nn.Conv2d(input_nc, out_channels, kernel_size=7))
        model.append(nn.InstanceNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        in_channels = out_channels
        
        # 다운샘플링(Downsampling)
        for _ in range(2):
            out_channels *= 2
            model.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            model.append(nn.InstanceNorm2d(out_channels))
            model.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        # Residual Blocks
        for _ in range(n_residual_blocks):
            model.append(ResidualBlock(out_channels))

        # 업샘플링(Upsampling)
        for _ in range(2):
            out_channels //= 2
            model.append(nn.Upsample(scale_factor=2))
            model.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            model.append(nn.InstanceNorm2d(out_channels))
            model.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        # 출력 콘볼루션 블록(Convolution Block)
        model.append(nn.ReflectionPad2d(input_nc))
        model.append(nn.Conv2d(out_channels, input_nc, kernel_size=7))
        model.append(nn.Tanh())

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        """
        Discriminator 클래스
        Args:
        - input_nc: 입력 이미지 채널 수
        """
        super(Discriminator, self).__init__()

        # 콘볼루션 블록(Convolution Block)
        def discriminator_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_nc, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, x):
        return self.model(x)
