import torch, torch.nn as nn

class FoodDetectionModel(nn.Module):
    def __init__(self, in_channels:int, hidden_units:int):

        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.ReLU()
        )

        self.MLP = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=3072,
                      out_features=1280),
            nn.ReLU(),
            nn.Linear(in_features=1280, out_features=3),
        )

        self.model = nn.Sequential( self.conv_block,
                                    self.conv_block2,
                                    self.MLP)

    def forward(self, x):

        return self.model(x)