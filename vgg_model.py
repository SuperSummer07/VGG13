from torch import nn

conv_net = nn.Sequential(
    #Unit 1
    nn.Conv2d(3, 64, kernel_size=3, padding="same"),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 64, kernel_size=3, padding="same"),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),

    #Unit2
    nn.Conv2d(64, 128, kernel_size=3, padding="same"),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 128, kernel_size=3, padding="same"),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),

    #Unit 3
    nn.Conv2d(128, 256, kernel_size=3, padding="same"),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, kernel_size=3, padding="same"),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),	    
        
    #Unit 4
	nn.Conv2d(256, 512, kernel_size=3, padding="same"),
	nn.BatchNorm2d(512),
	nn.ReLU(inplace=True),
	nn.Conv2d(512, 512, kernel_size=3, padding="same"),
	nn.ReLU(inplace=True),
	nn.MaxPool2d(kernel_size=2, stride=2),

	#Unit 5
	nn.Conv2d(512, 512, kernel_size=3, padding="same"),
	nn.BatchNorm2d(512),
	nn.ReLU(inplace=True),
	nn.Conv2d(512, 512, kernel_size=3, padding="same"),
	nn.ReLU(inplace=True),
	nn.MaxPool2d(kernel_size=2, stride=2)
)


fc_net = nn.Sequential(
	nn.Linear(512, 256),
	nn.ReLU(inplace=True),
	nn.Dropout(p=0.5),
	nn.Linear(256, 128),
	nn.ReLU(inplace=True),
	nn.Linear(128, 10)
)


class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()
        self.feature = conv_net
        self.fc = fc_net

    def forward(self, x):
        x = self.feature(x)
        x = x.reshape([x.shape[0], 512])
        x = self.fc(x)
        return x
