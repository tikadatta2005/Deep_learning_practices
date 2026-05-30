import torch
import torch.nn as nn

class CnnModel (nn.Module):
    def __init__(self):
        # initialize super 
        super().__init__()
        
        # Kernel
        self.kernel1 = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(8)
        
        self.kernel2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(16)
        
        self.kernel3 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn3 = nn.BatchNorm2d(32)
        
        self.kernel4 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn4 = nn.BatchNorm2d(64)
        
        # Linear neural networks
        self.W1 = nn.Parameter(torch.randn(64*8*8, 8) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(8))
        
        self.W2 = nn.Parameter(torch.randn(8, 2) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(2))

        # loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
    def forward(self, x):
        
        # first kernel
        x = self.kernel1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        
        # second kernel
        x = self.kernel2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        
        # third kernel
        x = self.kernel3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        
        # fourth kernel
        x = self.kernel4(x)
        x = self.bn4(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
                
        # reshape images
        x = x.reshape((x.shape[0], -1))
        
        x = x @ self.W1 + self.b1
        x = nn.functional.relu(x)
        
        x = x @ self.W2 + self.b2
                
        return x
    
    def compute_loss(self, y_hat, y):        
        # loss calling
        loss = self.criterion(y_hat, y)
        return loss
    
    def backward_propagation(self, loss, learning_rate=0.001):
        # start loss backward
        loss.backward()
        
        with torch.no_grad():
            for param in self.parameters():
                if param.grad is not None:
                    param.data -= learning_rate * param.grad
                    param.grad.zero_()
        
        return loss.item()