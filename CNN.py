import torch
import torch.nn as nn
import torch.nn.functional as F

class BirdCNN(nn.Module):
    """
    CNN model for bird detection in image patches.
    
    Architecture:
    - 3 convolutional blocks with increasing channels (32 -> 64 -> 128)
    - Each block consists of Conv2D, BatchNorm, SiLU activation, and MaxPool
    - Classification head with dropout and 2 fully connected layers
    - Output is a single binary probability indicating if the patch contains a bird
    """
    def __init__(self, input_channels=3, input_size=40):
        """
        Initialize the CNN model with configurable input parameters.
        
        Args:
            input_channels (int): Number of input channels, 3 for RGB images
            input_size (int): Input image size (assumes square input)
        """
        super(BirdCNN, self).__init__()
        
        # Validate input dimensions
        if input_size % 8 != 0:
            raise ValueError(f"Input size must be divisible by 8, got {input_size}")
        
        # Calculate the final feature map size after 3 max pooling operations (each reducing by 1/2)
        final_size = input_size // 8
        
        # Convolutional layers
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),  # (B, 3, 40, 40) → (B, 32, 40, 40)
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),                             # (B, 32, 20, 20)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (B, 64, 20, 20)
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),                             # (B, 64, 10, 10)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# (B, 128, 10, 10)
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(2)                              # (B, 128, 5, 5)
        )
        
        # Fully connected classifier layers
        self.classifier = nn.Sequential(
            nn.Flatten(),                                # (B, 128*final_size*final_size)
            nn.Dropout(0.1),
            nn.Linear(128 * final_size * final_size, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),                           # Output single logit
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W)
            
        Returns:
            torch.Tensor: Probability of each input patch containing a bird, shape (B,)
        """
        # Input validation
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B,C,H,W), got {x.dim()}D")
            
        x = self.conv_block(x)
        x = self.classifier(x)
        return torch.sigmoid(x).squeeze(1)  # Apply sigmoid and remove extra dimension
        
    def get_input_size(self):
        """Return the expected input size for the model."""
        return 40