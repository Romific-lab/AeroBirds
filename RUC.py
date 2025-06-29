import torch
from torchcam.methods import GradCAM
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Assuming you have a trained model loaded
model.eval()

# Hook GradCAM to last conv layer (your third Conv2d in conv_block)
cam_extractor = GradCAM(model, target_layer='conv_block.8')  # index of 3rd Conv2d

# Get a single input image patch
# Assume `input_tensor` is shape (1, 3, 40, 40)
with torch.no_grad():
    out = model(input_tensor)  # forward pass
    pred_class = int(out.item() > 0.5)  # binary decision

# Generate GradCAM heatmap
activation_map = cam_extractor(pred_class, out)

# Convert input image for display
to_pil = ToPILImage()
input_img = to_pil(input_tensor.squeeze(0).cpu())

# Convert activation map to 0-1 range
cam = activation_map[0].cpu().numpy()
cam = (cam - cam.min()) / (cam.max() - cam.min())

# Overlay GradCAM on image
plt.imshow(input_img)
plt.imshow(cam, cmap='jet', alpha=0.5)  # 0.5 opacity for overlay
plt.title("GradCAM Activation for Bird Prediction")
plt.axis('off')
plt.show()
