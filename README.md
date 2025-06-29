# AeroBirds
AeroBirds is an experimental light-weight bird detection algorithm designed to detect birds via live video streams in airports. Note this is all experimental and only for research/educational purposes. The results are too poor and inaccurate for real-time bird detection.

### Pipeline
AeroBird first extracts a frame from a video stream from a chunk of five (can be adjusted via frame_skipping, increases performances but decreases accuracy). Afterwards, it preprocesses the image using:

- HSV colour filter for bird-like colours
- Blob detection for bird-like objects (20 by 20 masks)
- Motion extraction using frame-differencing

It then goes through a pre-trained custom light weight bird detector CNN trained on Airbirds dataset to then classify if a mask is a bird or not.

### Libraries
The following libraries are used:

```bash
Python 3.7+
import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix

from torchcam.methods import GradCAM
```

### Test
To test the code, run example.py or the code below:

```bash
from Detector import Detector

Detector = Detector(video_path="test.mp4", frame_skipping=5)
Detector.run(show=True)
```
