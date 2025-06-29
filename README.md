# AeroBirds
AeroBirds is an experimental light-weight bird detection algorithm designed to detect birds via live video streams in airports. Note this is all experimental and only for research/educational purposes. The results are too poor and innacurate for real-time bird detection.

### Pipeline
AeroBird first extracts a frame from a video stream from a chunk of five (can be adjusted via frame_skipping, increases performances but decreases accuracy). Afterwards, it preprocesses the image using:

  HSV colour filter for bird-like colours
  Blob detection for bird-like objects (20 by 20 masks)
  Motion extraction using frame-differencing

It then goes through a pre-trained custom light weight bird detector CNN trained on Airbirds dataset to then classify if a mask is a bird or not.
