# AeroBirds
AeroBirds is an experimental light-weight bird detection algorithm designed to detect birds via live video streams in airports.

# Algorithm
AeroBird first extracts a frame from a video stream from a chunk of five (can be adjusted via frame_skipping, increases performances but decreases accuracy). Afterwards, it preprocesses the image using edge/blod detection to pre-detect bird-like objects. It then 
