from Detector import Detector

Detector = Detector(video_path="test.mp4", frame_skipping=5)
Detector.run(show=True)