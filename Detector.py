import cv2
import numpy as np
import torch
from torchvision import transforms
from CNN import BirdCNN

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BirdCNN().to(device)
model.load_state_dict(torch.load('models/bird_cnn_best.pth'))  # Ensure this path is correct
model.eval()

# Define the transformations
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert to PIL Image
    transforms.Resize((40, 40)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
])

class Detector:
    def __init__(self, video_path, frame_skipping):
        self.cap = cv2.VideoCapture(video_path)  # Resize the video capture to 640x480
        self.ret, self.current_frame = self.cap.read()
        self.current_frame = cv2.resize(self.current_frame, (1920, 1080))  # Resize the frame to 640x480
        self.previous_frame = self.current_frame.copy()
        self.detector = self._init_blob_detector()
        self.frame_number = 0
        self.frame_skipping = frame_skipping  # Process every 5th frame
        self.threshold = 0.37  # Determine via PRC
        self.lk_params = dict( winSize  = (40, 40), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.keypoints = np.array([])
        
    def is_bird_coloured(self, patch, bird_hue_ranges=None, sat_thresh=(30, 255), val_thresh=(30, 255)):
        """
        Simple HSV colour filter for bird-like colours.
        """
        if bird_hue_ranges is None:
            # Define rough hue ranges for browns, yellows, reds, blues
            bird_hue_ranges = [
                (0, 15),     # reds & browns
                (15, 35),    # yellows
                (90, 135),   # blues
                # exclude full (0–180) or grey will overpower
            ]
        
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        mask = np.zeros(h.shape, dtype=np.uint8)
        for (lower, upper) in bird_hue_ranges:
            hue_mask = cv2.inRange(h, lower, upper)
            sat_mask = cv2.inRange(s, sat_thresh[0], sat_thresh[1])
            val_mask = cv2.inRange(v, val_thresh[0], val_thresh[1])
            combined = cv2.bitwise_and(hue_mask, sat_mask)
            combined = cv2.bitwise_and(combined, val_mask)
            mask = cv2.bitwise_or(mask, combined)

        # If significant portion of patch matches, keep
        birdish_ratio = np.sum(mask > 0) / mask.size
        return birdish_ratio > 0.1  # tweak this threshold as needed
    
    def _init_blob_detector(self):
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 10
        params.maxThreshold = 200  # Increased maxThreshold
        params.filterByArea = True
        params.minArea = 30  # Increased minArea
        params.maxArea = 500 # Increased maxArea
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = True
        params.filterByColor = False  # Disable color filtering
        return cv2.SimpleBlobDetector_create(params)

    def _get_motion_mask(self):
        current_frame_gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        previous_frame_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray).astype(np.float32)
        motion_weights = frame_diff / 255.0
        motion_weights[motion_weights < 0.1] = 0
        weighted_frame = current_frame_gray.astype(np.float32) * motion_weights
        return np.clip(weighted_frame * 255, 0, 255).astype(np.uint8)

    def filter_keypoints(self, keypoints):
        filtered_keypoints = []
        for _, keypoint in enumerate(keypoints):
            x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
            size = int(keypoint.size) // 2
            x1 = max(0, x - size)
            y1 = max(0, y - size)
            x2 = min(self.current_frame.shape[1], x + size)
            y2 = min(self.current_frame.shape[0], y + size)
            patch = self.current_frame[y1:y2, x1:x2]
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)  # Convert to RGB
            
            if patch.shape[0] > 5 and patch.shape[1] > 5:
                patch = cv2.resize(patch, (40, 40))
                
                if self.is_bird_coloured(patch):
                    patch = transform(patch).unsqueeze(0)  # Apply transformations

                    with torch.no_grad():
                        pred = model(patch)

                    if pred >= self.threshold:
                        filtered_keypoints.append(keypoint)
                else:
                    continue
        return filtered_keypoints

    def optical_flow(self, prev_gray, curr_gray, points):
        p0 = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)
        return p1[st==1], p0[st==1]

    def is_inside_existing(self, pt, tracked_pts):
        for tracked in tracked_pts:
            tx, ty = tracked.ravel()
            if abs(pt[0] - tx) <= 20 and abs(pt[1] - ty) <= 20:
                return True
        return False

    def run(self, show):
        self.frame_number = 1
        keypoint_coords = np.array([kp.pt for kp in self.keypoints], dtype=np.float32).reshape(-1, 1, 2)

        while self.cap.isOpened() and self.ret:
            prev_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)

            if keypoint_coords is not None and len(keypoint_coords) > 0:
                next_pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, keypoint_coords, None, **self.lk_params)
                keypoint_coords = next_pts[st == 1].reshape(-1, 1, 2)  # Keep only valid points

            if self.frame_number % self.frame_skipping == 0:
                # Redetect on motion map
                motion_frame = self._get_motion_mask()
                new_kps = self.detector.detect(motion_frame)
                new_kps = self.filter_keypoints(new_kps)
                new_coords = np.array([kp.pt for kp in new_kps], dtype=np.float32).reshape(-1, 1, 2)

                # Combine with tracked points, remove near duplicates
                if keypoint_coords is not None and len(keypoint_coords) > 0:
                    combined = np.concatenate((keypoint_coords, new_coords), axis=0)
                    # Remove near-duplicates
                    _, unique_idx = np.unique(combined.round(decimals=1), axis=0, return_index=True)
                    keypoint_coords = combined[np.sort(unique_idx)]
                else:
                    keypoint_coords = new_coords

            if show and keypoint_coords is not None:
                vis_frame = self.current_frame.copy()
                for pt in keypoint_coords:
                    x, y = int(pt[0][0]), int(pt[0][1])
                    cv2.circle(vis_frame, (x, y), 5, (0, 0, 255), -1)
                cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL) 
                cv2.resizeWindow("Tracking", 640, 360)
                cv2.imshow("Tracking", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Tracking", cv2.WND_PROP_VISIBLE) < 1:
                    break

            self.previous_frame = self.current_frame.copy()
            self.ret, self.current_frame = self.cap.read()
            self.frame_number += 1

        self.cap.release()
        cv2.destroyAllWindows()

