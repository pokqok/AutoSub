import cv2
import numpy as np
import os
from typing import List, Tuple

class SubtitleSampler:
    """
    영상에서 자막 변화를 감지하여 최적의 프레임들을 추출하는 클래스
    """
    def __init__(self, roi_bottom_percent=40, diff_threshold=5.0):
        self.roi_bottom_percent = roi_bottom_percent
        self.diff_threshold = diff_threshold

    def extract_frames(self, video_path: str, output_folder: str, progress_callback=None) -> List[Tuple[float, str]]:
        """
        영상을 분석하여 자막 변화가 있을 때의 프레임들을 저장하고 (시간, 경로) 리스트를 반환합니다.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # ROI 설정: 하단 영역 분석
        roi_top = int(height * (1 - self.roi_bottom_percent / 100))
        
        prev_edge_roi = None
        saved_frames = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 1. ROI 추출 및 전처리
            roi = frame[roi_top:height, 0:width]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            # 2. 변화 감지 (Edge-based)
            is_changed = True
            if prev_edge_roi is not None:
                diff = cv2.absdiff(edges, prev_edge_roi)
                if np.mean(diff) < self.diff_threshold:
                    is_changed = False

            if is_changed:
                timestamp = frame_count / fps
                filename = f"frame_{frame_count:06d}_{timestamp:.2f}s.jpg"
                filepath = os.path.join(output_folder, filename)
                cv2.imwrite(filepath, frame)
                saved_frames.append((timestamp, filepath))
                prev_edge_roi = edges
            
            if progress_callback:
                progress_callback(frame_count, total_frames)

        cap.release()
        return saved_frames
