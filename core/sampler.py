import cv2
import numpy as np
import os
from typing import List, Tuple

class SubtitleSampler:
    """
    영상에서 자막 변화를 감지하여 최적의 프레임들을 추출하는 클래스.

    - 최소 샘플링 간격: 자막 화면에 머무르는 동안 프레임을 과도하게 쌓지 않도록,
      이전 저장 프레임으로부터 최소 N초가 지나야만 다음 프레임을 저장한다.
    - 최대 프레임 수: 제한을 초과하면 전체 타임라인에서 균등하게 서브샘플링한다.
    """
    def __init__(self, roi_bottom_percent=40, diff_threshold=5.0, min_interval_sec=1.0, max_frames=200):
        self.roi_bottom_percent = roi_bottom_percent
        self.diff_threshold = diff_threshold
        self.min_interval_sec = min_interval_sec  # 이전 저장 프레임과의 최소 간격(초)
        self.max_frames = max_frames              # 전체 최대 저장 프레임 수

    def extract_frames(self, video_path: str, output_folder: str, progress_callback=None) -> List[Tuple[float, str]]:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        roi_top = int(height * (1 - self.roi_bottom_percent / 100))
        
        prev_edge_roi = None
        saved_frames = []
        frame_count = 0
        last_saved_timestamp = -self.min_interval_sec  # 첫 프레임은 무조건 저장 가능하도록

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            
            roi = frame[roi_top:height, 0:width]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            is_changed = True
            if prev_edge_roi is not None:
                diff = cv2.absdiff(edges, prev_edge_roi)
                if np.mean(diff) < self.diff_threshold:
                    is_changed = False

            if is_changed:
                timestamp = frame_count / fps
                # 최소 간격 체크: 이전 저장보다 min_interval_sec 이상 지났을 때만 저장
                if timestamp - last_saved_timestamp >= self.min_interval_sec:
                    filename = f"frame_{frame_count:06d}_{timestamp:.2f}s.jpg"
                    filepath = os.path.join(output_folder, filename)
                    cv2.imwrite(filepath, frame)
                    saved_frames.append((timestamp, filepath))
                    last_saved_timestamp = timestamp
                # else: 변화가 있어도 시간이 너무 짧으면 스킵 (자막 화면 유지 중)
                prev_edge_roi = edges
            
            if progress_callback and frame_count % 30 == 0:
                progress_callback(frame_count, total_frames)

        cap.release()

        # 최대 프레임 수 제한: 너무 많으면 균등 서브샘플링
        if len(saved_frames) > self.max_frames:
            # 전체 길이에서 max_frames 개를 균등하게 뽑음
            total = len(saved_frames)
            step = total / self.max_frames
            sampled = [saved_frames[int(i * step)] for i in range(self.max_frames)]
            # 나머지 프레임 이미지 파일은 삭제하여 디스크 절약
            sampled_set = set(f[1] for f in sampled)
            for _, path in saved_frames:
                if path not in sampled_set and os.path.exists(path):
                    os.remove(path)
            saved_frames = sampled

        return saved_frames
