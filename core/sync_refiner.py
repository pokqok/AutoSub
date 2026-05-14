import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

# position -> (x1_ratio, y1_ratio, x2_ratio, y2_ratio) for ROI
POSITION_ROI = {
    "top-left":      (0.0,  0.0,  0.33, 0.33),
    "top-center":    (0.33, 0.0,  0.66, 0.33),
    "top-right":     (0.66, 0.0,  1.0,  0.33),
    "middle-left":   (0.0,  0.33, 0.33, 0.66),
    "middle-center": (0.33, 0.33, 0.66, 0.66),
    "middle-right":  (0.66, 0.33, 1.0,  0.66),
    "bottom-left":   (0.0,  0.66, 0.33, 1.0),
    "bottom-center": (0.33, 0.66, 0.66, 1.0),
    "bottom-right":  (0.66, 0.66, 1.0,  1.0),
}
DEFAULT_ROI = (0.0, 0.0, 1.0, 1.0)


class SyncRefiner:
    """
    LLM이 반환한 대략적 타이밍을 프레임 단위(0.05초)로 정밀 보정합니다.
    position 필드 기반 ROI를 사용하여 배경 노이즈를 최소화합니다.
    """
    def __init__(self, scan_radius_sec: float = 0.5, scan_interval_sec: float = 0.1,
                 diff_threshold: float = 15.0):
        self.scan_radius_sec = scan_radius_sec
        self.scan_interval_sec = scan_interval_sec
        self.diff_threshold = diff_threshold

    @staticmethod
    def _get_roi_coords(frame_shape: Tuple[int, ...],
                        position: Optional[str]) -> Tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        roi = POSITION_ROI.get(position, DEFAULT_ROI)
        x1 = int(w * roi[0])
        y1 = int(h * roi[1])
        x2 = int(w * roi[2])
        y2 = int(h * roi[3])
        return x1, y1, x2, y2

    @staticmethod
    def _extract_roi_gray(frame: np.ndarray, coords: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = coords
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        return gray

    def _find_change_point(self, cap: cv2.VideoCapture,
                           fps: float,
                           center_sec: float,
                           direction: str,  # "forward" or "backward"
                           position: Optional[str]) -> float:
        """
        center_sec 주변에서 ROI diff가 가장 큰 지점을 찾습니다.
        direction: forward=end 보정(텍스트 사라짐 지점), backward=start 보정(텍스트 등장 지점)
        """
        frame_shape = (
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )
        coords = self._get_roi_coords(frame_shape, position)

        # 탐색 범위
        half_radius = self.scan_radius_sec / 2
        if direction == "backward":
            start_t = max(0.0, center_sec - self.scan_radius_sec)
            end_t = center_sec + half_radius
        else:  # forward
            start_t = max(0.0, center_sec - half_radius)
            end_t = center_sec + self.scan_radius_sec

        # 0.05초 간격으로 프레임 추출
        scan_times = np.arange(start_t, end_t, self.scan_interval_sec)
        if len(scan_times) < 2:
            return center_sec

        frames_gray = []
        valid_times = []
        for t in scan_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, int(t * 1000))
            ret, frame = cap.read()
            if not ret:
                continue
            gray = self._extract_roi_gray(frame, coords)
            frames_gray.append(gray)
            valid_times.append(t)

        if len(frames_gray) < 2:
            return center_sec

        # diff 계산: 연속된 프레임들의 차이
        diffs = []
        for i in range(len(frames_gray) - 1):
            # resize to same size if needed
            if frames_gray[i].shape != frames_gray[i+1].shape:
                h, w = frames_gray[i].shape
                next_resized = cv2.resize(frames_gray[i+1], (w, h))
            else:
                next_resized = frames_gray[i+1]
            diff = cv2.absdiff(frames_gray[i], next_resized)
            mean_diff = float(np.mean(diff))
            diffs.append(mean_diff)

        if not diffs:
            return center_sec

        max_diff = max(diffs)
        if max_diff < self.diff_threshold:
            # 의미 있는 변화가 없음: 보정 불필요
            return center_sec

        max_idx = int(np.argmax(diffs))

        if direction == "backward":
            # 텍스트 등장: diff가 spike되기 직전이 start
            # spike 지점의 다음 프레임 시간이 텍스트가 나타난 시점
            refined_t = valid_times[min(max_idx + 1, len(valid_times) - 1)]
        else:
            # 텍스트 사라짐: diff spike 지점이 end
            refined_t = valid_times[max_idx]

        return refined_t

    def refine(self, video_path: str, subtitles: List[Dict]) -> List[Dict]:
        """
        subtitles: LLM 결과
        반환: 동일 구조, start/end만 정밀 보정
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)

        refined: List[Dict] = []
        for i, sub in enumerate(subtitles):
            position = sub.get("position")
            original_start = sub["start"]
            original_end = sub["end"]

            # start 보정
            refined_start = self._find_change_point(
                cap, fps, original_start, "backward", position
            )

            # end 보정
            refined_end = self._find_change_point(
                cap, fps, original_end, "forward", position
            )

            # 안전 검사: end < start 방지
            if refined_end < refined_start:
                refined_end = refined_start + 1.0

            refined.append({
                **sub,
                "start": refined_start,
                "end": refined_end,
            })

        cap.release()
        return refined
