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
    def _get_roi_coords(frame_shape, position=None, bbox=None):
        """
        ROI 좌표 계산. bbox가 있으면 그대로 사용, 없으면 position 기반 추정.
        """
        img_h, img_w = frame_shape[:2]
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            return x1, y1, x2, y2
        roi = POSITION_ROI.get(position, DEFAULT_ROI)
        x1 = int(img_w * roi[0])
        y1 = int(img_h * roi[1])
        x2 = int(img_w * roi[2])
        y2 = int(img_h * roi[3])
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
                           direction: str,
                           position=None,
                           bbox=None,
                           forced_end_t: float = None):
        """
        center_sec 주변에서 ROI diff가 가장 큰 지점을 찾습니다.
        direction: forward=end 보정(텍스트 사라짐 지점), backward=start 보정(텍스트 등장 지점)
        forced_end_t: forward 방향에서 검색 끝 경계를 외부에서 지정 (다음 자막 시작 전)
        """
        frame_shape = (
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )
        coords = self._get_roi_coords(frame_shape, position, bbox)

        # 탐색 범위
        if direction == "backward":
            start_t = max(0.0, center_sec - self.scan_radius_sec)
            end_t = center_sec + self.scan_radius_sec / 2
        else:  # forward
            start_t = center_sec
            if forced_end_t is not None:
                end_t = forced_end_t
            else:
                end_t = center_sec + self.scan_radius_sec

        # 0.1초 간격으로 프레임 추출
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
            # 의미 있는 변화가 없음: 자막이 검색 구간 끝까지 남아있음
            # → 구간 끝에서 사라진 것으로 처리
            if direction == "forward" and valid_times:
                return valid_times[-1]
            return center_sec

        max_idx = int(np.argmax(diffs))

        if direction == "backward":
            # 텍스트 등장: diff spike 직후 시점
            refined_t = valid_times[min(max_idx + 1, len(valid_times) - 1)]
        else:
            # 텍스트 사라짐: diff spike 지점이 end
            refined_t = valid_times[max_idx]

        return refined_t

    def refine(self, video_path: str, subtitles: List[Dict]) -> List[Dict]:
        """
        subtitles: VLM 결과
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
            bbox = sub.get("bbox")

            # 다음 자막 시작 시점 (없으면 무한대)
            next_start = subtitles[i + 1]["start"] if i + 1 < len(subtitles) else float('inf')

            # start 보정: 등장 지점 탐색
            refined_start = self._find_change_point(
                cap, fps, original_start, "backward", position, bbox
            )

            # end 보정: 사라짐 지점 탐색
            # 최소 0.3초는 표시, 다음 자막 0.1초 전까지만 검색
            search_limit = min(next_start - 0.1, original_start + 5.0)
            if search_limit <= original_start + 0.3:
                search_limit = original_start + 0.3

            refined_end = self._find_change_point(
                cap, fps, original_start + 0.3, "forward", position, bbox,
                forced_end_t=search_limit
            )

            # 다음 자막과 겹치지 않도록 clamp
            if refined_end > next_start - 0.05:
                refined_end = next_start - 0.05

            # 안전 검사
            if refined_end <= refined_start:
                refined_end = refined_start + 0.5

            refined.append({
                **sub,
                "start": refined_start,
                "end": refined_end,
            })

        cap.release()
        return refined
