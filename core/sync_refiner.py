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

    def _binary_search_edge(self, cap: cv2.VideoCapture,
                           position=None,
                           bbox=None,
                           lo: float = 0.0,
                           hi: float = 10.0,
                           direction: str = "appear",  # "appear" or "disappear"
                           min_resolution: float = 0.1) -> float:
        """
        이진 탐색으로 자막 등장/사라짐 경계를 0.1초까지 좁혀 찾습니다.
        direction="appear":  lo(없음) --- hi(있음)  경계 찾기
        direction="disappear": lo(있음) --- hi(없음)  경계 찾기
        """
        frame_shape = (
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )
        coords = self._get_roi_coords(frame_shape, position, bbox)

        # ROI가 비어있으면 그대로 반환
        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            return lo if direction == "appear" else hi

        def _has_text_at(t: float) -> bool:
            cap.set(cv2.CAP_PROP_POS_MSEC, int(t * 1000))
            ret, frame = cap.read()
            if not ret:
                return False
            gray = self._extract_roi_gray(frame, coords)
            if gray.size == 0:
                return False
            # 텍스트 존재 여부: 평균 밝기가 어두운 편이면 텍스트 있음으로 간주
            mean_val = float(np.mean(gray))
            return mean_val < 200  # 임계값: 밝으면 배경, 어두우면 텍스트

        # 이진 탐색
        while (hi - lo) > min_resolution:
            mid = (lo + hi) / 2
            has_text = _has_text_at(mid)

            if direction == "appear":
                # lo = 없음, hi = 있음
                if has_text:
                    hi = mid
                else:
                    lo = mid
            else:  # disappear
                # lo = 있음, hi = 없음
                if has_text:
                    lo = mid
                else:
                    hi = mid

        return hi if direction == "appear" else lo

    def _find_appearance(self, cap: cv2.VideoCapture,
                        marker_start: float,
                        position=None,
                        bbox=None) -> float:
        """
        기본 marker_start 앞 1초에서 시작.
        텍스트가 있으면 0.5초씩 더 앞으로 확장 (최대 3번).
        """
        lo = max(0.0, marker_start - 1.0)
        hi = marker_start
        # lo에 텍스트가 있으면 점진 확장
        extend = 0
        while self._has_text_at(cap, lo, position, bbox) and extend < 3:
            lo = max(0.0, lo - 0.5)
            extend += 1
        return self._binary_search_edge(cap, position, bbox, lo, hi, "appear")

    def _find_disappearance(self, cap: cv2.VideoCapture,
                           marker_end: float,
                           next_start: float,
                           position=None,
                           bbox=None) -> float:
        """
        기본 marker_end 뒤 1초에서 시작.
        텍스트가 있으면 0.5초씩 더 뒤로 확장 (최대 3번).
        """
        lo = marker_end
        # 기본 사라짐 탐색 범위: marker_end + 1초 (다음 자막 전이면 더 짧게)
        search_limit = next_start - 0.05 if next_start != float('inf') else marker_end + 1.0
        hi = min(search_limit, marker_end + 1.0)
        # hi에 텍스트가 있으면 점진 확장
        extend = 0
        while self._has_text_at(cap, hi, position, bbox) and extend < 3 and hi < search_limit - 0.1:
            hi = min(hi + 0.5, search_limit)
            extend += 1
        return self._binary_search_edge(cap, position, bbox, lo, hi, "disappear")

    def _has_text_at(self, cap, t, position, bbox) -> bool:
        frame_shape = (
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )
        coords = self._get_roi_coords(frame_shape, position, bbox)
        cap.set(cv2.CAP_PROP_POS_MSEC, int(t * 1000))
        ret, frame = cap.read()
        if not ret:
            return False
        gray = self._extract_roi_gray(frame, coords)
        if gray.size == 0:
            return False
        mean_val = float(np.mean(gray))
        return mean_val < 200

    def refine(self, video_path: str, subtitles: List[Dict]) -> List[Dict]:
        """
        subtitles: VLM 결과
        반환: 동일 구조, start/end만 정밀 보정
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        refined: List[Dict] = []
        for i, sub in enumerate(subtitles):
            position = sub.get("position")
            original_start = sub["start"]
            original_end = sub["end"]
            bbox = sub.get("bbox")
            next_start = subtitles[i + 1]["start"] if i + 1 < len(subtitles) else float('inf')

            # 등장 지점: original_start 앞쪽 2초를 이진 탐색
            refined_start = self._find_appearance(
                cap, original_start, position, bbox
            )

            # 사라짐 지점: original_end 뒤쪽을 이진 탐색 (다음 자막 전까지)
            # 사라짐 탐색 시작점: 최소 0.3초 후부터 (등장 직후 바로 사라지는 건 막기 위해)
            disappear_search_start = max(original_start + 0.3, refined_start + 0.2)
            refined_end = self._find_disappearance(
                cap, disappear_search_start, next_start, position, bbox
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
