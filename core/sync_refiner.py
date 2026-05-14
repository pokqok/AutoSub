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
    LLM이 반환한 대략적 타이밍을 이진 탐색으로 정밀 보정합니다.
    참조 ROI를 기준으로 텍스트 내용까지 구분하여 경계를 찾습니다.
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

    @staticmethod
    def _roi_similarity(roi1, roi2) -> float:
        """
        두 ROI의 유사도를 반환 (0~1). 배경 색 변화에 강건하도록
        엣지(Edge) 형태 + 구조적 유사도(NCC)를 결합합니다.
        """
        if roi1 is None or roi2 is None or roi1.size == 0 or roi2.size == 0:
            return 0.0
        try:
            r1 = cv2.resize(roi1, (64, 64))
            r2 = cv2.resize(roi2, (64, 64))
            g1 = cv2.cvtColor(r1, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(r2, cv2.COLOR_BGR2GRAY)

            # --- 1. 엣지(Edge) 유사도: 텍스트 형태 기반, 배경 색 무관 ---
            e1 = cv2.Canny(g1, 50, 150)
            e2 = cv2.Canny(g2, 50, 150)
            edge_count1 = np.count_nonzero(e1)
            edge_count2 = np.count_nonzero(e2)

            # 엣지가 너무 적으면(=자막 거의 없음) edge 신뢰도 낮음
            if edge_count1 < 30 or edge_count2 < 30:
                edge_sim = 0.0
            else:
                intersection = np.count_nonzero(np.logical_and(e1, e2))
                union = np.count_nonzero(np.logical_or(e1, e2))
                edge_sim = intersection / union if union > 0 else 0.0

            # --- 2. 구조적 유사도 (NCC): 전체 형태, 밝기/색상 무관 ---
            g1f = g1.astype(np.float32).flatten()
            g2f = g2.astype(np.float32).flatten()
            g1n = (g1f - g1f.mean()) / (g1f.std() + 1e-8)
            g2n = (g2f - g2f.mean()) / (g2f.std() + 1e-8)
            ncc = float(np.dot(g1n, g2n) / len(g1n))
            ncc = (np.clip(ncc, -1.0, 1.0) + 1.0) / 2.0  # -1..1 → 0..1

            # 최종: 엣지 형태를 우선(60%), 구조적 유사도 보조(40%)
            return 0.6 * edge_sim + 0.4 * ncc
        except Exception:
            return 0.0

    def _get_subtitle_roi_at(self, cap, t, position, bbox):
        """특정 시간(t) 프레임에서 ROI를 추출합니다."""
        cap.set(cv2.CAP_PROP_POS_MSEC, int(t * 1000))
        ret, frame = cap.read()
        if not ret:
            return None
        frame_shape = frame.shape[:2]
        coords = self._get_roi_coords(frame_shape, position, bbox)
        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _is_same_subtitle(self, cap, t, ref_roi, position, bbox, threshold=0.75) -> bool:
        """참조 ROI와 현재 시간 t의 ROI를 비교하여 같은 자막인지 판단합니다."""
        current_roi = self._get_subtitle_roi_at(cap, t, position, bbox)
        if current_roi is None or current_roi.size == 0:
            return False
        sim = self._roi_similarity(current_roi, ref_roi)
        return sim >= threshold

    def _binary_search_edge(self, cap, ref_roi, position, bbox, lo, hi,
                          direction: str, min_resolution: float = 0.1) -> float:
        """
        이진 탐색으로 자막 경계를 좁혀갑니다.
        direction="appear": lo에 SAME이 없음(또는 DIFF), hi에 SAME이 있음
        direction="disappear": lo에 SAME이 있음, hi에 SAME이 없음(또는 DIFF)
        """
        while (hi - lo) > min_resolution:
            mid = (lo + hi) / 2
            is_same = self._is_same_subtitle(cap, mid, ref_roi, position, bbox)
            if direction == "appear":
                if is_same:
                    hi = mid
                else:
                    lo = mid
            else:  # disappear
                if is_same:
                    lo = mid
                else:
                    hi = mid
        return hi if direction == "appear" else lo

    def _find_appearance(self, cap, marker_start, position=None, bbox=None) -> float:
        """
        기본 marker_start 앞 1초에서 시작.
        SAME이면 0.5초씩 더 앞으로 확장 (최대 3번).
        """
        ref_roi = self._get_subtitle_roi_at(cap, marker_start, position, bbox)
        if ref_roi is None:
            return marker_start
        lo = max(0.0, marker_start - 1.0)
        hi = marker_start
        # lo에도 SAME이면 앞으로 확장
        extend = 0
        while self._is_same_subtitle(cap, lo, ref_roi, position, bbox) and extend < 3:
            lo = max(0.0, lo - 0.5)
            extend += 1
        return self._binary_search_edge(cap, ref_roi, position, bbox, lo, hi, "appear")

    def _find_disappearance(self, cap, marker_end, next_start,
                            position=None, bbox=None) -> float:
        """
        기본 marker_end 뒤 1초에서 시작.
        SAME이면 0.5초씩 더 뒤로 확장 (최대 3번, 다음 자막 전까지만).
        """
        ref_roi = self._get_subtitle_roi_at(cap, marker_end, position, bbox)
        if ref_roi is None:
            return marker_end
        lo = marker_end
        search_limit = next_start - 0.05 if next_start != float('inf') else marker_end + 1.0
        hi = min(search_limit, marker_end + 1.0)
        # hi에도 SAME이면 뒤로 확장
        extend = 0
        while (self._is_same_subtitle(cap, hi, ref_roi, position, bbox)
               and extend < 3 and hi < search_limit - 0.1):
            hi = min(hi + 0.5, search_limit)
            extend += 1
        return self._binary_search_edge(cap, ref_roi, position, bbox, lo, hi, "disappear")

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

            # 등장 지점: original_start 앞쪽에서 이진 탐색
            refined_start = self._find_appearance(
                cap, original_start, position, bbox
            )

            # 사라짐 지점: original_end 뒤쪽에서 이진 탐색
            # 최소 0.3초 후부터 (등장 직후 바로 사라지는 건 막기 위해)
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
