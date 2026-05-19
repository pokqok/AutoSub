import cv2
import numpy as np
from typing import List, Dict, Optional

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

MAX_BBOX_RATIO = 0.20   # 화면 면적 20% 초과 bbox 무시
MAX_SEARCH_SEC = 15.0   # end 탐색 최대 범위
SPIKE_FACTOR   = 2.0    # baseline 대비 몇 배 이상이면 변화로 판정
MIN_DIFF       = 0.5    # 절대 최소 diff (노이즈 제거) — 정적 구간 diff 수준에 맞게 조정


class SyncRefiner:
    def __init__(self,
                 max_search_sec: float = MAX_SEARCH_SEC,
                 spike_factor: float = SPIKE_FACTOR):
        self.max_search_sec = max_search_sec
        self.spike_factor = spike_factor

    # ── 내부 유틸 ─────────────────────────────────────────────

    @staticmethod
    def _load_frame(filepath: str):
        """한글 경로 안전 로딩"""
        try:
            with open(filepath, 'rb') as f:
                buf = f.read()
            arr = np.frombuffer(buf, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    @staticmethod
    def _cap_bbox(bbox, orig_w: int, orig_h: int):
        """bbox가 너무 크면 무시 (세로쓰기 오감지 방어)"""
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        area_ratio = ((x2 - x1) * (y2 - y1)) / max(orig_w * orig_h, 1)
        return bbox if area_ratio <= MAX_BBOX_RATIO else None

    def _get_roi(self, frame_data: Dict, position, bbox) -> Optional[np.ndarray]:
        """프레임 로딩 + ROI 크롭"""
        frame = self._load_frame(frame_data["filepath"])
        if frame is None:
            return None
        h, w = frame.shape[:2]

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
        else:
            roi_ratio = POSITION_ROI.get(position, (0.0, 0.0, 1.0, 1.0))
            x1 = int(w * roi_ratio[0])
            y1 = int(h * roi_ratio[1])
            x2 = int(w * roi_ratio[2])
            y2 = int(h * roi_ratio[3])

        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _frame_diff(roi1, roi2) -> float:
        """ROI 간 평균 픽셀 차이"""
        if roi1 is None or roi2 is None:
            return 0.0
        try:
            r1 = cv2.resize(roi1, (64, 64))
            r2 = cv2.resize(roi2, (64, 64))
            return float(np.mean(cv2.absdiff(r1, r2)))
        except Exception:
            return 0.0

    def _get_frames_in_window(self, frame_list: List[Dict],
                               t_start: float, t_end: float) -> List[Dict]:
        """시간 범위 내 프레임 반환 (timestamp 순 정렬)"""
        frames = [f for f in frame_list if t_start <= f["timestamp"] <= t_end]
        return sorted(frames, key=lambda f: f["timestamp"])

    # ── 핵심 탐색 ─────────────────────────────────────────────

    def _find_end(self, frames: List[Dict], position, bbox) -> Optional[float]:
        """
        연속 프레임 diff로 자막 사라지는 시점 탐색.
        baseline(초반 안정 구간) 대비 spike_factor 배 이상 → 변화 시점.
        """
        if len(frames) < 2:
            print(f"[FIND_END] frames 부족: {len(frames)}개")
            return None

        diffs = []
        for i in range(1, len(frames)):
            roi1 = self._get_roi(frames[i - 1], position, bbox)
            roi2 = self._get_roi(frames[i],     position, bbox)
            d = self._frame_diff(roi1, roi2)
            diffs.append((frames[i]["timestamp"], d))

        baseline = max(float(np.percentile([d for _, d in diffs], 25)), MIN_DIFF)
        threshold = baseline * self.spike_factor

        # 로그 추가
        print(f"[FIND_END] frames={len(frames)}, baseline={baseline:.2f}, threshold={threshold:.2f}")
        print(f"[FIND_END] diffs={[(f'{t:.2f}', f'{d:.2f}') for t, d in diffs]}")

        for ts, d in diffs:
            if d >= threshold:
                return ts
        return None

    def _find_start(self, frames: List[Dict], position, bbox,
                    original_start: float) -> Optional[float]:
        """
        original_start 이전 구간에서 자막 등장 시점 탐색.
        뒤에서 앞으로 가면서 첫 번째 spike = 자막 등장.
        """
        if len(frames) < 2:
            return None

        diffs = []
        for i in range(1, len(frames)):
            roi1 = self._get_roi(frames[i - 1], position, bbox)
            roi2 = self._get_roi(frames[i],     position, bbox)
            d = self._frame_diff(roi1, roi2)
            diffs.append((frames[i]["timestamp"], d))

        if not diffs:
            return None

        # baseline: 전체 diff의 median (이상치에 강건)
        baseline_vals = [d for _, d in diffs]
        baseline = max(float(np.median(baseline_vals)), MIN_DIFF)
        threshold = baseline * self.spike_factor

        # original_start 이전에서 가장 가까운 spike
        best = None
        for ts, d in reversed(diffs):
            if ts >= original_start:
                continue
            if d >= threshold:
                best = ts
                break

        return best

    # ── 공개 인터페이스 ───────────────────────────────────────

    def refine(self, video_path: str, subtitles: List[Dict],
               frame_list: List[Dict] = None) -> List[Dict]:
        if not frame_list or not subtitles:
            return subtitles

        frame_list = sorted(frame_list, key=lambda f: f["timestamp"])
        max_t = frame_list[-1]["timestamp"]
        min_t = frame_list[0]["timestamp"]

        refined = []

        for i, sub in enumerate(subtitles):
            original_start = sub["start"]
            position       = sub.get("position")
            next_start     = subtitles[i + 1]["start"] if i + 1 < len(subtitles) else float('inf')

            # bbox: VLM bbox 또는 nearest frame bbox, 크기 제한 적용
            nearest = min(frame_list, key=lambda f: abs(f["timestamp"] - original_start))
            raw_bbox = sub.get("bbox") or nearest.get("bbox")
            orig_w = nearest.get("orig_w", 1920)
            orig_h = nearest.get("orig_h", 1080)
            bbox = self._cap_bbox(raw_bbox, orig_w, orig_h)

            # ── start 탐색 ──
            appear_frames = self._get_frames_in_window(
                frame_list,
                max(min_t, original_start - 1.0),
                original_start
            )
            refined_start = self._find_start(appear_frames, position, bbox, original_start)
            if refined_start is None:
                refined_start = original_start

            # ── end 탐색 ──
            vlm_end = sub["end"]
            # vlm_end와 현재 end(CRAFT end) 중 더 이른 쪽부터 탐색
            search_start_t = min(vlm_end, sub.get("vlm_end", vlm_end))
            disappear_frames = self._get_frames_in_window(
                frame_list,
                max(search_start_t - 1.0, min_t),
                min(vlm_end + 1.0, max_t)  # 소멸이 vlm_end 이후일 수도 있으므로 +1초 확장
            )
            refined_end = self._find_end(disappear_frames, position, bbox)

            # 탐색 실패 → VLM 원본 end 사용
            if refined_end is None:
                refined_end = sub["end"]

            # ── 안전장치 ──
            # next_start 침범 방지
            if next_start != float('inf'):
                refined_end = min(refined_end, next_start - 0.05)
            # 너무 짧으면 최소 0.3초 보장
            if refined_end <= refined_start:
                refined_end = refined_start + 0.3
            # 범위 초과 방지
            refined_end = min(refined_end, max_t)

            print(f"[P3] sub {i}: {original_start:.2f}~{sub['end']:.2f} "
                  f"→ {refined_start:.2f}~{refined_end:.2f}")

            refined.append({**sub, "start": refined_start, "end": refined_end})

        # ── 후처리: refined start 기준 겹침 최종 제거 ──
        for i in range(len(refined) - 1):
            if refined[i]["end"] > refined[i + 1]["start"]:
                old = refined[i]["end"]
                refined[i]["end"] = refined[i + 1]["start"] - 0.01
                if refined[i]["end"] <= refined[i]["start"]:
                    refined[i]["end"] = refined[i]["start"] + 0.1
                print(f"[P3-POST] sub {i}: end {old:.2f} → {refined[i]['end']:.2f} "
                      f"(next start={refined[i+1]['start']:.2f})")

        return refined
