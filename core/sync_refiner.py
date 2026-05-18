import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

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
    VLM이 반환한 대략적 타이밍을 이진 탐색으로 정밀 보정합니다.
    Phase 1에서 저장된 크롭 프레임 파일들을 직접 읽어 비교합니다.
    """
    def __init__(self):
        pass

    @staticmethod
    def _get_roi_coords(frame_shape, position=None, bbox=None):
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
    def _roi_similarity(roi1, roi2) -> float:
        """Edge 60% + NCC 40% — 배경 색 변화에 강건"""
        if roi1 is None or roi2 is None or roi1.size == 0 or roi2.size == 0:
            return 0.0
        try:
            r1 = cv2.resize(roi1, (64, 64))
            r2 = cv2.resize(roi2, (64, 64))
            g1 = cv2.cvtColor(r1, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(r2, cv2.COLOR_BGR2GRAY)

            e1 = cv2.Canny(g1, 50, 150)
            e2 = cv2.Canny(g2, 50, 150)
            ec1 = np.count_nonzero(e1)
            ec2 = np.count_nonzero(e2)
            if ec1 < 10 or ec2 < 10:
                edge_sim = 0.0
            else:
                inter = np.count_nonzero(np.logical_and(e1, e2))
                union = np.count_nonzero(np.logical_or(e1, e2))
                edge_sim = inter / union if union > 0 else 0.0

            g1f = g1.astype(np.float32).flatten()
            g2f = g2.astype(np.float32).flatten()
            g1n = (g1f - g1f.mean()) / (g1f.std() + 1e-8)
            g2n = (g2f - g2f.mean()) / (g2f.std() + 1e-8)
            ncc = float(np.dot(g1n, g2n) / len(g1n))
            ncc = (np.clip(ncc, -1.0, 1.0) + 1.0) / 2.0

            return 0.6 * edge_sim + 0.4 * ncc
        except Exception:
            return 0.0

    @staticmethod
    def _has_subtitle(roi) -> bool:
        """ROI에 자막 텍스트가 남아있는지 edge 개수로 빠르게 판정. 배경만 남았으면 False."""
        if roi is None or roi.size == 0:
            return False
        try:
            g = cv2.cvtColor(cv2.resize(roi, (64, 64)), cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(g, 50, 150)
            return np.count_nonzero(edges) >= 10
        except Exception:
            return True

    def _get_nearest_frame(self, t: float, frame_list: List[Dict], max_gap: float = 1.0) -> Optional[Dict]:
        """
        시간 t에 가장 가까운 저장된 프레임을 반환.
        gap이 max_gap(기본 1.0초)를 초과하면 None → '자막 없음'으로 간주.
        """
        nearest = None
        min_gap = float('inf')
        for f in frame_list:
            gap = abs(f["timestamp"] - t)
            if gap < min_gap:
                min_gap = gap
                nearest = f
        if nearest and min_gap <= max_gap:
            return nearest
        return None

    def _get_subtitle_roi_at(self, t: float, position, bbox, frame_list: List[Dict]):
        """특정 시간 t 근처의 저장된 프레임을 읽어 ROI를 추출합니다. 없으면 None."""
        print(f"[ROI DEBUG] looking for t={t:.2f}, frame_list size={len(frame_list)}")
        print(f"[ROI DEBUG] frame_list timestamps: {[f['timestamp'] for f in frame_list[:5]]}")
        frame_data = self._get_nearest_frame(t, frame_list)
        print(f"[ROI DEBUG] nearest frame: {frame_data}")
        if frame_data is None:
            return None
        # cv2.imread 안 씀 — Windows 한글 경로 버그
        with open(frame_data["filepath"], 'rb') as f:
            img_bytes = f.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        actual_bbox = bbox if bbox is not None else frame_data.get("bbox")
        coords = self._get_roi_coords(frame.shape[:2], position, actual_bbox)
        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _is_same_subtitle(self, t: float, ref_roi, position, bbox, frame_list: List[Dict], threshold=0.55) -> bool:
        """참조 ROI와 시간 t 근처 프레임의 ROI를 비교."""
        current_roi = self._get_subtitle_roi_at(t, position, bbox, frame_list)
        if current_roi is None or current_roi.size == 0:
            return False
        # _has_subtitle 체크 제거 — 짧은 세로쓰기 자막 오탐(False Negative) 방지.
        # tight CRAFT bbox 쓰면 자막 없는 배경은 ref_roi와 similarity가 자연히 낮아짐.
        sim = self._roi_similarity(current_roi, ref_roi)
        return sim >= threshold

    def _binary_search_edge(self, ref_roi, position, bbox, lo: float, hi: float,
                            direction: str, frame_list: List[Dict],
                            threshold: float = 0.55, min_resolution: float = 0.1) -> float:
        """이진 탐색으로 자막 경계를 좁혀갑니다."""
        while (hi - lo) > min_resolution:
            mid = (lo + hi) / 2
            is_same = self._is_same_subtitle(mid, ref_roi, position, bbox, frame_list, threshold)
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

    def _verify_boundary(self, boundary: float, direction: str, ref_roi, position, bbox,
                         frame_list: List[Dict], threshold: float = 0.55) -> float:
        """
        B. 경계 ±0.1s 프레임 추가 검증.
        불일치 시 0.1s씩 보정 재시도 (최대 3회).
        """
        min_t = min(f["timestamp"] for f in frame_list) if frame_list else 0.0
        max_t = max(f["timestamp"] for f in frame_list) if frame_list else boundary

        max_iter = 10
        iter_count = 0
        while iter_count < max_iter:
            iter_count += 1
            before = self._is_same_subtitle(boundary - 0.1, ref_roi, position, bbox, frame_list, threshold)
            after  = self._is_same_subtitle(boundary + 0.1, ref_roi, position, bbox, frame_list, threshold)

            if direction == "appear":
                # before = 없음(False), after = 있음(True) 가 정상
                if (not before) and after:
                    return boundary
                if before and after:
                    boundary = max(min_t, boundary - 0.1)   # 너무 뒤 → 앞으로
                elif (not before) and (not after):
                    boundary = min(max_t, boundary + 0.1)   # 너무 앞 → 뒤로
                else:
                    break  # before=T, after=F ?  역전된 경우 → 그냥 반환
            else:  # disappear
                # before = 있음(True), after = 없음(False) 가 정상
                if before and (not after):
                    return boundary
                if before and after:
                    boundary = min(max_t, boundary + 0.1)   # 너무 앞 → 뒤로
                elif (not before) and (not after):
                    boundary = max(min_t, boundary - 0.1)   # 너무 뒤 → 앞으로
                else:
                    break  # before=F, after=T ?  역전된 경우 → 그냥 반환
        return boundary

    def _find_appearance(self, marker_start: float, position=None, bbox=None,
                         frame_list: List[Dict] = None,
                         ref_roi=None,
                         threshold: float = 0.55) -> float:
        """기본 marker_start 앞 1초에서 시작. SAME이면 0.5초씩 더 앞으로 확장."""
        if not frame_list:
            return marker_start
        if ref_roi is None:
            ref_roi = self._get_subtitle_roi_at(marker_start, position, bbox, frame_list)
        if ref_roi is None:
            return marker_start

        min_t = min(f["timestamp"] for f in frame_list)
        lo = max(min_t, marker_start - 1.0)
        hi = marker_start

        while self._is_same_subtitle(lo, ref_roi, position, bbox, frame_list, threshold) and lo > min_t:
            lo = max(min_t, lo - 0.5)

        return self._binary_search_edge(
            ref_roi, position, bbox, lo, hi, "appear", frame_list, threshold
        )

    def _find_disappearance(self, marker_end: float, next_start: float,
                            position=None, bbox=None,
                            frame_list: List[Dict] = None,
                            ref_roi=None,
                            threshold: float = 0.55) -> float:
        """기본 marker_end 뒤 1초에서 시작. SAME이면 0.5초씩 더 뒤로 확장."""
        if not frame_list:
            return marker_end
        if ref_roi is None:
            ref_roi = self._get_subtitle_roi_at(marker_end, position, bbox, frame_list)
        if ref_roi is None:
            return marker_end

        max_t = max(f["timestamp"] for f in frame_list)
        lo = marker_end
        search_limit = next_start if next_start != float('inf') else marker_end + 5.0
        hi = min(search_limit, marker_end + 2.0, max_t)

        while (self._is_same_subtitle(hi, ref_roi, position, bbox, frame_list, threshold)
               and hi < search_limit - 0.1):
            hi = min(hi + 0.5, search_limit, max_t)

        return self._binary_search_edge(
            ref_roi, position, bbox, lo, hi, "disappear", frame_list, threshold
        )

    def refine(self, video_path: str, subtitles: List[Dict],
               frame_list: List[Dict] = None) -> List[Dict]:
        """
        subtitles: VLM 결과
        frame_list: Phase 1에서 저장된 [{timestamp, filepath, bbox, orig_w, orig_h}, ...]
        반환: 동일 구조, start/end만 정밀 보정
        """
        if not frame_list:
            print("[SyncRefiner] Warning: No frame_list provided. Skipping sync.")
            return subtitles

        refined: List[Dict] = []
        min_t = min(f["timestamp"] for f in frame_list)

        for i, sub in enumerate(subtitles):
            position = sub.get("position")
            original_start = sub["start"]
            # VLM 결과에 bbox 없으면 nearest frame의 bbox 사용
            nearest = self._get_nearest_frame(original_start, frame_list)
            bbox = sub.get("bbox") or (nearest.get("bbox") if nearest else None)
            next_start = subtitles[i + 1]["start"] if i + 1 < len(subtitles) else float('inf')

            print(f"[P3 IN ] sub {i}: start={original_start:.2f} end={sub['end']:.2f} "
                  f"next_start={next_start:.2f} text='{sub.get('translated','')[:20]}'")

            ref_roi = self._get_subtitle_roi_at(original_start, position, bbox, frame_list)
            print(f"[P3 ROI] sub {i}: ref_roi={'OK' if ref_roi is not None else 'NONE'} "
                  f"bbox={bbox} position={position}")
            if ref_roi is None or ref_roi.size == 0:
                # 스킵 경로: start==end 보정 추가
                sub_copy = dict(sub)
                if sub_copy["end"] <= sub_copy["start"]:
                    sub_copy["end"] = sub_copy["start"] + 1.0
                refined.append(sub_copy)
                continue

            # ★ 고정 threshold: 배경 유사도에 따라 threshold가 변하면 타이밍이 오락가락함
            adaptive_threshold = 0.55
            print(f"[P3 THR] sub {i}: fixed_threshold={adaptive_threshold:.3f}")

            refined_start = self._find_appearance(
                original_start, position, bbox, frame_list, ref_roi, adaptive_threshold
            )

            # ★ Phase 1 disappear 마커가 있으면 binary search 스킵
            disappear_frame = next((f for f in frame_list
                                    if f.get('is_disappear') and f['timestamp'] > original_start), None)
            if disappear_frame:
                print(f"[SYNC] Using disappear marker: {disappear_frame['timestamp']:.2f}")
                refined_end = disappear_frame['timestamp']
            else:
                disappear_search_start = max(original_start + 0.5, refined_start + 0.4)
                refined_end = self._find_disappearance(
                    disappear_search_start, next_start, position, bbox,
                    frame_list, ref_roi, adaptive_threshold
                )

            if refined_end > next_start:
                refined_end = next_start
            if refined_end - refined_start < 1.0:
                refined_end = refined_start + 1.0

            print(f"[P3 OUT] sub {i}: refined_start={refined_start:.2f} "
                  f"refined_end={refined_end:.2f} "
                  f"(was {original_start:.2f}~{sub['end']:.2f})")
            refined.append({**sub, "start": refined_start, "end": refined_end})

        return refined
