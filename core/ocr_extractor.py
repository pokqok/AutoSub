import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Callable, Optional


class SubtitleFrameFilter:
    """
    PaddleOCR Detection으로 자막 있는 프레임만 골라내는 필터.
    - 박스 좌표로 자막 ROI 크롭
    - 히스토그램으로 중복 프레임 제거 (95% threshold)
    - bbox 좌표 메타데이터 저장 (SyncRefiner용)
    """
    def __init__(self, interval_sec: float = 1.0, min_boxes: int = 1,
                 dup_threshold: float = 0.95, log_callback: Optional[Callable] = None):
        self.interval_sec = interval_sec
        self.min_boxes = min_boxes
        self.dup_threshold = dup_threshold
        self.log_callback = log_callback
        self.ocr = None

    def _init_ocr(self):
        import os as _os
        _os.environ['GLOG_minloglevel'] = '2'
        import logging
        logging.getLogger('ppocr').setLevel(logging.WARNING)
        logging.getLogger('paddle').setLevel(logging.WARNING)
        from paddleocr import PaddleOCR
        try:
            self.ocr = PaddleOCR(use_angle_cls=False, lang='japan', rec=False)
        except (ValueError, TypeError):
            self.ocr = PaddleOCR(use_angle_cls=False, lang='japan')

    def _log(self, msg: str):
        if self.log_callback:
            self.log_callback(msg)

    def _extract_boxes(self, ocr_res):
        """PaddleOCR 결과에서 4점 좌표 박스 리스트 반환"""
        if ocr_res is None:
            return []
        boxes = []
        # 구버전/신버전 형식 모두 처리
        if isinstance(ocr_res, list) and len(ocr_res) > 0:
            for item in ocr_res:
                if isinstance(item, list):
                    for sub in item:
                        if isinstance(sub, list) and len(sub) >= 1:
                            box = sub[0]
                            if isinstance(box, list) and len(box) == 4:
                                boxes.append(box)
                elif isinstance(item, dict):
                    if 'box' in item:
                        boxes.append(item['box'])
        return boxes

    @staticmethod
    def _get_subtitle_roi(frame: np.ndarray, boxes: List[List]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """모든 박스를 감싸는 최소 사각형으로 ROI 크롭"""
        all_pts = [pt for box in boxes for pt in box]
        x_coords = [p[0] for p in all_pts]
        y_coords = [p[1] for p in all_pts]
        h, w = frame.shape[:2]
        x1 = max(0, int(min(x_coords)))
        y1 = max(0, int(min(y_coords)))
        x2 = min(w, int(max(x_coords)))
        y2 = min(h, int(max(y_coords)))
        # 패딩 추가 (약간 여유 있게)
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        roi = frame[y1:y2, x1:x2]
        return roi, (x1, y1, x2, y2)

    @staticmethod
    def _is_similar_roi(roi1: np.ndarray, roi2: np.ndarray, threshold: float = 0.95) -> bool:
        """두 ROI의 히스토그램을 비교하여 유사도 판정"""
        if roi1 is None or roi2 is None:
            return False
        if roi1.size == 0 or roi2.size == 0:
            return False
        try:
            r1 = cv2.resize(roi1, (64, 64))
            r2 = cv2.resize(roi2, (64, 64))
            gray1 = cv2.cvtColor(r1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(r2, cv2.COLOR_BGR2GRAY)
            hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])
            cv2.normalize(hist1, hist1)
            cv2.normalize(hist2, hist2)
            corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return corr > threshold
        except Exception:
            return False

    def filter_frames(self, video_path: str, output_folder: str,
                      progress_callback=None) -> List[Dict]:
        """
        자막이 있는 프레임만 추출하여 output_folder에 저장.
        반환: [{"timestamp", "filepath", "bbox", "orig_w", "orig_h"}, ...]
        """
        if self.ocr is None:
            self._init_ocr()

        os.makedirs(output_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        interval_frames = max(1, int(fps * self.interval_sec))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        filtered: List[Dict] = []
        frame_idx = 0
        saved_count = 0
        skipped_count = 0
        dup_count = 0
        prev_roi = None

        self._log(f"  [Filter] Starting: video={os.path.basename(video_path)}, "
                  f"fps={fps:.1f}, interval={self.interval_sec}s, "
                  f"expected_checks≈{int(duration / self.interval_sec)}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                if frame_idx % interval_frames == 0:
                    curr_t = frame_idx / fps
                    if progress_callback:
                        progress_callback(frame_idx, total_frames)

                    # 4K 리사이즈
                    h, w = frame.shape[:2]
                    scale = 1.0
                    max_w = 1920
                    if w > max_w:
                        scale = max_w / w
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    try:
                        ocr_res = self.ocr.ocr(frame)
                        raw_boxes = self._extract_boxes(ocr_res)
                        box_count = len(raw_boxes)
                    except Exception as e:
                        self._log(f"  [Filter Error frame {frame_idx}] {str(e)}")
                        raw_boxes = []
                        box_count = 0

                    if box_count >= self.min_boxes:
                        roi, resized_bbox = self._get_subtitle_roi(frame, raw_boxes)

                        # 중복 체크
                        if prev_roi is not None and self._is_similar_roi(prev_roi, roi, self.dup_threshold):
                            dup_count += 1
                            continue

                        # bbox를 원본 해상도 기준으로 역변환
                        if scale != 1.0:
                            orig_bbox = tuple(int(v / scale) for v in resized_bbox)
                        else:
                            orig_bbox = resized_bbox

                        # 저장
                        filename = f"frame_{curr_t:.3f}.jpg"
                        filepath = os.path.join(output_folder, filename)
                        cv2.imwrite(filepath, roi, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                        filtered.append({
                            "timestamp": curr_t,
                            "filepath": filepath,
                            "bbox": orig_bbox,
                            "orig_w": orig_w,
                            "orig_h": orig_h
                        })
                        saved_count += 1
                        prev_roi = roi
                    else:
                        skipped_count += 1
        finally:
            cap.release()

        self._log(f"  [Filter Summary] checked: {frame_idx // interval_frames}, "
                  f"saved: {saved_count}, skipped: {skipped_count}, "
                  f"duplicates removed: {dup_count}, "
                  f"final: {len(filtered)}")
        return filtered
