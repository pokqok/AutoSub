import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Callable, Optional


class SubtitleFrameFilter:
    """
    PaddleOCR Detection-Only로 자막 있는 프레임만 골라내는 필터.
    텍스트 인식(rec)은 하지 않아 속도가 3~5배 빠름.
    """
    def __init__(self, interval_sec: float = 1.0, min_boxes: int = 1,
                 log_callback: Optional[Callable] = None):
        self.interval_sec = interval_sec
        self.min_boxes = min_boxes  # 감지된 텍스트 박스 최소 개수
        self.log_callback = log_callback
        self.ocr = None

    def _init_ocr(self):
        import os as _os
        _os.environ['GLOG_minloglevel'] = '2'
        import logging
        logging.getLogger('ppocr').setLevel(logging.WARNING)
        logging.getLogger('paddle').setLevel(logging.WARNING)
        from paddleocr import PaddleOCR
        # rec=False: 텍스트 인식 OFF, 감지만
        self.ocr = PaddleOCR(use_angle_cls=False, lang='japan', rec=False)

    def _log(self, msg: str):
        if self.log_callback:
            self.log_callback(msg)

    def _normalize_det_result(self, ocr_res):
        """PaddleOCR Detection 결과 정규화: 텍스트 박스 개수 반환"""
        if ocr_res is None:
            return 0
        if isinstance(ocr_res, dict):
            # 최신 PaddleX format
            results = []
            self._extract_boxes(ocr_res, results)
            return len(results)
        if not isinstance(ocr_res, (list, tuple)):
            return 0
        if len(ocr_res) == 0:
            return 0
        first = ocr_res[0]
        if first is None:
            return 0
        if isinstance(first, dict):
            results = []
            self._extract_boxes(first, results)
            return len(results)
        if isinstance(first, (list, tuple)) and len(first) > 0:
            if isinstance(first[0], (list, tuple)):
                return len(first)
            else:
                return len(list(ocr_res))
        return len(list(ocr_res))

    def _extract_boxes(self, obj, results):
        """dict/list 내부에서 bbox 좌표들을 재귀 탐색"""
        if isinstance(obj, dict):
            # text 또는 score 키가 있으면 박스 하나
            if any(k in obj for k in ('text', 'score', 'confidence', 'bbox', 'box')):
                results.append(obj)
                return
            for v in obj.values():
                self._extract_boxes(v, results)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._extract_boxes(item, results)

    def filter_frames(self, video_path: str, output_folder: str,
                      progress_callback=None) -> List[Tuple[float, str]]:
        """
        자막이 있는 프레임만 추출하여 output_folder에 저장.
        반환: [(timestamp, filepath), ...]
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

        filtered: List[Tuple[float, str]] = []
        frame_idx = 0
        saved_count = 0
        skipped_count = 0

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
                    max_w = 1920
                    if w > max_w:
                        scale = max_w / w
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    try:
                        ocr_res = self.ocr.ocr(frame, cls=False)
                        box_count = self._normalize_det_result(ocr_res)
                    except Exception as e:
                        self._log(f"  [Filter Error frame {frame_idx}] {str(e)}")
                        box_count = 0

                    if box_count >= self.min_boxes:
                        # 자막 있는 프레임 저장
                        filename = f"frame_{curr_t:.3f}.jpg"
                        filepath = os.path.join(output_folder, filename)
                        cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        filtered.append((curr_t, filepath))
                        saved_count += 1
                    else:
                        skipped_count += 1
        finally:
            cap.release()

        self._log(f"  [Filter Summary] checked: {frame_idx // interval_frames}, "
                  f"saved: {saved_count}, skipped: {skipped_count}, "
                  f"total_filtered: {len(filtered)}")
        return filtered
