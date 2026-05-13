import cv2
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from difflib import SequenceMatcher
import re

class OCRExtractor:
    """
    PaddleOCR 기반 영상 자막 추출 클래스.
    """
    def __init__(self, interval_sec: float = 0.3, similarity_threshold: float = 0.6,
                 conf_threshold: float = 0.4, log_callback: Optional[Callable] = None):
        self.interval_sec = interval_sec
        self.similarity_threshold = similarity_threshold
        self.conf_threshold = conf_threshold
        self.log_callback = log_callback
        self.ocr = None

    def _init_ocr(self):
        import os
        os.environ['GLOG_minloglevel'] = '2'
        import logging
        logging.getLogger('ppocr').setLevel(logging.WARNING)
        logging.getLogger('paddle').setLevel(logging.WARNING)
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang='japan')

    def _log(self, msg: str):
        if self.log_callback:
            self.log_callback(msg)

    @staticmethod
    def _is_junk_text(text: str) -> bool:
        has_japanese = re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', text)
        if has_japanese:
            return False
        text_stripped = text.strip()
        if len(text_stripped) <= 15:
            return False
        if re.search(r'[a-zA-Z0-9]{15,}', text):
            return True
        return False

    def _safe_get_text_conf(self, line_item):
        try:
            if not line_item or not isinstance(line_item, (list, tuple)) or len(line_item) < 2:
                return None, None
            text_conf = line_item[1]
            if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                return str(text_conf[0]), float(text_conf[1])
            elif isinstance(text_conf, str):
                return text_conf, 1.0
            return None, None
        except (IndexError, TypeError, ValueError):
            return None, None

    def _safe_get_box(self, line_item):
        try:
            if not line_item or not isinstance(line_item, (list, tuple)) or len(line_item) < 1:
                return None
            box = line_item[0]
            if isinstance(box, (list, tuple)) and len(box) >= 3:
                return box
            return None
        except (IndexError, TypeError):
            return None

    def _normalize_ocr_result(self, ocr_res):
        if ocr_res is None:
            return []
        if not isinstance(ocr_res, (list, tuple)):
            return []
        if len(ocr_res) == 0:
            return []
        first = ocr_res[0]
        if first is None:
            return []
        if isinstance(first, (list, tuple)) and len(first) > 0:
            if isinstance(first[0], (list, tuple)):
                return first
            else:
                return list(ocr_res)
        return list(ocr_res)

    def _parse_ocr_result(self, ocr_res, frame_shape, frame_idx: int = 0):
        lines = self._normalize_ocr_result(ocr_res)
        h, w = frame_shape[:2]

        if frame_idx <= 5:
            raw_info = []
            for l in lines:
                text, conf = self._safe_get_text_conf(l)
                if text is not None:
                    raw_info.append(f"'{text}'({conf:.2f})")
                else:
                    raw_info.append(f"PARSE_FAIL:{str(l)[:60]}")
            self._log(f"  [OCR Debug frame {frame_idx}] raw lines ({len(lines)}): {raw_info}")

        valid_lines = []
        for l in lines:
            text, conf = self._safe_get_text_conf(l)
            if text is None or conf is None:
                continue
            if conf < self.conf_threshold:
                continue
            if self._is_junk_text(text):
                continue
            valid_lines.append(l)

        if not valid_lines:
            return "", ""

        is_vertical = False
        v_count = 0
        for l in valid_lines:
            box = self._safe_get_box(l)
            if box is None or len(box) < 3:
                continue
            try:
                box_w = abs(box[1][0] - box[0][0])
                box_h = abs(box[2][1] - box[1][1])
                if box_h > box_w * 1.5:
                    v_count += 1
            except (IndexError, TypeError):
                continue
        is_vertical = v_count > (len(valid_lines) * 0.4)

        def sort_key(x):
            box = self._safe_get_box(x)
            if box is None:
                return (0, 0)
            try:
                if is_vertical:
                    return (-box[0][0], box[0][1])
                else:
                    return (box[0][1], box[0][0])
            except (IndexError, TypeError):
                return (0, 0)

        valid_lines.sort(key=sort_key)
        full_text = "".join([self._safe_get_text_conf(l)[0] or "" for l in valid_lines]).strip()

        lowest_line = None
        lowest_y = -1
        for l in valid_lines:
            box = self._safe_get_box(l)
            if box is None:
                continue
            try:
                cy = (box[0][1] + box[2][1]) / 2
                if cy > lowest_y:
                    lowest_y = cy
                    lowest_line = l
            except (IndexError, TypeError):
                continue

        position = "bottom-center"
        if lowest_line:
            box = self._safe_get_box(lowest_line)
            try:
                cx = (box[0][0] + box[2][0]) / 2
                cy = (box[0][1] + box[2][1]) / 2
                if cy < h / 3:
                    v_pos = "top"
                elif cy < 2 * h / 3:
                    v_pos = "middle"
                else:
                    v_pos = "bottom"
                if cx < w / 3:
                    h_pos = "left"
                elif cx < 2 * w / 3:
                    h_pos = "center"
                else:
                    h_pos = "right"
                position = f"{v_pos}-{h_pos}"
            except (IndexError, TypeError):
                pass

        return full_text, position

    def extract(self, video_path: str, progress_callback=None) -> List[Dict]:
        if self.ocr is None:
            self._init_ocr()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        interval_frames = max(1, int(fps * self.interval_sec))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        results: List[Dict] = []
        buffered_text = ""
        buffered_pos = ""
        start_time = -1.0
        frame_idx = 0
        last_timestamp = 0.0
        total_raw_lines = 0
        total_valid_lines = 0

        self._log(f"  [OCR] Starting: video={video_path}, fps={fps:.1f}, interval={self.interval_sec}s")
        
        first_frame_debug = True  # 첫 프레임의 OCR raw 결과를 상세 로깅

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % interval_frames == 0:
                curr_t = frame_idx / fps
                last_timestamp = curr_t
                if progress_callback:
                    progress_callback(frame_idx, total_frames)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                processed = clahe.apply(gray)

                # 4K 영상은 OCR 처리를 위해 리사이즈 (최대 1920 너비, 비율 유지)
                h, w = processed.shape
                max_w = 1920
                if w > max_w:
                    scale = max_w / w
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # 원본 BGR 프레임을 그대로 PaddleOCR에 넘김 (grayscale 변환 내부에서 처리)
                # 또는 내부에서 처리하도록 원본 그대로
                try:
                    import traceback
                    ocr_res = self.ocr.ocr(frame)
                    if first_frame_debug and ocr_res is not None:
                        # 첫 프레임의 OCR raw 결과 형식을 상세 로깅 (원인 파악용)
                        sample = str(ocr_res)[:300]
                        self._log(f"  [OCR Sample frame {frame_idx}] type={type(ocr_res).__name__}, len={len(ocr_res) if isinstance(ocr_res, (list, tuple)) else 'N/A'}, content={sample}")
                        first_frame_debug = False
                except Exception as e:
                    tb = traceback.format_exc()
                    self._log(f"  [OCR Error frame {frame_idx}] {str(e)}")
                    self._log(f"  [OCR Traceback] {tb}")
                    ocr_res = None

                current_raw, current_pos = self._parse_ocr_result(ocr_res, frame.shape, frame_idx)

                lines = self._normalize_ocr_result(ocr_res)
                total_raw_lines += len(lines)
                if current_raw:
                    total_valid_lines += 1

                if current_raw != buffered_text:
                    if buffered_text != "" and (
                        current_raw == "" or
                        SequenceMatcher(None, current_raw, buffered_text).ratio() < self.similarity_threshold
                    ):
                        if not self._is_junk_text(buffered_text):
                            results.append({
                                "start": start_time,
                                "end": curr_t,
                                "original": buffered_text,
                                "position": buffered_pos
                            })
                        start_time = curr_t if current_raw != "" else -1.0

                    buffered_text = current_raw
                    buffered_pos = current_pos

        cap.release()

        if buffered_text and not self._is_junk_text(buffered_text):
            if start_time < 0:
                start_time = last_timestamp
            results.append({
                "start": start_time,
                "end": last_timestamp,
                "original": buffered_text,
                "position": buffered_pos
            })

        self._log(f"  [OCR Summary] frames: {frame_idx // interval_frames}, "
                  f"raw lines: {total_raw_lines}, valid: {total_valid_lines}, final: {len(results)}")
        return results
