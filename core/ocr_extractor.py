import cv2
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from difflib import SequenceMatcher
import re

class OCRExtractor:
    """
    PaddleOCR 기반 영상 자막 추출 클래스.
    - 메모리 내 처리(디스크 저장 없음)
    - 전체 화면에서 OCR 수행
    - 고정 시간 간격 샘플링 + SequenceMatcher 기반 버퍼링/병합
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
        # Paddle/PaddleOCR의 C++ GLog stderr 출력 억제 (0=INFO, 1=WARNING, 2=ERROR, 3=FATAL)
        os.environ['GLOG_minloglevel'] = '2'
        # paddleocr python logging도 억제
        import logging
        logging.getLogger('ppocr').setLevel(logging.WARNING)
        logging.getLogger('paddle').setLevel(logging.WARNING)
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='japan',
        )

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

    def _safe_get_text_conf(self, line_item) -> Tuple[Optional[str], Optional[float]]:
        """PaddleOCR 버전별 반환 형식 안전하게 파싱"""
        try:
            if not line_item or not isinstance(line_item, (list, tuple)):
                return None, None
            # 신버전: [[box, (text, conf)], ...]
            # 구버전: [box, (text, conf)] 또는 [box, [text, conf]]
            if len(line_item) < 2:
                return None, None
            
            text_conf = line_item[1]
            if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                text = str(text_conf[0])
                conf = float(text_conf[1])
                return text, conf
            elif isinstance(text_conf, str):
                return text_conf, 1.0
            else:
                return None, None
        except (IndexError, TypeError, ValueError):
            return None, None

    def _safe_get_box(self, line_item) -> Optional[list]:
        """bounding box 안전하게 추출"""
        try:
            if not line_item or not isinstance(line_item, (list, tuple)) or len(line_item) < 1:
                return None
            box = line_item[0]
            if isinstance(box, (list, tuple)) and len(box) >= 3:
                return box
            return None
        except (IndexError, TypeError):
            return None

    def _normalize_ocr_result(self, ocr_res) -> list:
        """PaddleOCR 여러 버전의 반환값을 표준 형식으로 변환"""
        if ocr_res is None:
            return []
        # ocr_res가 리스트가 아니면 리스트로
        if not isinstance(ocr_res, (list, tuple)):
            return []
        # 빈 리스트
        if len(ocr_res) == 0:
            return []
        # 구버전: [None] 또는 [[box, ...], ...]
        # 신버전: [[[box, ...], ...], [...]]
        first = ocr_res[0]
        if first is None:
            return []
        # first가 리스트이고, 그 안의 첫 요소도 리스트면 신버전 2차원
        if isinstance(first, (list, tuple)) and len(first) > 0:
            if isinstance(first[0], (list, tuple)):
                return first  # 신버전: 첫 언어 결과만 사용
            else:
                return list(ocr_res)  # 구버전 1차원
        return list(ocr_res)

    def _parse_ocr_result(self, ocr_res, frame_shape, frame_idx: int = 0) -> Tuple[str, str]:
        """OCR 결과에서 원문과 대표 position을 반환"""
        lines = self._normalize_ocr_result(ocr_res)
        h, w = frame_shape[:2]

        # 원시 라인 정보 로깅
        if frame_idx <= 5:
            raw_info = []
            for l in lines:
                text, conf = self._safe_get_text_conf(l)
                if text is not None:
                    raw_info.append(f"'{text}'({conf:.2f})")
            self._log(f"  [OCR Debug frame {frame_idx}] raw lines: {raw_info}")

        # 유효 라인 필터링
        valid_lines = []
        filtered_reasons = []
        for l in lines:
            text, conf = self._safe_get_text_conf(l)
            if text is None:
                filtered_reasons.append(f"parse_error: {l}")
                continue
            if conf < self.conf_threshold:
                filtered_reasons.append(f"low_conf: '{text}'({conf:.2f})")
                continue
            if self._is_junk_text(text):
                filtered_reasons.append(f"junk: '{text}'")
                continue
            valid_lines.append(l)

        if frame_idx <= 5 and filtered_reasons:
            self._log(f"  [OCR Debug frame {frame_idx}] filtered: {filtered_reasons}")

        if not valid_lines:
            return "", ""

        # 세로쓰기 판단
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

        # 정렬
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

        # Position 계산: 가장 아래쪽 라인의 중심
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
                position = "bottom-center"
        else:
            position = "bottom-center"

        return full_text, position

    def extract(self, video_path: str, progress_callback=None) -> List[Dict]:
        """영상에서 자막을 추출하여 [{start, end, original, position}, ...] 반환"""
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
