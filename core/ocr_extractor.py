import cv2
import numpy as np
import os
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
import re

class OCRExtractor:
    """
    PaddleOCR 기반 영상 자막 추출 클래스.
    - 메모리 내 처리(디스크 저장 없음)
    - 전체 화면에서 OCR 수행
    - 고정 시간 간격 샘플링 + SequenceMatcher 기반 버퍼링/병합
    """
    def __init__(self, interval_sec: float = 0.3, similarity_threshold: float = 0.6):
        self.interval_sec = interval_sec
        self.similarity_threshold = similarity_threshold
        self.ocr = None

    def _init_ocr(self):
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='japan',
            show_log=False
        )

    @staticmethod
    def _is_junk_text(text: str) -> bool:
        has_japanese = re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', text)
        if not has_japanese:
            return True
        if re.search(r'[a-zA-Z0-9]{10,}', text):
            return True
        return False

    def _parse_ocr_result(self, ocr_res, frame_shape) -> Tuple[str, str]:
        """OCR 결과에서 원문과 대표 position을 반환"""
        if not ocr_res or not ocr_res[0]:
            return "", ""

        lines = ocr_res[0]
        h, w = frame_shape[:2]

        # 유효 라인 필터링 (신뢰도 > 0.6, junk text 제외)
        valid_lines = []
        for l in lines:
            if not l:
                continue
            text = l[1][0]
            conf = l[1][1]
            if not self._is_junk_text(text) and conf > 0.6:
                valid_lines.append(l)

        if not valid_lines:
            return "", ""

        # 세로쓰기 판단
        v_count = 0
        for l in valid_lines:
            box = l[0]
            box_w = abs(box[1][0] - box[0][0])
            box_h = abs(box[2][1] - box[1][1])
            if box_h > box_w * 1.5:
                v_count += 1
        is_vertical = v_count > (len(valid_lines) * 0.4)

        # 정렬: 세로는 오른쪽->위쪽, 가로는 위->아래, 왼->오른
        if is_vertical:
            valid_lines.sort(key=lambda x: (-x[0][0][0], x[0][0][1]))
        else:
            valid_lines.sort(key=lambda x: (x[0][0][1], x[0][0][0]))

        full_text = "".join([l[1][0] for l in valid_lines]).strip()

        # Position 계산: 가장 아래쪽 라인의 중심을 기준
        lowest_line = max(valid_lines, key=lambda x: (x[0][0][1] + x[0][2][1]) / 2)
        box = lowest_line[0]
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

        return full_text, f"{v_pos}-{h_pos}"

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

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval_frames == 0:
                curr_t = frame_idx / fps
                last_timestamp = curr_t
                if progress_callback:
                    progress_callback(frame_idx, total_frames)

                # 이미지 전처리 + OCR
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                processed = clahe.apply(gray)
                ocr_res = self.ocr.ocr(processed, cls=True)
                current_raw, current_pos = self._parse_ocr_result(ocr_res, frame.shape)

                # 버퍼링 / 병합 로직
                if current_raw != buffered_text:
                    # 새 텍스트가 감지됨
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

            frame_idx += 1

        cap.release()

        # 마지막 버퍼 Flush
        if buffered_text and not self._is_junk_text(buffered_text):
            if start_time < 0:
                start_time = last_timestamp
            results.append({
                "start": start_time,
                "end": last_timestamp,
                "original": buffered_text,
                "position": buffered_pos
            })

        return results
