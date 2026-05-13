import cv2
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Optional, Tuple

class PaddleVLExtractor:
    """
    PaddleOCR-VL-1.5 (0.9B) 로컬 GPU 추출기.
    transformers 라이브러리를 통해 로드하고, 영상 프레임에서 텍스트를 추출합니다.
    """
    def __init__(self, model_path: str = "PaddlePaddle/PaddleOCR-VL-1.5",
                 device: Optional[str] = None, interval_sec: float = 0.3,
                 similarity_threshold: float = 0.6):
        self.model_path = model_path
        self.interval_sec = interval_sec
        self.similarity_threshold = similarity_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _load_model(self):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()

    def _extract_text_from_frame(self, pil_image: Image.Image) -> str:
        """단일 PIL 이미지에서 텍스트를 추출합니다."""
        if self._model is None:
            self._load_model()

        # PaddleOCR-VL-1.5는 텍스트 스팟팅(text spotting) 프롬프트 권장
        prompt = """<ocr_prompt>Extract and recognize all text content in the image. Return only the extracted text, with each text line separated by newline. Do not include any other content.</ocr_prompt>"""
        inputs = self._processor(
            text=prompt,
            images=pil_image.convert("RGB"),
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

        generated_text = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]
        # 프롬프트 관련 텍스트 제거
        clean_text = generated_text.replace(prompt, "").strip()
        return clean_text

    def extract(self, video_path: str, progress_callback=None) -> List[Dict]:
        """영상에서 자막을 추출하여 [{start, end, original, position}] 반환"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        interval_frames = max(1, int(fps * self.interval_sec))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        from difflib import SequenceMatcher

        results: List[Dict] = []
        buffered_text = ""
        start_time = -1.0
        frame_idx = 0
        last_timestamp = 0.0

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

                # BGR -> RGB -> PIL
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                try:
                    current_raw = self._extract_text_from_frame(pil_image)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    current_raw = ""

                # 버퍼링 / 병합 로직 (OCR 기존과 동일)
                if current_raw != buffered_text:
                    if buffered_text != "" and (
                        current_raw == "" or
                        SequenceMatcher(None, current_raw, buffered_text).ratio() < self.similarity_threshold
                    ):
                        if buffered_text.strip():
                            results.append({
                                "start": start_time,
                                "end": curr_t,
                                "original": buffered_text.strip(),
                                "position": "bottom-center"  # VL은 위치 미지원
                            })
                        start_time = curr_t if current_raw != "" else -1.0

                    buffered_text = current_raw

        cap.release()

        # 마지막 버퍼 Flush
        if buffered_text and buffered_text.strip():
            if start_time < 0:
                start_time = last_timestamp
            results.append({
                "start": start_time,
                "end": last_timestamp,
                "original": buffered_text.strip(),
                "position": "bottom-center"
            })

        return results
