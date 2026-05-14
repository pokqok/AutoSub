import base64
import json
import re
import requests
import os
from typing import List, Dict, Tuple, Any


class VLMClient:
    """
    OpenAI 호환 멀티이미지 VLM 클라이언트.
    자막 프레임 배치를 전송하여 번역/색상/위치를 추출.
    """
    def __init__(self, api_key: str, model_name: str, base_url: str):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        if not self.base_url.endswith('/v1'):
            self.base_url += '/v1'

    def _parse_json_response(self, raw_text: str) -> Any:
        """JSON 응답 파싱 (마크다운 코드블록, 중괄호/대괄호 추출 등)"""
        if not raw_text:
            return None
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(code_block_pattern, raw_text)
        if matches:
            raw_text = matches[-1].strip()
        raw_text = raw_text.strip()

        # 1. 전체를 그대로 파싱 시도
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        # 2. JSON array([...])를 먼저 찾아 시도
        array_pattern = r'(\[[\s\S]*\])'
        array_matches = re.findall(array_pattern, raw_text)
        if array_matches:
            try:
                return json.loads(array_matches[-1])
            except json.JSONDecodeError:
                pass

        # 3. JSON object({...})를 찾아 시도
        obj_pattern = r'(\{[\s\S]*\})'
        obj_matches = re.findall(obj_pattern, raw_text)
        if obj_matches:
            try:
                return json.loads(obj_matches[-1])
            except json.JSONDecodeError:
                pass

        return None

    def test_connection(self) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": "Say exactly 'OK' and nothing else."}
            ],
            "temperature": 0.0,
            "max_tokens": 10
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers, json=payload, timeout=30
            )
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text[:500]}")
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            raise Exception(f"API Connection Test Failed: {str(e)}") from e

    def _encode_image(self, filepath: str) -> str:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Frame file not found: {filepath}")
        with open(filepath, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def analyze_batch(self, frame_batch: List[Dict],
                      custom_prompt: str = "") -> List[Dict]:
        """
        frame_batch: [{"timestamp", "filepath", "bbox", "orig_w", "orig_h"}, ...] 최대 10장
        반환: [{start, end, original, translated, color, position}, ...]
        """
        if not frame_batch:
            return []

        def _bbox_to_position(bbox, orig_w, orig_h):
            """bbox 좌표로 position 문자열 반환"""
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if cy < orig_h / 3:
                v = "top"
            elif cy < 2 * orig_h / 3:
                v = "middle"
            else:
                v = "bottom"
            if cx < orig_w / 3:
                h = "left"
            elif cx < 2 * orig_w / 3:
                h = "center"
            else:
                h = "right"
            return f"{v}-{h}"

        # 프롬프트 구성 (bbox 힌트 포함)
        prompt_lines = [
            "Analyze the following cropped subtitle frames in order.",
            "Each image is a cropped region containing Japanese subtitles from an anime/video.",
            "Extract ONLY character dialogue subtitles (ignore sound effects, logos, background text).",
            "Return a JSON array of subtitle entries. Each entry must include:",
            "- frame_index: index within this batch (0-based)",
            "- original: Japanese text",
            "- translated: Korean translation",
            "- color: subtitle text color as HEX (e.g., #FFFFFF)",
            "- position: one of [top-left, top-center, top-right, middle-left, middle-center, middle-right, bottom-left, bottom-center, bottom-right]",
            "",
            "Rules:",
            "1. Do NOT duplicate consecutive identical subtitles.",
            "2. Skip frames with no Japanese subtitle.",
            "3. Return ONLY the JSON array. No explanations, no markdown code blocks.",
        ]

        if custom_prompt:
            prompt_lines.append("")
            prompt_lines.append("User custom instructions:")
            prompt_lines.append(custom_prompt)

        prompt_text = "\n".join(prompt_lines)

        # 멀티모달 content 구성
        content = [{"type": "text", "text": prompt_text}]
        for idx, item in enumerate(frame_batch):
            b64 = self._encode_image(item["filepath"])
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": 4000
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers, json=payload, timeout=180
            )
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text[:500]}")
            result = response.json()
            raw_content = result['choices'][0]['message']['content']

            parsed = self._parse_json_response(raw_content)
            if parsed is None:
                raise Exception(f"JSON parse failed. Raw response: {raw_content[:500]}")

            # 응답 형식 처리
            if isinstance(parsed, dict) and 'subtitles' in parsed:
                subtitles = parsed['subtitles']
            elif isinstance(parsed, list):
                subtitles = parsed
            else:
                raise Exception(f"Unexpected format. Got: {type(parsed).__name__}. Raw: {raw_content[:500]}")

            if not isinstance(subtitles, list):
                raise Exception(f"Subtitles is not a list. Got: {type(subtitles).__name__}. Raw: {raw_content[:500]}")

            # frame_index -> timestamp 변환
            results: List[Dict] = []
            for i, sub in enumerate(subtitles):
                frame_idx = sub.get('frame_index', i)
                if 0 <= frame_idx < len(frame_batch):
                    item = frame_batch[frame_idx]
                    start_t = item["timestamp"]
                    # position: VLM 반환값이 없으면 bbox 기반 계산
                    pos = sub.get('position')
                    if not pos and "bbox" in item:
                        pos = _bbox_to_position(
                            item["bbox"], item.get("orig_w", 1920), item.get("orig_h", 1080)
                        )
                else:
                    start_t = frame_batch[0]["timestamp"] if frame_batch else 0.0
                    pos = sub.get('position', 'bottom-center')

                # end 계산
                if i + 1 < len(subtitles):
                    next_idx = subtitles[i + 1].get('frame_index', frame_idx + 1)
                    if 0 <= next_idx < len(frame_batch):
                        end_t = frame_batch[next_idx]["timestamp"] - 0.1
                    else:
                        end_t = start_t + 1.5
                else:
                    end_t = start_t + 1.5

                if end_t <= start_t:
                    end_t = start_t + 1.0

                results.append({
                    "start": start_t,
                    "end": end_t,
                    "original": sub.get('original', ''),
                    "translated": sub.get('translated', ''),
                    "color": sub.get('color', '#FFFFFF'),
                    "position": pos or 'bottom-center'
                })

            return results
        except Exception as e:
            raise Exception(f"VLM Processing Error: {str(e)}") from e
