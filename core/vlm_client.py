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
        """JSON 응답 파싱 (마크다운 코드블록, 중괄호 추출 등)"""
        if not raw_text:
            return None
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(code_block_pattern, raw_text)
        if matches:
            raw_text = matches[-1].strip()
        raw_text = raw_text.strip()
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass
        json_pattern = r'(\{[\s\S]*\})'
        json_matches = re.findall(json_pattern, raw_text)
        if json_matches:
            try:
                return json.loads(json_matches[-1])
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(raw_text)
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
        with open(filepath, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def analyze_batch(self, frame_batch: List[Tuple[float, str]],
                      custom_prompt: str = "") -> List[Dict]:
        """
        frame_batch: [(timestamp, filepath), ...] 최대 10장
        반환: [{start, end, original, translated, color, position}, ...]
        """
        if not frame_batch:
            return []

        # 프롬프트 구성
        prompt_parts = [
            "Analyze the following frames in order.",
            "Each frame is from a Japanese anime/video with hardcoded subtitles.",
            "Extract ONLY character dialogue subtitles (ignore logos, background text, sound effects, UI).",
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
            "3. Return ONLY the JSON array. No explanations, no markdown code blocks."
        ]

        if custom_prompt:
            prompt_parts.append("")
            prompt_parts.append("User custom instructions:")
            prompt_parts.append(custom_prompt)

        prompt_text = "\n".join(prompt_parts)

        # 멀티모달 content 구성
        content = [{"type": "text", "text": prompt_text}]
        for idx, (ts, filepath) in enumerate(frame_batch):
            b64 = self._encode_image(filepath)
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

            # 응답 형식 처리: {"subtitles": [...]} 또는 [...]
            if isinstance(parsed, dict) and 'subtitles' in parsed:
                subtitles = parsed['subtitles']
            elif isinstance(parsed, list):
                subtitles = parsed
            else:
                raise Exception(f"Unexpected response format. Got: {type(parsed).__name__}. Raw: {raw_content[:500]}")

            if not isinstance(subtitles, list):
                raise Exception(f"Subtitles is not a list. Got: {type(subtitles).__name__}. Raw: {raw_content[:500]}")

            # frame_index -> timestamp 변환
            results: List[Dict] = []
            for i, sub in enumerate(subtitles):
                frame_idx = sub.get('frame_index', i)
                if 0 <= frame_idx < len(frame_batch):
                    start_t = frame_batch[frame_idx][0]
                else:
                    start_t = frame_batch[0][0] if frame_batch else 0.0

                # end 계산: 다음 subtitle의 start - 0.1초, 마지막은 start + 1.5초
                if i + 1 < len(subtitles):
                    next_idx = subtitles[i + 1].get('frame_index', frame_idx + 1)
                    if 0 <= next_idx < len(frame_batch):
                        end_t = frame_batch[next_idx][0] - 0.1
                    else:
                        end_t = start_t + 1.5
                else:
                    end_t = start_t + 1.5

                # 최소 지속시간 보장
                if end_t <= start_t:
                    end_t = start_t + 1.0

                results.append({
                    "start": start_t,
                    "end": end_t,
                    "original": sub.get('original', ''),
                    "translated": sub.get('translated', ''),
                    "color": sub.get('color', '#FFFFFF'),
                    "position": sub.get('position', 'bottom-center')
                })

            return results
        except Exception as e:
            raise Exception(f"VLM Processing Error: {str(e)}") from e
