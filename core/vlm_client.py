import requests
import json
import base64
import re
from typing import Optional, Dict, Any

class VLMClient:
    """
    Ollama Cloud 및 OpenAI 호환 API를 통한 VLM 분석 클래스
    """
    def __init__(self, api_key: str, model_name: str, base_url: str):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        if not self.base_url.endswith('/v1'):
            self.base_url += '/v1'

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _parse_json_response(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """
        모델이 반환한 다양한 형태(순수 JSON, 마크다운 코드블록 등)를 파싱합니다.
        """
        if not raw_text:
            return None

        # 1. 마크다운 코드블록 내부의 JSON 추출 시도
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(code_block_pattern, raw_text)
        if matches:
            # 여러 블록이 있으면 마지막 블록을 사용 (보통 결론이 마지막에 나옴)
            raw_text = matches[-1].strip()

        raw_text = raw_text.strip()

        # 2. JSON 파싱 시도
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        # 3. 중괄호로 감싸진 부분만 찾아서 파싱 시도 (모델이 앞뒤로 설명을 덧붙인 경우)
        json_pattern = r'(\{[\s\S]*\})'
        json_matches = re.findall(json_pattern, raw_text)
        if json_matches:
            try:
                return json.loads(json_matches[-1])
            except json.JSONDecodeError:
                pass

        return None

    def analyze_frame(self, image_path: str, custom_prompt: str = "") -> Optional[Dict[str, Any]]:
        """
        단일 프레임을 분석하여 모든 대사를 리스트 형태로 추출합니다.
        """
        base64_image = self._encode_image(image_path)
        
        core_instruction = (
            "Analyze this Japanese video frame. Identify ALL text that are character's spoken dialogues. "
            "Differentiate them from background text, logos, UI, or sound effects. "
            "You must respond ONLY with a valid JSON object. Do not include markdown formatting, explanations, or any other text. "
            "If no dialogue is found, return exactly: {\"dialogues\": []} "
            "If dialogue is found, return exactly in this format:\n"
            "{\"dialogues\": [{\"original\": \"...\", \"translated\": \"...\", \"color\": \"#RRGGBB\", \"position\": \"...\"}]}"
        )

        full_prompt = f"{custom_prompt}\n\n{core_instruction}" if custom_prompt else core_instruction

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "temperature": 0.0
            # Note: response_format 제거 — Ollama/Gemma 등에서 지원하지 않는 경우가 많음
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            raw_content = result['choices'][0]['message']['content']
            
            parsed = self._parse_json_response(raw_content)
            if parsed is None:
                print(f"VLM Warning ({image_path}): Could not parse JSON from response: {raw_content[:200]}...")
                return None
            return parsed
        except Exception as e:
            print(f"VLM Error ({image_path}): {e}")
            return None
