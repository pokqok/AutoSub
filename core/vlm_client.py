import requests
import json
import base64
from typing import Optional, Dict, Any

class VLMClient:
    """
    Ollama Cloud 및 OpenAI 호환 API를 통한 VLM 분석 클래스
    """
    def __init__(self, api_key: str, model_name: str, base_url: str):
        self.api_key = api_key
        self.model_name = model_name
        # base_url이 /v1으로 끝나지 않는 경우 처리
        self.base_url = base_url.rstrip('/')
        if not self.base_url.endswith('/v1'):
            self.base_url += '/v1'

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_frame(self, image_path: str, custom_prompt: str = "") -> Optional[Dict[str, Any]]:
        """
        단일 프레임을 분석하여 일본어 원문, 한국어 번역문, 그리고 텍스트의 색상 정보를 추출합니다.
        """
        base64_image = self._encode_image(image_path)
        
        core_instruction = (
            "Analyze this Japanese video frame. Identify all text that is clearly 'spoken dialogue'. "
            "Differentiate it from background text, logos, UI, or sound effects. "
            "Return only a JSON object. If no dialogue is found, return {'is_dialogue': false}. "
            "If found, return the following JSON format:\n"
            "{\n"
            "  \"is_dialogue\": true,\n"
            "  \"original\": \"Japanese text\",\n"
            "  \"translated\": \"Korean translation\",\n"
            "  \"color\": \"HEX color code of the text (e.g. #FFFFFF)\",\n"
            "  \"position\": \"text position (e.g. bottom-center, top-left)\"\n"
            "}"
        )

        full_prompt = f"{custom_prompt}\n\nCore Task: {core_instruction}" if custom_prompt else core_instruction

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
            "response_format": {"type": "json_object"},
            "temperature": 0.0
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content']
            return json.loads(content)
        except Exception as e:
            print(f"VLM Error ({image_path}): {e}")
            return None

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content']
            return json.loads(content)
        except Exception as e:
            print(f"VLM Error ({image_path}): {e}")
            return None
