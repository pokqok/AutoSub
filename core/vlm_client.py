import requests
import json
import base64
from typing import Optional, Dict, Any

class VLMClient:
    """
    VLM API(DeepSeek, GLM, Gemma 등)와 연동하여 이미지에서 대사를 추출하고 번역하는 클래스
    """
    def __init__(self, api_key: str, model_name: str, base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_frame(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        단일 프레임을 분석하여 일본어 원문과 한국어 번역문을 추출합니다.
        """
        base64_image = self._encode_image(image_path)
        
        prompt = (
            "You are a professional Japanese-to-Korean translator and video analyst. "
            "Scan the entire image and identify any text that is a character's spoken dialogue. "
            "IMPORTANT DIRECTIONS:\n"
            "1. Ignore atmospheric text, background signs, or sound effects (SFX) like '쾅!', '슥'.\n"
            "2. Dialogue can be anywhere on the screen, in any font, color, or orientation (vertical/horizontal).\n"
            "3. Extract the original Japanese text and translate it into natural Korean.\n"
            "4. If no dialogue is found, return {'is_dialogue': false}.\n"
            "5. If dialogue is found, return exactly in this JSON format: "
            "{\"is_dialogue\": true, \"original\": \"일본어 원문\", \"translated\": \"한국어 번역문\"}"
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "response_format": {"type": "json_object"}, # JSON 모드 지원 모델 대상
            "temperature": 0.0 # 일관성을 위해 0 설정
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            content = result['choices'][0]['message']['content']
            return json.loads(content)
        except Exception as e:
            print(f"Error during VLM analysis for {image_path}: {e}")
            return None
