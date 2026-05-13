import requests
import json
import re
from typing import List, Dict, Any

class LLMClient:
    """
    텍스트 기반 번역/정제 LLM 클라이언트 (OpenAI 호환 API)
    """
    def __init__(self, api_key: str, model_name: str, base_url: str):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        if not self.base_url.endswith('/v1'):
            self.base_url += '/v1'

    def _parse_json_response(self, raw_text: str) -> Any:
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
            # 혹시 바로 list
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

    def translate_and_refine(self, ocr_results: List[Dict], custom_prompt: str = "") -> List[Dict]:
        system_prompt = (
            "당신은 일본어 애니메이션 및 영상의 전문 자막 번역가 겸 편집자입니다.\n"
            "아래 JSON 목록은 OCR 엔진이 비디오에서 대략적으로 추출한 일본어 대사와 시간대, 위치 정보입니다.\n"
            "OCR은 완벽하지 않으므로 다음 작업을 수행하세요:\n\n"
            "1. 병합: 한 문장이 OCR 실수로 두세 덩어리로 나뉘었다면, 문맥상 하나의 자연스러운 문장으로 합치세요.\n"
            "2. 노이즈 제거: 로마자, 'Next', copyright, 제작사명, UI 텍스트 등 캐릭터 대화가 아닌 것은 제거하세요.\n"
            "3. 번역: 일본어 원문을 자연스러운 한국어로 번역하되, 캐릭터 말투와 상황을 반영하세요. 너무 직역하지 마세요.\n"
            "4. 시간 교정: start/end가 매끄럽지 않거나 잘렸다면, 대화의 호흡에 맞게 자연스럽게 조정하세요. 원본 타이밍을 지나치게 벗어나지 마세요.\n"
            "5. 위치: 각 대사의 position 필드를 보존하세요.\n\n"
            "출력은 반드시 아래 JSON 배열만 사용하세요. 설명, 마크다운, 코드 블록, 다른 텍스트는 절대 포함하지 마세요.\n"
            '[{"start": 0.0, "end": 2.345, "original": "...", "translated": "...", "position": "bottom-center"}]'
        )
        if custom_prompt:
            system_prompt += f"\n\n사용자 추가 지시사항:\n{custom_prompt}"

        user_content = json.dumps(ocr_results, ensure_ascii=False, indent=2)

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
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
                headers=headers, json=payload, timeout=120
            )
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text[:500]}")
            result = response.json()
            raw_content = result['choices'][0]['message']['content']

            parsed = self._parse_json_response(raw_content)
            if parsed is None:
                raise Exception(f"JSON parse failed. Raw response: {raw_content[:500]}")
            if not isinstance(parsed, list):
                raise Exception(f"Response is not a JSON list. Got: {type(parsed).__name__}. Raw: {raw_content[:500]}")
            return parsed
        except Exception as e:
            raise Exception(f"LLM Processing Error: {str(e)}") from e
