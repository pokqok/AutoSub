import sys
import base64
import requests
import json
import os

sys.path.append(r"E:\utils\AutoSub")
from core.vlm_client import VLMClient

vlm = VLMClient(api_key="test", model_name="gemini-3-flash-preview:cloud", base_url="http://localhost:11434/v1")

img_path = r"D:\a\ani\pigtarotaro\Pigtarotaro\.autosub_temp\frame_90.000.jpg"
text_boxes = [(100, 100, 500, 200)] # 더미 텍스트 박스. 실제로는 craft_filter가 찾은걸 줘야함.
# 로그에서 가져온 실제 텍스트 박스가 없으니, 일단 원본 이미지를 블랙 마스킹해서 테스트
# 잠깐, _encode_image_masked를 그대로 쓰면 됨.
# 근데 bbox를 알아야함.

print("Testing simple masked image reading...")
# 테스트를 위해 임의의 영역을 남기고 까맣게 칠해봄
# 프레임 90 근처 자막 위치: bottom-right (대략 x: 1200~1800, y: 800~1000)
boxes = [[1200, 800, 1800, 1000]]
b64 = vlm._encode_image_masked(img_path, boxes, 1920, 1080)

payload = {
    "model": "gemini-3-flash-preview:cloud",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe exactly what you see in this image. Is there any text? If so, what does it say? Please answer in Korean."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }
    ],
    "temperature": 0.0
}

print("Sending request to Gemini...")
res = requests.post("http://localhost:11434/v1/chat/completions", json=payload, headers={"Authorization": "Bearer test"})
print("Status:", res.status_code)
print("Response:", res.json()['choices'][0]['message']['content'])
