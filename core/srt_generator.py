import os
import datetime
from typing import List, Tuple

class SRTGenerator:
    @staticmethod
    def format_time(seconds: float) -> str:
        td = datetime.timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int(td.microseconds / 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    def generate(self, analysis_results: List[Tuple[float, str]], output_path: str):
        if not analysis_results: return

        merged = []
        curr_start, curr_text = analysis_results[0]
        
        for i in range(1, len(analysis_results)):
            next_start, next_text = analysis_results[i]
            if next_text == curr_text:
                continue # 텍스트가 같으면 계속 유지 (지속 시간 확장)
            else:
                merged.append({"start": curr_start, "end": next_start, "text": curr_text})
                curr_start, curr_text = next_start, next_text
        
        merged.append({"start": curr_start, "end": curr_start + 2.0, "text": curr_text})

        with open(output_path, "w", encoding="utf-8") as f:
            for idx, sub in enumerate(merged, 1):
                f.write(f"{idx}\n{self.format_time(sub['start'])} --> {self.format_time(sub['end'])}\n{sub['text']}\n\n")
