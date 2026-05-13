from typing import List, Tuple, Dict
import datetime

class SRTGenerator:
    """
    VLM 분석 결과(시간, 텍스트)를 바탕으로 최종 SRT 자막 파일을 생성하는 클래스
    """
    @staticmethod
    def format_time(seconds: float) -> str:
        """초 단위를 SRT 시간 형식(HH:MM:SS,mmm)으로 변환"""
        td = datetime.timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int(td.microseconds / 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    def generate(self, analysis_results: List[Tuple[float, str]], output_path: str):
        """
        분석 결과를 병합하여 SRT 파일을 생성합니다.
        analysis_results: [(timestamp, translated_text), ...]
        """
        if not analysis_results:
            print("No subtitles to generate.")
            return

        merged_subtitles = []
        if not analysis_results:
            return

        # 첫 번째 자막 초기화
        curr_start, curr_text = analysis_results[0]
        curr_end = curr_start + 2.0 # 기본 지속 시간 2초 (나중에 보정됨)

        for i in range(1, len(analysis_results)):
            next_start, next_text = analysis_results[i]
            
            # 텍스트가 동일하면 지속 시간을 늘림 (병합)
            if next_text == curr_text:
                curr_end = next_start + 2.0 
            else:
                # 텍스트가 바뀌면 지금까지의 자막을 확정하고 새로 시작
                merged_subtitles.append({
                    "start": curr_start,
                    "end": next_start, # 다음 자막 시작 시점을 종료 시점으로 설정
                    "text": curr_text
                })
                curr_start, curr_text = next_start, next_text
                curr_end = next_start + 2.0

        # 마지막 자막 추가
        merged_subtitles.append({
            "start": curr_start,
            "end": curr_end,
            "text": curr_text
        })

        # SRT 파일 쓰기
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, sub in enumerate(merged_subtitles, 1):
                f.write(f"{idx}\n")
                f.write(f"{self.format_time(sub['start'])} --> {self.format_time(sub['end'])}\n")
                f.write(f"{sub['text']}\n\n")

        return output_path
