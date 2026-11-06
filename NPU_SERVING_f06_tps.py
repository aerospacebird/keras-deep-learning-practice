"""
LLM 연결
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 --devices npu:4 --host 0.0.0.0 --port 8000
"""

from openai import OpenAI
from langchain_openai import ChatOpenAI
import time

max_tokens = 1000

llm = ChatOpenAI(
    base_url = "http://127.0.0.1:8000/v1",
    api_key="EMPTY",
    max_tokens=max_tokens,
    temperature=0,
    top_p=0.5,
)

start = time.perf_counter() #  time.time() 이건 컴퓨터 시간으로 함 그래서 perf_counter을 사용함
response = llm.invoke("NPU와 GPU의 차이점을 설명해줘.")
end = time.perf_counter()
elapsed = round(end - start, 2) # elapsed = 경과

print("추론시간 : ", elapsed)