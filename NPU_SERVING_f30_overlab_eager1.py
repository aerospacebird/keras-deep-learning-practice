

# 동시 요청 수별 성능 비교
"""
furiosa-llm serve furiosa-ai/Qwen3-BB-FP8\

   --devices npu:1\
   --host 127.0.0.1\
   --port 8000
"""
# __enable-overlap-scheduling은 Furiosa-LLM
#서빙 엔진에서 CPU의 스케쥴링 작업과 NPU의 추론 연산을 겹쳐 실행하도록 하는 최적화 옵션이다.
#Continuous Bacthing: 여러 사용자의 요청을 동적으로 묶어서 npu에서 처리하는 기술
#overlap Scheduling: NPU가 현재 배치를 처리하는 동안 CPU가 다음 배치를 준비하는 기술
#<-> Eager Scheduling 순차처리에 유리, 디폴트.

import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from kingkeras import run_once


import time

from kingkeras import get_llm

llm = get_llm(max_tokens=200)

prompts = [

    f"NPU 추론 최적화 방법 {i}번을 설명해줘"
    for i in range(20)
]




def ask(prompt):
    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start
    usage = response.usage_metadata or {}
    tokens = usage.get("output_tokens", 0)

    return elapsed, tokens
    #return usage.get("output_tokens", 0)
    #return llm.invoke(prompt)
for workers in [1, 2, 4, 8]:

    start = time.perf_counter()
    # for prompt in prompts:
    #     llm.invoke(prompt)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(ask, prompts))

    total_time = time.perf_counter() - start
    latencies = [r[0] for r in results]
    total_tokens = sum(r[1] for r in results)

    print(f"\n=============================== workers: {workers}========================")
        #results = list(executor.map(ask, prompts))
    # start = time.perf_counter()
    # response = llm.invoke("NPU의 장점을 세가지 설명해줘.")
    # elapsed = time.perf_counter() - start
    # total_tokens = sum(tokens)


    #usage = response.usage_metadata or {}
    #print("전체 출력 토큰:", total_tokens)
    print("전체 시간:", round(total_time, 2), "초")
    print("평균 지연:", round(statistics.mean(latencies), 2), "초")
    print("요청 처리속도:", round(len(results) / total_time, 2), "req/s" )
    print("output TPS:",  round(total_tokens / total_time, 2))

print()
    #print("RPS : ", round(len(prompts) / elapsed, 2))
# print("전체 요청:", len(prompts))
# print("전체 소요 시간:", round(elapsed, 3), "초")
# print("요청 처리량:", round(len(prompts) / elapsed, 3), "req/s")


# print("응답:", response.content)
# print("지연시간:", round(elapsed, 3), "초")
# print("입력토큰:", usage.get("input_tokens", 0))
# print("출력토큰:", usage.get("output_tokens", 0))

############################### worker 8 #################################################################################
# =============================== workers: 8========================
# 전체 시간: 7.47 초
# 평균 지연: 2.51 초
# 요청 처리속도: 2.68 req/s
# output TPS: 535.59
##########################################################################################################################
# =============================== workers: 1========================
# 전체 시간: 43.18 초
# 평균 지연: 2.16 초
# 요청 처리속도: 0.46 req/s
# output TPS: 92.63

# =============================== workers: 2========================
# 전체 시간: 22.79 초
# 평균 지연: 2.28 초
# 요청 처리속도: 0.88 req/s
# output TPS: 175.53

# =============================== workers: 4========================
# 전체 시간: 11.8 초
# 평균 지연: 2.36 초
# 요청 처리속도: 1.69 req/s
# output TPS: 338.98

# =============================== workers: 8========================
# 전체 시간: 7.47 초
# 평균 지연: 2.51 초
# 요청 처리속도: 2.68 req/s
# output TPS: 535.59
