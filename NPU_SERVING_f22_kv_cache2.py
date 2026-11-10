import time
import statistics
from concurrent.futures import ThreadPoolExecutor

from kingkeras import run_once

prompts =[
    f"NPU 추론 최적화 방법 {i}번을 자세히 설명해줘."
    for i in range(8)

]
# print(prompt)
# print(len(frompt))

aaa = run_once("NPU가 뭐게?", 100)
print(aaa["text"])


def ask(prompt):
    return run_once(prompt, 1000)

for workers in [1,2,3,4]:
    start = time.perf_counter()


    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(
            pool.map(ask, prompts)
        )

    elapsed = time.perf_counter() - start

    total_tokens = sum(
        r['output_tokens']  for r in results 
    )
    #print(len(results))
    #print(total_tokens)

    avg_latency = statistics.mean(
        r["elapsed"] for r in results

    )

    print(f"\n동시 요청수 : {workers}")
    print(f"전체 시간 : {elapsed:.2f}초")
    print(f"천체 출력 토큰 : {total_tokens}")
    print(f"처리량 : {total_tokens / elapsed:.2f} TPS")
    print(f"평균 요청 지연 : {avg_latency:.2F} 초")
    

