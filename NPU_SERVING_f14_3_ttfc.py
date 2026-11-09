# time to first chunk : ttfc 약어 입니다.

from kingkeras import get_llm

import time

llm = get_llm()

prompt = "NPU의 장점을 자세히 설명해줘?"

start = time.perf_counter()

chunk = next(iter(llm.stream(prompt)))

first_time= time.perf_counter() - start

# first_time = None




# for chunk in llm.stream(prompt):

#     if chunk.content:
#         first_time= time.perf_counter() - start
#         break


print("=================================================")
print("첫 컨텐츠 응답시간 :", round(first_time, 2), "초")
