from kingkeras import get_llm

llm = get_llm(max_tokens=500)

prompt = "인공지능의 발전과정을 설명해줘."

for chunk in llm.stream(prompt):
    print(chunk.content, end="", flush=True)