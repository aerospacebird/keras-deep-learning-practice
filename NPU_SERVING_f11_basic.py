
from kingkeras import get_llm

llm = get_llm()

prompt = " NPU와 GPU의 차이를 말해달라?"

response = llm.invoke(prompt)
print(response.content)
