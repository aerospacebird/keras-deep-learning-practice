
from a_kingkeras import get_llm


llm = get_llm(
    max_tokens=1000,
    temperature=0.9,
    top_p=1.0,

)

response = llm.invoke("npu와 gpu의 차이점을 말해죠?")