from kingkeras import get_llm
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
# pip install langchain


@tool
def add(a:int, b:int)-> int:
    """ 두 정수를 더합니다."""
    return a+b

@tool
def subtract(a:int, b:int)-> int:
    """ 두 정수를 뺄셈을 합니다."""
    return a-b

@tool

def multiply(a:int, b:int)-> int:
    """ 두 정수를 곱합니다."""
    return a*b

@tool
def divide(a:int, b:int)-> float:
    """ 첫번째 정수를 두번째 정수를 나눕니다."""
    return a/b

llm = get_llm(max_tokens=500)

tools = [add, multiply, subtract, divide]

agent = create_agent(
    model = llm,
    tools = tools, #[add, subtract, multiply, divide],
    system_prompt= "계산이 필요하면 도구를 사용하세요."
)

questions =[
    "70에서 35를 뺀후 ",
    "그 결과에 3을 곱하고",
    "마지막으로 10을 더해줘",
    "12와 8을 곱해줘",
    "100을 4로 나눠줘",
    "15와 27을 더해줘"
]





result = agent.invoke({
    "messages": [
        {"role":"user", "content": "questions"}
        ]
    },
    config={"recursion_limit": 20}


print("=========================================================================")
print(result)
print("=========================================================================")

from langchain_core.messages import AIMessage
for i, msg in enumerate(result["messages"]):
    print(f"{i+1}번째 메시지")
    print("타입:", type(msg).__name__)
    print("내용:", msg.content)

    if isinstance(msg, AIMessage) and msg.tool_calls:
        for call in msg.tool_calls:
            print("도구 이름:", call['name'])
            print("전달 인자:", call['args'])



print("==========================================================================")
print("최종 답변:", result["messages"][-1].content)