

from kingkeras import get_llm
from langchain_core.tools import tool

@tool   # Decorator (데코레이터) : 함수나 클래스의 동작을 변경하거나 확장하는 기능입니다.
def add(aaa: int, bbb:int) -> int:
    """ 두 정수를 더합니다."""

    return aaa + bbb


print(add.name) # 함수명
print(add.description) # add함수의 주석
print(add.args)

################################################################################################################
# @decorator	함수·클래스 기능 확장
# A @ B	행렬 곱셈
# @staticmethod	정적 메서드 선언
# @classmethod	클래스 메서드 선언
# @property	메서드를 속성처럼 사용

########################################### LLM 생성  ####################################################################
llm = get_llm(max_tokens=200)


############################################  도구 등록  ############################################################
llm_with_tools = llm.bind_tools([add])

###########################################  질문 ##################################################################
response = llm_with_tools.invoke(
            "237341과 3123321을 add로 더해줘."
)

############################################## 결과 확인 ##############################################################

# print("일반 응답:", response.content)
# print("===========================================================================")
# print("Tool Calls :", response.tool_calls)
# print("메타 데이터:", response.response_metadata)

for call in response.tool_calls:
    print("tool 이름:", call['name']) # add
    print("매개변수 :", call['args']) 
    print("호출 ID:", call['id'])

##################################################################################################################
# "tool 이름:", call['name']) # add
#     print("매개변수 :", call['args']) 
#     print("호출 ID:", call['id'