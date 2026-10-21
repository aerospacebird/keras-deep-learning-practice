
# tool calling : 외부 기능을 가져와서 사용하기!!
from urllib import response

from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    max_tokens=2000,
    api_key=api_key,
    base_url=base_url,
)

########################### 0. Tool 정의 ###########################
# pip install langchain_tavily
from langchain_tavily import TavilySearch

#웹 검색 도구 생성
web_search = TavilySearch(
    max_results=3,
)
# result = web_search.invoke({
#     "query": "삼성전자의 주가를 예측해줘"
# })

# print(result)  # 웹 검색 결과 출력
# print(result["results"][-1]["content"])  # 첫 번째 결과의 제목 출력

tools = [web_search]  # 도구를 리스트에 추가

llm_with_tools = llm.bind_tools(tools)  # LLM에 도구 바인딩

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END  
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 메모리 추가 첫번째
from langgraph.checkpoint.memory import InMemorySaver

########################## 1. state 정의 ########################## 
class State(TypedDict):
    messages : Annotated[list, add_messages]  # messages 필드에 add_messages 데코레이터 적용
    search_result : str
    analysis_result : str

################# 2. node 정의 ##########################
# 검색 에이전트
def search_agent(state: Stage):
    print("검색에이전트 실행")
    question = state["message"][-1].content

    result = web_search.invoke({
        "query": question
    })
    return{"search_result": str(result)}


def analysis_agent(state: State):
    print("분석 에이전트 실행")
    question = state["messages"][-1].content

    response = llm.invoke(
        f"다음 질문을 기준으로 지식을 바탕으로 분석해줘.\n {question}"
    )
    return{"analysis_result": response.content}

def merge_agent(state:State):
    result = (
        state["search_result"] + "\n\n" + 
        "================================================= "+ "\n" +
        state["analysis_result"]
    )
    return{"message": [("assistant", result)]}
    

##################################################################################################
#5. 그래프 구성

builder = StateGraph(State)
builder.add_node("search", search_agent)
builder.add_node("analysis", analysis_agent)
builder.add_node("merge", merge_agent)

############################################## 병렬 실행 ##########################################################

builder.add_edge(START, "search_agent")
builder.add_edge(START, "analysis_agent")

################################################## 병합 ########################################################
builder.add_edge("search", "merge")
builder.add_edge("analysis", "merge")
builder.add_edge("merge", "END")

# 컴파일
graph = builder.compile()
################################################ 그래프 실행 ##################################################
# 실행
query = "엔비디아의 미래 전망에 대해 알려줘"


response = graph.invoke({
    {"message": [()"}
})

print(response["final_result"])

import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO






def chatbot(state : State) :
    return {"messages" : [llm_with_tools.invoke(state["messages"])]}

##################### 3. graph, node 정의 ##########################
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools)
# graph_builder.add_node("tools", tool_node)
graph_builder.add_node("tools", ToolNode(tools))


############### 4. edge 정의 ##########################
graph_builder.add_edge(START, "chatbot")

# graph_builder.add_edge("chatbot", "tools", condition=tools_condition)
graph_builder.add_conditional_edges(
    "chatbot", 
    tools_condition #tool_calls 이 있냐?
    #tool_Calls 있으면 'tools' 반환
    #tool_Calls 없으면 'END' 반환
    #tools_condition에 End가 들어있음
    )
graph_builder.add_edge("tools", "chatbot")

################################## 5. compile #########################
# 메모리추가 두번째 
memory = InMemorySaver()

graph = graph_builder.compile(checkpointer=memory)

################ 6. graph 실행 ##########################
# 메모리 추가 세번째

config = {"configurable": {"thread_id" : "user01"}}



# query = "삼성전자의 내일 주가를 예측해줘"

# response = graph.invoke({
#     "messages": [("user", query)],
#     })

# print("AI : ", response["messages"][-1].content)  # 마지막 메시지 출력

response = graph.invoke(
    {"messages": [("user", "내이름은 이동욱이야")]},
    config=config
    
)

print(response["messages"][-1].content)

print("================== 두번째 질문 ==============================")
response = graph.invoke(
    {"messages": [('user', " 내이름은 누구게")]},
    config=config
)

print(response["messages"][-1].content)

#########################그림 그리기

import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO




# ###############################################3
# # 그래프 이미지 저장 2
# ###############################################
# #path = './_save/image/'


# import matplotlib.pyplot as plt
# from PIL import Image
# from io import BytesIO





# png_data = graph.get_graph().draw_mermaid_png()  # PNG 파일로 저장


# img = Image.open(BytesIO(png_data))
# # with open(path + 'lg07_ToolNode_graph.png', 'wb') as file:
# #     file.write(png_data)

# # file = open(path + "lg07_ToolNode.png", "wb")
# # file.write(png_data)
# # file.close()

# # 그림출력
# plt.figure(figsize=(12, 8))
# plt.imshow(img)
# plt.axis('off')
# plt.show()



