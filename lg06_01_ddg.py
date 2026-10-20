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

# ########################### 0. Tool 정의 ###########################
# # pip install langchain_tavily
# from langchain_tavily import TavilySearch

# #웹 검색 도구 생성
# web_search = TavilySearch(
#     max_results=3,
# )

#pip install ddgs
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun

web_search = DuckDuckGoSearchRun(
    max_results=5,
)

result = web_search.invoke({
    "query": "삼성전자의 주가를 예측해줘"
})

print(result)  # 웹 검색 결과 출력
# # print(result["results"][-1]["content"])  # 첫 번째 결과의 제목 출력

# exit()

tools = [web_search]  # 도구를 리스트에 추가

llm_with_tools = llm.bind_tools(tools)  # LLM에 도구 바인딩

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END  
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

########################## 1. state 정의 ########################## 
class State(TypedDict):
    messages : Annotated[list, add_messages]  # messages 필드에 add_messages 데코레이터 적용

################# 2. node 정의 ##########################
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
graph_builder.add_conditional_edge(
    "chatbot", 
    tools_condition #tool_calls 이 있냐?
    #tool_Calls 있으면 'tools' 반환
    #tool_Calls 없으면 'END' 반환
    #tools_condition에 End가 들어있음
    )
graph_builder.add_edge("tools", "chatbot")

################################## 5. compile #########################
graph = graph_builder.compile()

################ 6. graph 실행 ##########################
query = "삼성전자의 내일 주가를 예측해줘"

response = graph.invoke({
    "messages": [("user", query)],
    })

print("AI : ", response["messages"][-1].content)  # 마지막 메시지 출력