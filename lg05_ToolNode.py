
# tool calling : 외부 기능을 가져와서 사용하기!!!
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

############################################### 0. TOOL 정의 ####################################################
#pip install langchain-tavily
from langchain_tavily import TavilySearch

# 웹 검색 도구 생성
web_search = TavilySearch(
    max_results=9,
)

# result = web_search.invoke({
#     "query": "삼성전자의 주가 알려줘"
# })

# print(result)
# print(result['results'][-1]['content'])
###################################################################################################################

tools = [web_search]  #, #, #, #,]

llm.bind_tools = llm.bind_tools(tools)
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.prebuilt import ToolNode, tools_condition
##############################################3 1. state 정의 #####################################################
class State(TypedDict):
    messages :Annotated[list, add_messages] 

###########################################3### 2. 노드 정의 #####################################################
def chatbot(state : State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

################################################# 3. graph 구성, node 추가 ########################################
# 그래프를 생성

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

# tool_node = ToolNode(tools)
# graph_builder.add_node("tools", tool_node)
graph_builder.add_node("tools", ToolNode(tools))

################################################### 4. Edge add  ###################################################
graph_builder.add_edge(START, "chatbot")

#graph_builder.add_edge("chatbot", "tools")
graph_builder.add_conditional_edges(
    "chatbot", 
    tools_condition # tool_calls 이 있나?
    # tool_calls가 있으면 " tools" 반환
    # tool_calls가 없으면 END 반환
    # tools_condition에 END 포함된 상태이다.
)
graph_builder.add_edge("tools", "chatbot")

################################################# 5. 컴파일 #########################################################
graph = graph_builder.compile()

################################################# 6. 그래프 실행 ####################################################
query = "삼성전자 주가 알려줘"

response = graph.invoke({
    "messages": [("user", query)]
})

print("AI :", response["message"][-1].content)





# from typing import TypedDict, Annotated
# from langgraph.graph import StateGraph, START, END
# from lsanggraph.graph.message import add_messages
# from langchain_core.messages import AIMessage, HumanMessage

# ################ 1. state 정의 ###################
# class State(TypedDict):
#     messages : Annotated[list, add_messages]  # messages 필드에 add_messages 데코레이터 적용
#     aaa : str
#     bbb : str
#     ccc : str


# # 위에꺼는 한마디로 아래다!
# state = {
#     "messages": [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Hello!"},
#     ],
#     "aaa": "This is a sample string."
# }

# ############### 2. node 정의 ###################
# def prepare(state : State) :
#     return {"messages" : 
#             [AIMessage(content="I am a helpful assistant.")]}


# def chatbot(state : State) :
#     return {"messages" : [llm.invoke(state["messages"])]}


# def finish(state : State) :
#     return {"messages" : [AIMessage(content="답변처리가 완료되었습니다.")]}

# ############3. graph 정의 ###################
# graph_01 = StateGraph(State)

# graph_01.add_node("prepare", prepare)
# graph_01.add_node("chatbot", chatbot)
# graph_01.add_node("finish", finish)

# ############ 4. edge 정의 ###################
# graph_01.add_edge(START, "prepare")
# graph_01.add_edge("prepare", "chatbot")
# graph_01.add_edge("chatbot", "finish")
# graph_01.add_edge("finish", END)

# ############ 5. graph compile ###################
# graph = graph_01.compile()

# ############# 6. graph 실행 ###################
# query = "삼성전자의 주가를 예측해줘"

# response = graph.invoke({
#     "messages": [("user", query)],
# })

# # print(response)
# # print(response["messages"][0])    
# print(response["messages"][1])    


# ################ 7. 결과출력 ####################
# for i, msg in enumerate(response["messages"]):
#     print(f"[{i}] {msg.type}: {msg.content}") 

