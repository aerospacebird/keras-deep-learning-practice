
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

#1. State 정의

class State(TypedDict):
    question : str
    search_result : str
    analysis_result : str
    final_result : str


#2. 검색 노드
def search_agent(state: State):
    print("검색 에이전트 실행")
    return {"search_result": "엔디비아는 ai gpu 시장의 주요 기업입니다."}

#3. 분석노드
def analysis_agent(state : State):
    print("분석 에이전트 실행")
    return {"analysis_result" : "ai 반도체 수요 증가가 성장 요인이다."}

#4. 결과 통합 노드
def merge_agent(state: State):
    result = (
        state["search_result"] + "\n" +
        state["analysis_result"]
    )
    return{"final_result": result}

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

# 실행
response = graph.invoke({
    "question": "엔비디아의 미래 전망은?"
})

print(response["final_result"])

import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


