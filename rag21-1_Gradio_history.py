
import os
import gradio as gr

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from concurrent.futures import ThreadPoolExecutor
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

# 01. Embedding
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name = "Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={
        "device" : "cpu",
        # "local_files_only" : True
    }
)

# 02. 저장된 Chroma DB 불러오기
DB_PATH = "./_db/Transformer_Paper/"
COLLECTION_NAME = "attention_is_all_you_need"

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name=COLLECTION_NAME,
)

print("벡터 저장소 문서 수 :", vector_store._collection.count())

# 03. Retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)

# 04. LLM
model = ChatOpenAI(
    model="gpt-5-nano-chat",
    temperature=0,
    max_tokens=1500,
    api_key=api_key,
    base_url=base_url,
)

# 05. Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system","""
    당신은 'Attention Is All You Need' 논문을 설명하는 AI 도우미입니다.

    다음 컨텍스트는 논문에서 검색된 내용입니다.
    반드시 컨텍스트를 우선 근거로 질문에 답변해 주세요.

    규칙:
    1. 답변은 한국어로 쉽게 설명해 주세요.
    2. 중요한 용어는 영어 원문도 함께 표시해 주세요.
    3. 답변의 근거 뒤에는 반드시 [논문 N페이지] 형식으로 페이지를 표시해 주세요.
    4. 서로 다른 페이지의 내용을 사용했다면 각각 페이지를 표시해 주세요.
    5. 컨텍스트에 관련 정보가 없다면
    \"주어진 논문 정보로는 답변할 수 없습니다.\"라고 답변해 주세요.
    6. 논문 문장을 그대로 길게 복사하지 말고 이해하기 쉽게 요약해서 설명해 주세요.

    컨텍스트:{context}

"""),
    MessagesPlaceholder(variable_name='chat_history'),
    ("human", "{input}")
])

# 각 chunk의 page_num metadata를 실제 context에 포함시킨다.
document_prompt = PromptTemplate.from_template("""
[논문 {page_num}페이지]

{page_content}
""")

# 06. RAG Chain
docu_chain = create_stuff_documents_chain(
    model,
    prompt,
    document_prompt=document_prompt,
)#prompt | model

rag_chain = create_retrieval_chain(
    retriever,
    docu_chain,
)# 검색 | docu_chain
##############################################################################################################

# ============================================================
# 06-1. 일반 LLM Chain 추가
# 기존 model을 재사용하므로 별도의 모델 생성이 필요하지 않음
# ============================================================

general_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        당신은 일반적인 질문에 답변하는 AI 도우미입니다.

        1. 한국어로 명확하게 답변하세요.
        2. 질문에 필요한 개념과 예제를 설명하세요.
        3. Chroma DB나 논문 검색 결과는 사용하지 않습니다.
        4. 논문 페이지 번호를 임의로 생성하지 마세요.
        """
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# 기존 model을 재사용
general_chain = general_prompt | model | StrOutputParser()


#############################################################################################################


# ============================================================
# 07. RAG + 일반 LLM 병렬 실행
# ============================================================

def answer_invoke(message, history):
    chat_history = []

    # 기존 Gradio 대화 기록 변환 로직 재사용
    for item in history:
        content = item["content"]

        if item["role"] == "user":
            chat_history.append(HumanMessage(content=content))

        elif item["role"] == "assistant":
            chat_history.append(AIMessage(content=content))

    # --------------------------------------------------------
    # 두 Chain을 병렬 실행
    # --------------------------------------------------------

    with ThreadPoolExecutor(max_workers=2) as executor:

        # 경로 A: 기존 RAG Chain
        rag_future = executor.submit(
            rag_chain.invoke,
            {
                "input": message,
                "chat_history": chat_history
            }
        )

        # 경로 B: 독립적인 일반 LLM Chain
        llm_future = executor.submit(
            general_chain.invoke,
            {
                "input": message,
                "chat_history": chat_history
            }
        )

        # 두 결과 수집
        try:
            rag_result = rag_future.result()
            rag_answer = rag_result["answer"]

        except Exception as e:
            rag_answer = (
                f"RAG 실행 오류: {type(e).__name__}: {e}"
            )

        try:
            llm_answer = llm_future.result()

        except Exception as e:
            llm_answer = (
                f"일반 LLM 실행 오류: {type(e).__name__}: {e}"
            )

    # --------------------------------------------------------
    # 두 답변 통합
    # --------------------------------------------------------

    return (
        "## A. 논문 기반 RAG 답변\n\n"
        + rag_answer
        + "\n\n---\n\n"
        + "## B. 일반 LLM 답변\n\n"
        + llm_answer
    )



demo = gr.ChatInterface(
    fn=answer_invoke,
    title="Attention Is All You Need 논문 챗봇",
    description="""
Transformer 논문을 기반으로 질문할 수 있습니다.

예)
- Self-Attention이 무엇인가요?
- Multi-Head Attention을 설명해 주세요.
- Positional Encoding은 왜 필요한가요?
- Transformer와 RNN의 차이는 무엇인가요?
- 근거 페이지와 함께 설명해 주세요.
""",
)

# 08. 실행
demo.launch(share=True)


"""
1.안녕
history = [
    {"role" : "user", "content" : "안녕"}
    {"role" : "assistant" , "content" : "안녕하세요"}
]
2.나는 윤영선이다
history = [
    {"role" : "user", "content" : "안녕"}
    {"role" : "assistant" , "content" : "안녕하세요"}
    {"role" : "user", "content" : "나는 이동욱이다."}
    {"role" : "assistant" , "content" : "당신은 이동욱이군요"}
]
3. 내가 누구게

"""

# # ============================================================
# # CRAZY AI Transformer Paper Chatbot V2.0
# #
# # 기능:
# # 1. 저장된 Chroma Vector DB 연결
# # 2. Qwen3 Embedding을 이용한 논문 검색
# # 3. RAG 기반 논문 답변
# # 4. 독립적인 일반 LLM 답변
# # 5. RAG와 일반 LLM의 병렬 실행
# # 6. 두 답변의 실행 시간 측정
# # 7. 예외 발생 시 개별 경로 오류 처리
# # 8. Gradio ChatInterface 제공
# #
# # 실행 파일:
# # transformer_paper_chatbot.py
# # ============================================================

# import os
# import time
# import logging
# from concurrent.futures import ThreadPoolExecutor

# import gradio as gr

# from dotenv import load_dotenv

# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_chroma import Chroma

# from langchain_core.prompts import (
#     ChatPromptTemplate,
#     PromptTemplate,
#     MessagesPlaceholder,
# )

# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.output_parsers import StrOutputParser

# from langchain_classic.chains.combine_documents import (
#     create_stuff_documents_chain
# )
# from langchain_classic.chains import create_retrieval_chain

# from langchain_huggingface.embeddings import HuggingFaceEmbeddings


# # ============================================================
# # 00. 기본 설정
# # ============================================================

# load_dotenv()

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s"
# )

# api_key = os.environ.get("MONOROUTER_API_KEY", "").strip()

# if not api_key:
#     raise ValueError(
#         "MONOROUTER_API_KEY가 없습니다. .env 파일 또는 환경 변수를 확인하세요."
#     )

# # API Base URL
# base_url = "https://monogpt.kr/api/monorouter/v1"

# # LLM 설정
# MODEL_NAME = "gpt-5-nano-chat"
# MAX_TOKENS = 1500

# # Chroma DB 설정
# DB_PATH = "./_db/Transformer_Paper/"
# COLLECTION_NAME = "attention_is_all_you_need"


# # ============================================================
# # 01. Embedding 모델
# # ============================================================

# # 기존에 저장된 Chroma DB를 검색하기 위해
# # DB 생성 당시와 호환되는 Embedding 모델을 사용해야 합니다.
# #
# # 최초 실행 시 모델 파일 다운로드가 필요할 수 있습니다.
# # CPU에서 실행하므로 임베딩 생성 속도는 하드웨어에 따라 달라집니다.

# embeddings = HuggingFaceEmbeddings(
#     model_name="Qwen/Qwen3-Embedding-0.6B",
#     model_kwargs={
#         "device": "cpu",
#     },
# )


# # ============================================================
# # 02. 저장된 Chroma Vector DB 연결
# # ============================================================

# if not os.path.isdir(DB_PATH):
#     raise FileNotFoundError(
#         f"Chroma DB 경로를 찾을 수 없습니다: {DB_PATH}"
#     )

# vector_store = Chroma(
#     embedding_function=embeddings,
#     persist_directory=DB_PATH,
#     collection_name=COLLECTION_NAME,
# )

# document_count = vector_store._collection.count()

# if document_count == 0:
#     raise ValueError(
#         "Chroma DB에 문서가 없습니다. "
#         "DB 경로와 collection 이름을 확인하세요."
#     )

# logging.info("Chroma DB 연결 완료")
# logging.info("저장된 문서 수: %s", document_count)


# # ============================================================
# # 03. Retriever
# # ============================================================

# retriever = vector_store.as_retriever(
#     search_kwargs={"k": 4}
# )


# # ============================================================
# # 04. LLM 모델 생성
# # ============================================================

# # 경로 A와 경로 B는 서로 다른 Chain으로 구성됩니다.
# #
# # 아래 예제에서는 동일한 API 모델을 사용하지만,
# # 각각 독립적으로 호출하므로 서로 다른 답변을 생성합니다.
# #
# # 필요하다면 GENERAL_MODEL_NAME을 지원되는 다른 모델명으로
# # 변경하여 모델 간 비교 실험을 진행할 수 있습니다.

# rag_model = ChatOpenAI(
#     model=MODEL_NAME,
#     temperature=0,
#     max_tokens=MAX_TOKENS,
#     api_key=api_key,
#     base_url=base_url,
# )

# general_model = ChatOpenAI(
#     model=MODEL_NAME,
#     temperature=0.3,
#     max_tokens=MAX_TOKENS,
#     api_key=api_key,
#     base_url=base_url,
# )


# # ============================================================
# # 05. RAG Prompt
# # ============================================================

# rag_prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         """
# 당신은 'Attention Is All You Need' 논문을 설명하는
# AI 연구 도우미입니다.

# [답변 규칙]

# 1. 논문에서 검색된 컨텍스트를 우선적인 근거로 사용하세요.
# 2. 한국어로 명확하고 이해하기 쉽게 설명하세요.
# 3. 중요한 전문 용어는 영어 원문을 함께 표시하세요.
# 4. 논문 내용에 관한 주장에는 실제 컨텍스트에 표시된
#    페이지 정보를 근거로 [논문 N페이지] 형식으로 인용하세요.
# 5. 서로 다른 페이지의 내용을 사용했다면 각 근거에
#    해당하는 페이지를 구분하여 표시하세요.
# 6. 컨텍스트에 관련 정보가 없다면
#    "주어진 논문 정보로는 답변할 수 없습니다."
#    라고 답변하세요.
# 7. 페이지 번호를 추측하거나 만들어내지 마세요.
# 8. 논문 문장을 길게 그대로 복사하지 말고 요약하세요.
# 9. 검색된 컨텍스트에 없는 논문 내용을
#    검색 결과에 있는 것처럼 주장하지 마세요.

# [검색된 논문 컨텍스트]

# {context}
# """
#     ),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{input}"),
# ])


# # ============================================================
# # 06. 문서 Prompt
# # ============================================================

# # Chroma DB의 각 문서에 저장된 page_num metadata를
# # LLM이 읽을 수 있도록 컨텍스트에 포함합니다.
# #
# # 주의:
# # 실제 metadata 키가 page_num인지 확인해야 합니다.
# # PDF 페이지를 0부터 저장했다면 사용자에게 표시할 때
# # 1을 더해야 하는지도 DB 구축 코드를 확인하세요.

# document_prompt = PromptTemplate.from_template(
#     """
# [논문 {page_num}페이지]

# {page_content}
# """
# )


# # ============================================================
# # 07. RAG Chain 구성
# # ============================================================

# # 문서 검색 결과를 Prompt에 넣고 LLM이 답변을 생성합니다.

# document_chain = create_stuff_documents_chain(
#     llm=rag_model,
#     prompt=rag_prompt,
#     document_prompt=document_prompt,
# )

# # Retriever + Document Chain
# rag_chain = create_retrieval_chain(
#     retriever,
#     document_chain,
# )

# logging.info("RAG Chain 구성 완료")


# # ============================================================
# # 08. 독립적인 일반 LLM Chain 구성
# # ============================================================

# # 일반 LLM 경로에서는 Chroma DB를 검색하지 않습니다.
# # 검색 컨텍스트 없이 사용자의 질문과 대화 기록으로 답변합니다.

# general_prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         """
# 당신은 일반적인 질문에 답변하는 AI 도우미입니다.

# 규칙:
# 1. 한국어로 명확하고 정확하게 답변하세요.
# 2. 필요한 경우 예시와 기술적인 설명을 제공하세요.
# 3. 확실하지 않은 정보는 불확실하다고 명시하세요.
# 4. 논문을 직접 검색하지 않았으므로
#    논문 페이지 번호나 검색 근거를 임의로 만들지 마세요.
# 5. 논문에 관한 질문이라도 별도의 논문 검색 결과가
#    제공되지 않았다는 점을 고려하여 답변하세요.
# """
#     ),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{input}"),
# ])

# # Prompt -> LLM -> 문자열 출력
# general_chain = (
#     general_prompt
#     | general_model
#     | StrOutputParser()
# )

# logging.info("General LLM Chain 구성 완료")


# # ============================================================
# # 09. Gradio 대화 기록 변환
# # ============================================================

# def convert_history(history):
#     """
#     Gradio의 대화 기록을 LangChain 메시지로 변환합니다.

#     Gradio messages 형식 예:
#     [
#         {"role": "user", "content": "안녕"},
#         {"role": "assistant", "content": "안녕하세요"}
#     ]

#     반환:
#         [HumanMessage(...), AIMessage(...)]
#     """

#     chat_history = []

#     if not history:
#         return chat_history

#     for item in history:
#         # 현재 코드는 Gradio messages 형식을 사용합니다.
#         if not isinstance(item, dict):
#             continue

#         role = item.get("role")
#         content = item.get("content", "")

#         # 텍스트 메시지만 대화 기록에 추가합니다.
#         # 이미지 또는 기타 멀티모달 메시지는 여기서 제외합니다.
#         if not isinstance(content, str) or not content.strip():
#             continue

#         if role == "user":
#             chat_history.append(
#                 HumanMessage(content=content)
#             )

#         elif role == "assistant":
#             chat_history.append(
#                 AIMessage(content=content)
#             )

#     return chat_history


# # ============================================================
# # 10. RAG 실행 함수
# # ============================================================

# def run_rag(message, chat_history):
#     """
#     논문 검색 기반 RAG를 실행합니다.

#     반환:
#         answer: 답변 문자열
#         elapsed: 실행 시간(초)
#     """

#     start_time = time.perf_counter()

#     result = rag_chain.invoke({
#         "input": message,
#         "chat_history": chat_history,
#     })

#     elapsed = time.perf_counter() - start_time

#     return result["answer"], elapsed


# # ============================================================
# # 11. 일반 LLM 실행 함수
# # ============================================================

# def run_general_llm(message, chat_history):
#     """
#     Vector DB를 사용하지 않는 일반 LLM을 실행합니다.

#     반환:
#         answer: 답변 문자열
#         elapsed: 실행 시간(초)
#     """

#     start_time = time.perf_counter()

#     answer = general_chain.invoke({
#         "input": message,
#         "chat_history": chat_history,
#     })

#     elapsed = time.perf_counter() - start_time

#     return answer, elapsed


# # ============================================================
# # 12. 병렬 실행 함수
# # ============================================================

# def parallel_answer(message, history):
#     """
#     RAG와 일반 LLM을 동시에 실행합니다.

#     실행 흐름:
#         사용자 질문
#              |
#              +---- RAG: Chroma 검색 + RAG LLM
#              |
#              +---- General LLM: 독립적인 LLM
#              |
#              +---- 결과 통합

#     한 경로에서 오류가 발생하더라도
#     다른 경로의 결과는 최대한 유지합니다.
#     """

#     if not isinstance(message, str) or not message.strip():
#         return "질문을 입력해 주세요."

#     chat_history = convert_history(history)

#     # 각 Chain은 동일한 사용자 질문과 대화 기록을 받습니다.
#     rag_input = {
#         "message": message,
#         "chat_history": chat_history,
#     }

#     llm_input = {
#         "message": message,
#         "chat_history": chat_history,
#     }

#     # 기본 오류 메시지
#     rag_answer = None
#     llm_answer = None

#     rag_elapsed = None
#     llm_elapsed = None

#     # 두 경로를 동시에 실행합니다.
#     with ThreadPoolExecutor(max_workers=2) as executor:

#         rag_future = executor.submit(
#             run_rag,
#             rag_input["message"],
#             rag_input["chat_history"],
#         )

#         llm_future = executor.submit(
#             run_general_llm,
#             llm_input["message"],
#             llm_input["chat_history"],
#         )

#         # RAG 결과 수집
#         try:
#             rag_answer, rag_elapsed = rag_future.result()

#         except Exception as exc:
#             logging.exception("RAG 실행 오류")
#             rag_answer = (
#                 "RAG 답변 생성 중 오류가 발생했습니다.\n\n"
#                 f"오류 유형: {type(exc).__name__}"
#             )

#         # 일반 LLM 결과 수집
#         try:
#             llm_answer, llm_elapsed = llm_future.result()

#         except Exception as exc:
#             logging.exception("일반 LLM 실행 오류")
#             llm_answer = (
#                 "일반 LLM 답변 생성 중 오류가 발생했습니다.\n\n"
#                 f"오류 유형: {type(exc).__name__}"
#             )

#     # 실행 시간 표시
#     rag_time_text = (
#         f"{rag_elapsed:.2f}초"
#         if rag_elapsed is not None
#         else "측정 실패"
#     )

#     llm_time_text = (
#         f"{llm_elapsed:.2f}초"
#         if llm_elapsed is not None
#         else "측정 실패"
#     )

#     # 두 답변을 구분하여 출력합니다.
#     output = f"""
# # CRAZY AI Parallel RAG Chatbot

# ## A. 논문 기반 RAG 답변

# **실행 시간:** {rag_time_text}

# {rag_answer}

# ---

# ## B. 독립적인 일반 LLM 답변

# **실행 시간:** {llm_time_text}

# {llm_answer}

# ---

# ## C. 실행 정보

# - RAG: Chroma DB 검색 후 LLM 답변 생성
# - General LLM: Vector DB 검색 없이 답변 생성
# - 실행 방식: 두 경로 병렬 호출
# - 출력 방식: 두 결과를 모두 수집한 후 통합 출력

# **참고:** 두 답변은 독립적으로 생성되며,
# 일반 LLM 답변은 논문 근거 검증을 거치지 않았습니다.
# """

#     return output


# # ============================================================
# # 13. Gradio ChatInterface
# # ============================================================

# demo = gr.ChatInterface(
#     fn=parallel_answer,
#     title="CRAZY AI - Transformer Paper Parallel Chatbot",
#     description="""
# Attention Is All You Need 논문 기반 RAG와
# 독립적인 일반 LLM을 병렬 실행합니다.

# 예시 질문:
# - Self-Attention이 무엇인가요?
# - Multi-Head Attention을 설명해 주세요.
# - Positional Encoding은 왜 필요한가요?
# - Transformer와 RNN의 차이는 무엇인가요?
# - 논문에서 Scaled Dot-Product Attention은
#   어떻게 정의되어 있나요?

# 답변 A: Chroma DB 검색 결과를 이용한 논문 기반 답변
# 답변 B: 독립적인 일반 LLM 답변
# """,
#     type="messages",
# )

# # ============================================================
# # 14. 실행
# # ============================================================

# if __name__ == "__main__":
#     demo.launch(
#         share=True,
#         show_error=True,
#     )


# transformer_paper_chatbot.py
# 저장된 Chroma DB -> Retriever -> Prompt -> LLM -> Gradio

# import os
# import gradio as gr

# from dotenv import load_dotenv
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_chroma import Chroma
# from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain_classic.chains import create_retrieval_chain
# from langchain_core.prompts import MessagesPlaceholder
# from langchain_core.messages import HumanMessage, AIMessage
# from concurrent.futures import ThreadPoolExecutor
# from langchain_core.output_parsers import StrOutputParser

# load_dotenv()

# api_key = os.environ["MONOROUTER_API_KEY"].strip()
# base_url = "https://monogpt.kr/api/monorouter/v1"

# # 01. Embedding
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(
#     model_name = "Qwen/Qwen3-Embedding-0.6B",
#     model_kwargs={
#         "device" : "cpu",
#         # "local_files_only" : True
#     }
# )

# # 02. 저장된 Chroma DB 불러오기
# DB_PATH = "./_db/Transformer_Paper/"
# COLLECTION_NAME = "attention_is_all_you_need"

# vector_store = Chroma(
#     embedding_function=embeddings,
#     persist_directory=DB_PATH,
#     collection_name=COLLECTION_NAME,
# )

# print("벡터 저장소 문서 수 :", vector_store._collection.count())

# # 03. Retriever
# retriever = vector_store.as_retriever(
#     search_kwargs={"k": 4}
# )

# # 04. LLM
# model = ChatOpenAI(
#     model="gpt-5-nano-chat",
#     temperature=0,
#     max_tokens=1500,
#     api_key=api_key,
#     base_url=base_url,
# )

# # 05. Prompt
# prompt = ChatPromptTemplate.from_messages([
#     ("system","""
#     당신은 'Attention Is All You Need' 논문을 설명하는 AI 도우미입니다.

#     다음 컨텍스트는 논문에서 검색된 내용입니다.
#     반드시 컨텍스트를 우선 근거로 질문에 답변해 주세요.

#     규칙:
#     1. 답변은 한국어로 쉽게 설명해 주세요.
#     2. 중요한 용어는 영어 원문도 함께 표시해 주세요.
#     3. 답변의 근거 뒤에는 반드시 [논문 N페이지] 형식으로 페이지를 표시해 주세요.
#     4. 서로 다른 페이지의 내용을 사용했다면 각각 페이지를 표시해 주세요.
#     5. 컨텍스트에 관련 정보가 없다면
#     \"주어진 논문 정보로는 답변할 수 없습니다.\"라고 답변해 주세요.
#     6. 논문 문장을 그대로 길게 복사하지 말고 이해하기 쉽게 요약해서 설명해 주세요.

#     컨텍스트:{context}

# """),
#     MessagesPlaceholder(variable_name='chat_history'),
#     ("human", "{input}")
# ])

# # 각 chunk의 page_num metadata를 실제 context에 포함시킨다.
# document_prompt = PromptTemplate.from_template("""
# [논문 {page_num}페이지]

# {page_content}
# """)

# # 06. RAG Chain
# docu_chain = create_stuff_documents_chain(
#     model,
#     prompt,
#     document_prompt=document_prompt,
# )#prompt | model

# rag_chain = create_retrieval_chain(
#     retriever,
#     docu_chain,
# )# 검색 | docu_chain
# ##############################################################################################################

# # ============================================================
# # 06-1. 일반 LLM Chain 추가
# # 기존 model을 재사용하므로 별도의 모델 생성이 필요하지 않음
# # ============================================================

# general_prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         """
#         당신은 일반적인 질문에 답변하는 AI 도우미입니다.

#         1. 한국어로 명확하게 답변하세요.
#         2. 질문에 필요한 개념과 예제를 설명하세요.
#         3. Chroma DB나 논문 검색 결과는 사용하지 않습니다.
#         4. 논문 페이지 번호를 임의로 생성하지 마세요.
#         """
#     ),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{input}")
# ])

# # 기존 model을 재사용
# general_chain = general_prompt | model | StrOutputParser()


# #############################################################################################################









# # ============================================================
# # 07. RAG + 일반 LLM 병렬 실행
# # ============================================================

# def answer_invoke(message, history):
#     chat_history = []

#     # 기존 Gradio 대화 기록 변환 로직 재사용
#     for item in history:
#         content = item["content"]

#         if item["role"] == "user":
#             chat_history.append(HumanMessage(content=content))

#         elif item["role"] == "assistant":
#             chat_history.append(AIMessage(content=content))

#     # --------------------------------------------------------
#     # 두 Chain을 병렬 실행
#     # --------------------------------------------------------

#     with ThreadPoolExecutor(max_workers=2) as executor:

#         # 경로 A: 기존 RAG Chain
#         rag_future = executor.submit(
#             rag_chain.invoke,
#             {
#                 "input": message,
#                 "chat_history": chat_history
#             }
#         )

#         # 경로 B: 독립적인 일반 LLM Chain
#         llm_future = executor.submit(
#             general_chain.invoke,
#             {
#                 "input": message,
#                 "chat_history": chat_history
#             }
#         )

#         # 두 결과 수집
#         try:
#             rag_result = rag_future.result()
#             rag_answer = rag_result["answer"]

#         except Exception as e:
#             rag_answer = (
#                 f"RAG 실행 오류: {type(e).__name__}: {e}"
#             )

#         try:
#             llm_answer = llm_future.result()

#         except Exception as e:
#             llm_answer = (
#                 f"일반 LLM 실행 오류: {type(e).__name__}: {e}"
#             )

#     # --------------------------------------------------------
#     # 두 답변 통합
#     # --------------------------------------------------------

#     return (
#         "## A. 논문 기반 RAG 답변\n\n"
#         + rag_answer
#         + "\n\n---\n\n"
#         + "## B. 일반 LLM 답변\n\n"
#         + llm_answer
#     )



# demo = gr.ChatInterface(
#     fn=answer_invoke,
#     title="Attention Is All You Need 논문 챗봇",
#     description="""
# Transformer 논문을 기반으로 질문할 수 있습니다.

# 예)
# - Self-Attention이 무엇인가요?
# - Multi-Head Attention을 설명해 주세요.
# - Positional Encoding은 왜 필요한가요?
# - Transformer와 RNN의 차이는 무엇인가요?
# - 근거 페이지와 함께 설명해 주세요.
# """,
# )

# # 08. 실행
# demo.launch(share=True)


# """
# 1.안녕
# history = [
#     {"role" : "user", "content" : "안녕"}
#     {"role" : "assistant" , "content" : "안녕하세요"}
# ]
# 2.나는 윤영선이다
# history = [
#     {"role" : "user", "content" : "안녕"}
#     {"role" : "assistant" , "content" : "안녕하세요"}
#     {"role" : "user", "content" : "나는 이동욱이다."}
#     {"role" : "assistant" , "content" : "당신은 이동욱이군요"}
# ]
# 3. 내가 누구게

# """