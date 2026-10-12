
# ============================================================
# CRAZY AI: Transformer Paper RAG + General LLM + Web Search
#
# A. RAG: Transformer 논문 Chroma DB 검색
# B. LLM: 일반적인 질문에 대한 독립적인 답변
# C. Web: 질문에 맞는 Google 검색 URL 생성
#
# A, B, C 세 경로를 ThreadPoolExecutor로 병렬 실행
# ============================================================


# ============================================================
# 01. 라이브러리
# ============================================================

import os
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)
from langchain_classic.chains import create_retrieval_chain

from langchain_huggingface.embeddings import HuggingFaceEmbeddings


# ============================================================
# 02. 환경 변수
# ============================================================

load_dotenv()

api_key = os.getenv("MONOROUTER_API_KEY", "").strip()

if not api_key:
    raise RuntimeError(
        "MONOROUTER_API_KEY가 설정되지 않았습니다. "
        ".env 파일의 API 키를 확인하세요."
    )

base_url = "https://monogpt.kr/api/monorouter/v1"


# ============================================================
# 03. 임베딩 모델
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={
        "device": "cpu",
        # "local_files_only": True,
    }
)


# ============================================================
# 04. Chroma 벡터 데이터베이스
# ============================================================

# 기존에 정상 작동하던 데이터베이스 경로 유지
DB_PATH = "./_db/Transformer_Paper/"
COLLECTION_NAME = "attention_is_all_you_need"

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name=COLLECTION_NAME,
)

document_count = vector_store._collection.count()

print("=" * 65)
print("CRAZY AI RAG SYSTEM")
print("=" * 65)
print(f"Chroma DB 경로: {DB_PATH}")
print(f"컬렉션 이름: {COLLECTION_NAME}")
print(f"벡터 저장소 문서 수: {document_count}")

if document_count == 0:
    print(
        "경고: 컬렉션에 문서가 없습니다. "
        "Transformer 논문 데이터가 저장되어 있는지 확인하세요."
    )


# ============================================================
# 05. Retriever
# ============================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)


# ============================================================
# 06. LLM
# ============================================================

model = ChatOpenAI(
    model="gpt-5-nano-chat",
    temperature=0,
    max_tokens=1500,
    api_key=api_key,
    base_url=base_url,
)


# ============================================================
# 07. Transformer 논문 RAG Prompt
# ============================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
당신은 'Attention Is All You Need' 논문을 설명하는 AI 도우미입니다.

다음 컨텍스트는 논문에서 검색된 내용입니다.
반드시 컨텍스트를 우선 근거로 질문에 답변해 주세요.

규칙:
1. 답변은 한국어로 쉽게 설명해 주세요.
2. 중요한 용어는 영어 원문도 함께 표시해 주세요.
3. 답변의 근거 뒤에는 [논문 N페이지] 형식으로 페이지를 표시해 주세요.
4. 서로 다른 페이지의 내용을 사용했다면 각각 페이지를 표시해 주세요.
5. 컨텍스트에 관련 정보가 없다면
   "주어진 논문 정보로는 답변할 수 없습니다."라고 답변해 주세요.
6. 논문 문장을 그대로 길게 복사하지 말고 이해하기 쉽게 요약해 주세요.
7. 검색 문서에 페이지 정보가 없으면 페이지 번호를 추측하지 마세요.

컨텍스트:
{context}
        """
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


# ============================================================
# 08. 검색 문서 포맷
# ============================================================

# 기존 DB의 문서 메타데이터에 page_num이 있어야 합니다.
# 이 필드가 실제로 존재하지 않으면 해당 문서 검색 시
# document_prompt에서 오류가 발생할 수 있습니다.

document_prompt = PromptTemplate.from_template(
    """
[논문 {page_num}페이지]

{page_content}
    """
)


# ============================================================
# 09. RAG Chain
# ============================================================

docu_chain = create_stuff_documents_chain(
    model,
    prompt,
    document_prompt=document_prompt,
)

rag_chain = create_retrieval_chain(
    retriever,
    docu_chain,
)


# ============================================================
# 10. 일반 LLM Chain
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

general_chain = general_prompt | model | StrOutputParser()


# ============================================================
# 11. Google Web Search
# ============================================================

def web_search_chain(question: str) -> str:
    """
    Google 검색 URL을 생성합니다.
    Google Search API 키는 필요하지 않습니다.
    실제 검색은 링크를 클릭할 때 브라우저에서 수행됩니다.
    """

    query = quote_plus(question)

    google_url = (
        "https://www.google.com/search?q="
        + query
    )

    return (
        "## C. Google 웹 검색\n\n"
        f"**검색어:** {question}\n\n"
        f"[현재 질문을 Google에서 검색하기]({google_url})"
    )


# ============================================================
# 12. Gradio 대화 이력 변환
# ============================================================

def convert_history(history):
    """
    Gradio ChatInterface의 메시지 딕셔너리를
    LangChain 메시지 객체로 변환합니다.

    지원 형식:
    {"role": "user", "content": "..."}
    {"role": "assistant", "content": "..."}
    """

    chat_history = []

    for item in history or []:

        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        # 텍스트 이력만 처리합니다.
        if not isinstance(content, str):
            continue

        if not content.strip():
            continue

        if role == "user":
            chat_history.append(
                HumanMessage(content=content)
            )

        elif role == "assistant":
            chat_history.append(
                AIMessage(content=content)
            )

    return chat_history


# ============================================================
# 13. RAG 실행 함수
# ============================================================

def run_rag(question, chat_history):
    """Transformer 논문 검색 및 RAG 답변."""

    try:
        result = rag_chain.invoke({
            "input": question,
            "chat_history": chat_history,
        })

        return result["answer"]

    except Exception as e:
        return (
            "RAG 실행 오류\n\n"
            f"오류 유형: {type(e).__name__}\n\n"
            f"오류 내용: {e}"
        )


# ============================================================
# 14. 일반 LLM 실행 함수
# ============================================================


def run_general_llm(question, chat_history):
    """일반 LLM 호출 및 상세 오류 진단."""

    print("\n" + "=" * 60)
    print("[LLM] 호출 시작")
    print("[LLM] 질문:", question)
    print("[LLM] 모델:", "gpt-5-nano-chat")
    print("=" * 60)

    try:
        # 체인을 실행하고 결과 확인
        result = general_chain.invoke({
            "input": question,
            "chat_history": chat_history,
        })

        # 결과 타입 및 내용 확인
        print("[LLM] 결과 타입:", type(result).__name__)
        print("[LLM] 결과 길이:", len(result) if result else 0)
        print("[LLM] 결과 내용:", repr(result))

        if not result or not str(result).strip():
            return (
                "LLM API 호출은 완료됐지만 답변이 비어 있습니다. "
                "터미널 로그를 확인하세요."
            )

        return str(result)

    except Exception as e:
        import traceback

        error_detail = traceback.format_exc()

        print("[LLM] 실행 실패!")
        print(error_detail)

        return (
            "### 일반 LLM 실행 실패\n\n"
            f"- 오류 유형: `{type(e).__name__}`\n"
            f"- 오류 내용: `{str(e)}`\n\n"
            "터미널에 출력된 상세 오류 로그를 확인하세요."
        )



# ============================================================
# 15. RAG + LLM + Web 병렬 실행
# ============================================================

def answer_invoke(message, history):

    # --------------------------------------------------------
    # 15-1. 입력값 검증
    # --------------------------------------------------------

    question = (message or "").strip()

    if not question:
        return "질문을 입력해 주세요."

    # --------------------------------------------------------
    # 15-2. 대화 이력 변환
    # --------------------------------------------------------

    chat_history = convert_history(history)

    # --------------------------------------------------------
    # 15-3. 세 작업 병렬 실행
    # --------------------------------------------------------

    with ThreadPoolExecutor(max_workers=3) as executor:

        # A. 논문 RAG
        rag_future = executor.submit(
            run_rag,
            question,
            chat_history,
        )

        # B. 일반 LLM
        llm_future = executor.submit(
            run_general_llm,
            question,
            chat_history,
        )

        # C. Google 검색 URL
        web_future = executor.submit(
            web_search_chain,
            question,
        )

        # ----------------------------------------------------
        # 15-4. 결과 수집
        # ----------------------------------------------------

        try:
            rag_answer = rag_future.result()

        except Exception as e:
            rag_answer = (
                f"RAG 결과 수집 오류: {type(e).__name__}: {e}"
            )

        try:
            llm_answer = llm_future.result()

        except Exception as e:
            llm_answer = (
                f"일반 LLM 결과 수집 오류: {type(e).__name__}: {e}"
            )

        try:
            web_answer = web_future.result()

        except Exception as e:
            web_answer = (
                f"Google 검색 링크 생성 오류: "
                f"{type(e).__name__}: {e}"
            )

    # --------------------------------------------------------
    # 15-5. 세 경로 결과 통합
    # --------------------------------------------------------

    
    # --------------------------------------------------------
    # 최종 결과 조립: 세 경로의 결과를 명시적으로 표시
    # --------------------------------------------------------

    final_answer = (
        "# CRAZY AI 통합 답변\n\n"
        "## A. 논문 기반 RAG 답변\n\n"
        + str(rag_answer)
        + "\n\n---\n\n"
        "## B. 일반 LLM 답변\n\n"
        + str(llm_answer)
        + "\n\n---\n\n"
        + str(web_answer)
    )

    print("\n[통합 응답] LLM 답변 포함 여부:",
          bool(llm_answer and str(llm_answer).strip()))

    return final_answer



# ============================================================
# 16. Gradio ChatInterface
# ============================================================

demo = gr.ChatInterface(
    fn=answer_invoke,
    title="CRAZY AI — Transformer RAG + LLM + Google Search",
    description="""
세 가지 답변 경로를 동시에 실행합니다.

**A. RAG:** Transformer 논문 벡터 DB 기반 답변

**B. 일반 LLM:** 논문 검색 없이 독립적으로 생성하는 답변

**C. Google Web Search:** 질문에 맞는 Google 검색 링크

예시:
- Self-Attention이 무엇인가요?
- Multi-Head Attention을 설명해 주세요.
- Positional Encoding은 왜 필요한가요?
- Transformer와 RNN의 차이는 무엇인가요?
- 관련 정보를 Google에서 검색해 주세요.
""",
)

# ============================================================
# 17. 실행
# ============================================================

if __name__ == "__main__":

    demo.queue().launch(
        share=True,
    )
