# ============================================================
# 회사 PDF RAG + LCEL + Chroma
# ============================================================


# ============================================================
# 1. 라이브러리
# ============================================================

import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma


# ============================================================
# 2. 환경변수
# ============================================================

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()

base_url = "https://monogpt.kr/api/monorouter/v1/"


# ============================================================
# 3. PDF 파일
# ============================================================

pdf_path = "./_data/company_documents/company_manual.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(
        f"PDF 파일을 찾을 수 없습니다.\n"
        f"확인할 경로 : {os.path.abspath(pdf_path)}"
    )

print("=" * 80)
print("PDF 파일")
print(os.path.abspath(pdf_path))


# ============================================================
# 4. PDF 로딩
# ============================================================

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print("=" * 80)
print("PDF 페이지 수 :", len(documents))


# ============================================================
# 5. PDF metadata 보강
# ============================================================

pdf_filename = os.path.basename(pdf_path)

for doc in documents:

    # PDF 파일명 저장
    doc.metadata["source_file"] = pdf_filename

    # PyPDFLoader의 page는 0부터 시작
    page = doc.metadata.get("page")

    if isinstance(page, int):
        doc.metadata["page_number"] = page + 1
    else:
        doc.metadata["page_number"] = "알 수 없음"


# ============================================================
# 6. PDF 문서 분할
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

split_documents = text_splitter.split_documents(documents)

print("분할된 문서 수 :", len(split_documents))


# ============================================================
# 7. Embedding 모델
# ============================================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url
)

print("=" * 80)
print("Embedding 모델 준비 완료")


# ============================================================
# 8. Vector DB
# ============================================================

vector_db_path = "./_data/company_vector_db"

vectorstore = Chroma.from_documents(
    documents=split_documents,
    embedding=embeddings,
    collection_name="company_manual",
    persist_directory=vector_db_path
)

print("=" * 80)
print("Vector DB 생성 완료")
print("Vector DB :", vector_db_path)


# ============================================================
# 9. Retriever
# ============================================================

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5
    }
)

print("Retriever 준비 완료")


# ============================================================
# 10. LLM
# ============================================================

model = ChatOpenAI(
    model="gpt-5.6-terra",
    temperature=0,
    api_key=api_key,
    base_url=base_url
)

print("LLM 준비 완료")


# ============================================================
# 11. RAG Prompt
# ============================================================

template = """
당신은 회사 내부 문서를 전문적으로 분석하는 AI Assistant입니다.

아래 [회사 문서]에 포함된 내용을 근거로 사용자의 질문에 답변하세요.

반드시 다음 규칙을 지키세요.

1. 회사 문서의 내용을 가장 우선적으로 사용합니다.

2. 회사 문서에 없는 내용을 사실처럼 만들어내지 않습니다.

3. 회사 문서에서 답을 찾을 수 없는 경우에는
   반드시 다음과 같이 답변합니다.

   "제공된 회사 문서에서 해당 내용을 확인할 수 없습니다."

4. 답변은 명확하고 구체적으로 작성합니다.

5. 답변에 사용한 문서의 파일명과 페이지 번호를 표시합니다.

6. 여러 문서 또는 여러 페이지가 근거가 되는 경우
   각각의 출처를 표시합니다.

7. 문서의 내용과 일반적인 상식을 혼합하지 않습니다.

------------------------------------------------------------
[회사 문서]
------------------------------------------------------------

{context}

------------------------------------------------------------
[사용자 질문]
------------------------------------------------------------

{question}

------------------------------------------------------------
[답변]
------------------------------------------------------------
"""


prompt = PromptTemplate.from_template(template)


# ============================================================
# 12. Output Parser
# ============================================================

output_parser = StrOutputParser()


# ============================================================
# 13. LCEL Chain
# ============================================================

chain = prompt | model | output_parser


# ============================================================
# 14. 사용자 질문
# ============================================================

question = "회사에서 출장비를 어떻게 지급하나요?"


# ============================================================
# 15. 질문 → Retriever → 관련 PDF 검색
# ============================================================

print("=" * 80)
print("질문 :", question)
print("PDF 검색 중...")


docs = retriever.invoke(question)


print("검색된 문서 수 :", len(docs))


# ============================================================
# 16. 검색 결과가 없는 경우
# ============================================================

if not docs:

    context = "관련 회사 문서를 찾을 수 없습니다."

else:

    context_list = []

    for doc in docs:

        page_number = doc.metadata.get(
            "page_number",
            "알 수 없음"
        )

        source_file = doc.metadata.get(
            "source_file",
            "알 수 없음"
        )

        page_content = doc.page_content.strip()

        context_list.append(
            f"""
[출처]
파일명 : {source_file}
페이지 : {page_number}

[문서 내용]
{page_content}
"""
        )

    context = "\n\n".join(context_list)


# ============================================================
# 17. LCEL 실행
# ============================================================

print("=" * 80)
print("LLM 분석 중...")

response = chain.invoke(
    {
        "context": context,
        "question": question
    }
)


# ============================================================
# 18. 최종 답변 출력
# ============================================================

print()
print("=" * 80)
print("최종 답변")
print("=" * 80)

print(response)


# ============================================================
# 19. 검색된 원문 출력
# ============================================================

print()
print("=" * 80)
print("검색된 회사 문서")
print("=" * 80)


for i, doc in enumerate(docs, 1):

    source_file = doc.metadata.get(
        "source_file",
        "알 수 없음"
    )

    page_number = doc.metadata.get(
        "page_number",
        "알 수 없음"
    )

    print()
    print(f"[검색 결과 {i}]")
    print("-" * 80)
    print("파일명 :", source_file)
    print("페이지 :", page_number)
    print("-" * 80)

    print(doc.page_content[:1000])