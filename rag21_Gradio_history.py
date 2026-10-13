# transformer_paper_chatbot.py
# 저장된 Chroma DB -> Retriever -> Prompt -> LLM -> Gradio

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

# 07. Gradio Chatbot
def answer_invoke(message, history):
    chat_history = []

    for item in history[-10:]: #최근 10개 까지만 시현
        content = item["content"]

        if item["role"] == "user":
            chat_history.append(HumanMessage(content=content))
        elif item["role"] == "assistant":
            chat_history.append(AIMessage(content=content))

    response = rag_chain.invoke({
        "input" : message,
        "chat_history" : chat_history
    })

    return response["answer"]


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