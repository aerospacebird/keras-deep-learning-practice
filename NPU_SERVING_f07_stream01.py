from openai import OpenAI

client = OpenAI(
    base_url = "http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)

# 서버에 로드도니 모델 확인
model = client.models.list().data[0].id

#  추론
streaming = client.chat.completions.create(
    model = model,
    messages = [
        {
            "role" : "user",
            "content" : "한국에 국가를 알어?",
        }
    ],
    max_tokens=3000,
    stream=True,
)

for chunk in streaming:
    if chunk.choices:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)