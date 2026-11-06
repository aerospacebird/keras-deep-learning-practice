
"""
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8
"""
from openai import OpenAI

client = OpenAI(

    base_url = "http://127.0.0.1:8000/v1",
    api_key="EMPTY"

)

models = client.models.list()
model_id = models.data[0].id

print("모델 :", model_id)

#추론
response = client.chat.completions.create(

    model = model_id,
    messages= [

        {
            "role": "user",
            "content": "npu와 gpu의 차이점을 한국말로 대답해줘"

        }               
               
               ],
        max_tokens=1000
)
print(response.choices[0].message.content)


