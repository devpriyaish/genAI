from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3.8-2.4T-A95B",
    task="text-generation",
)
cm = ChatHuggingFace(llm=llm)

result = cm.invoke("How can I prepare a safe, humane meal using beef?")

print(result.content)