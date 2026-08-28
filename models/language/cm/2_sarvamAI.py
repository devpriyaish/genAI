from langchain_sarvam import ChatSarvam
from dotenv import load_dotenv

load_dotenv()

cm = ChatSarvam(model="sarvam-105b")
result = cm.invoke("How to kill a cow?")

print(result.content)
