from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  

cm = ChatOpenAI(model="gpt-5.5")
result = cm.invoke("How to kill a cow?")

print(result)