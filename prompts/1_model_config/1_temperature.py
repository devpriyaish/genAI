from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  

cm = ChatOpenAI(model="gpt-5.5", temperature=0.9)
result = cm.invoke("Provide me fruits that are high in vitamin C.")

print(result.content)