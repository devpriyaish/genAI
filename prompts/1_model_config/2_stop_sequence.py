from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  

cm = ChatOpenAI(model="gpt-3.5-turbo", stop_sequences=["\n"])
result = cm.invoke("Write 4 lines of a poem about the sun.")

print(result.content)