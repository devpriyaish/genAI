from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

docs = [
	"🫘 Soy chunks, 🫘 Chickpeas and 🥣 Tofu provide us **💪 Protein**.", 
	"🍚 Rice, 🌾 Oats and 🥔 Potatoes provide us **⚡ Carbohydrates**.", 
	"🥑 Avocado, 🌱 Flaxseeds and 🥜 Walnuts provide us **🥑 Healthy Fats**."
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)
result = embeddings.embed_documents(docs)

print(result)