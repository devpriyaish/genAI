from langchain_huggingface import HuggingFaceEmbeddings

docs = [
	"🫘 Soy chunks, 🫘 Chickpeas and 🥣 Tofu provide us **💪 Protein**.", 
	"🍚 Rice, 🌾 Oats and 🥔 Potatoes provide us **⚡ Carbohydrates**.", 
	"🥑 Avocado, 🌱 Flaxseeds and 🥜 Walnuts provide us **🥑 Healthy Fats**."
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
result = embeddings.embed_documents(docs)

print(result)