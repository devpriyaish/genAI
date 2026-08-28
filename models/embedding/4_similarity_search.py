from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

docs = [
	"🫘 Soy chunks, 🫘 Chickpeas and 🥣 Tofu provide us **💪 Protein**.", 
	"🍚 Rice, 🌾 Oats and 🥔 Potatoes provide us **⚡ Carbohydrates**.", 
	"🥑 Avocado, 🌱 Flaxseeds and 🥜 Walnuts provide us **🥑 Healthy Fats**.", 
	"🫘 Lentils, 🥦 Broccoli and 🍎 Apples provide us **🌾 Fiber**.", 
	"🥛 Fortified soy milk, 🌱 Sesame seeds and 🥬 Kale provide us **🦴 Calcium**.", 
	"🫘 Lentils, 🌱 Pumpkin seeds and 🥬 Spinach provide us **🩸 Iron**.", 
	"🌱 Chia seeds, 🌱 Flaxseeds and 🥜 Walnuts provide us **🧠 Omega-3**.", 
	"🌱 Pumpkin seeds, 🥜 Almonds and 🫘 Black beans provide us **⚙️ Magnesium**.", 
	"🍌 Bananas, 🥔 Potatoes and 🫘 Beans provide us **🔋 Potassium**.", 
  "🫘 Lentils, 🫘 Kidney beans (Rajma) and 🌱 Peanuts provide us 💪 Protein."
	"🧂 Iodized salt, 🌊 Nori and 🌊 Wakame provide us **🦋 Iodine**.", 
	"🥛 Fortified plant milk, 🥣 Fortified nutritional yeast and 💊 B12 supplements provide us **💊 Vitamin B12**.", 
	"🥛 Vitamin-D-fortified plant milk, 🍄 UV-exposed mushrooms and 💊 Vitamin D supplements provide us **☀️ Vitamin D**."
]

query = "What are the sources of protein?"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embed_docs = embeddings.embed_documents(docs)
embed_query = embeddings.embed_query(query)

scores = cosine_similarity([embed_query], embed_docs)

enumerated_scores = list(enumerate(scores[0]))
sorted_scores = sorted(enumerated_scores, key=lambda x: x[1], reverse=True)

print("Scores for each document:", sorted_scores)

best_match_index = sorted_scores[0][0]
score = sorted_scores[0][1]
best_match_doc = docs[best_match_index]

print(f"Query: {query}")
print(f"Best match: {best_match_doc}")
print(f"Score: {score}")