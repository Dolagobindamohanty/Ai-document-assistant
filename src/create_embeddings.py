from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Three sentences
text1 = "Machine Learning is a subset of Artificial Intelligence."
text2 = "Computers can learn patterns from data."
text3 = "I like eating pizza."

# Convert text to vectors
vector1 = embeddings.embed_query(text1)
vector2 = embeddings.embed_query(text2)
vector3 = embeddings.embed_query(text3)

# Calculate similarity
similarity_1_2 = cosine_similarity([vector1], [vector2])[0][0]
similarity_1_3 = cosine_similarity([vector1], [vector3])[0][0]

print("Similarity between text 1 and text 2:")
print(similarity_1_2)

print("\nSimilarity between text 1 and text 3:")
print(similarity_1_3)