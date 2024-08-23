from nomic import embed
import numpy as np
import time
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader  # Updated import for PdfReader

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variables (if needed by the library)
api_key = os.getenv('NOMIC_API_KEY')

# Optional: Set API key if the library requires manual setting
# embed.set_api_key(api_key)  # Uncomment if necessary

# Function to load and extract text from a PDF file
def load_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks with overlap
def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

# Path to the PDF document
pdf_file_path = r"C:\Users\Responseinformatics\PycharmProjects\Speech-to-Speech\Project summary.pdf"

# Load and extract text from the PDF document
document_text = load_pdf(pdf_file_path)

# Split the document text into chunks with overlap
document_chunks = split_text(document_text, chunk_size=500, overlap=100)

# Prompt user for a query
query = input("Enter your query: ")

# Start timing
start_time = time.time()

# Generate embeddings for document chunks
document_embeddings = embed.text(
    texts=document_chunks,
    model='nomic-embed-text-v1.5',
)['embeddings']

# Generate embedding for the query
query_embedding = embed.text(
    texts=[query],
    model='nomic-embed-text-v1.5',
)['embeddings'][0]

# End timing
end_time = time.time()

# Calculate the duration
duration = end_time - start_time

# Compute similarity scores
similarities = cosine_similarity([query_embedding], document_embeddings)[0]

# Filter and rank chunks based on similarity
threshold = 0.7
filtered_chunks = [(doc, score) for doc, score in zip(document_chunks, similarities) if score >= threshold]

# If there are no results above the threshold
if not filtered_chunks:
    print("No relevant content found for the query.")
else:
    # Sort chunks by similarity score in descending order
    ranked_chunks = sorted(filtered_chunks, key=lambda x: x[1], reverse=True)

    print(f"Time taken for embedding and search: {duration:.2f} seconds")
    print("Query:", query)
    print("\nRelevant Chunks:")
    for i, (chunk, score) in enumerate(ranked_chunks, 1):
        print(f"{i}. {chunk[:500]}... (Score: {score:.4f})")  # Print the first 500 characters of each chunk
