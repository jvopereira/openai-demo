from dotenv import load_dotenv
import os
import chromadb
from chromadb.utils import embedding_functions
from helpers.prompt import Prompt

load_dotenv()

openai_key = os.getenv("OPENAI_KEY")
prompt = Prompt()

chromadb_client = chromadb.PersistentClient(path='db/')

openai_ef = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-ada-002", api_key=openai_key)

collection = chromadb_client.get_or_create_collection(name="Students",embedding_function=openai_ef)

collection.add(
    documents = [prompt.student_info, prompt.club_info, prompt.university_info],
    metadatas = [{"source": "student info"},{"source": "club info"},{'source':'university info'}],
    ids = ["id1", "id2", "id3"]
)

results = collection.query(
    query_texts=["What is the student name?"],
    n_results=2
)

print(results)