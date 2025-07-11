import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from langchain_together import ChatTogether


load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")

llm = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", api_key=api_key)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
