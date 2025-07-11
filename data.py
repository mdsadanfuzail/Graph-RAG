import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PDF_DIR = BASE_DIR / "data"

def load_pdfs_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            loader = PyMuPDFLoader(filepath)
            docs = loader.load()
            documents.extend(docs)
    return documents

corpus = load_pdfs_from_folder(PDF_DIR)

questions_answers = [
    ("What is the key aspects of Finance?", "Investment Management: Strategies for wealth growth and asset allocation. Risk Management: Tools to mitigate financial uncertainties. Fintech: Technology-driven innovations like mobile banking and blockchain. Regulatory Compliance: Ensuring adherence to financial laws and standards."),
    ("What is the benefits of sports?", "Promotes physical and mental health. Fosters teamwork and leadership skills. Boosts community engagement and economic activity"),
    ("Give an overview on healthcare?", "Healthcare encompasses the prevention, diagnosis, treatment, and management of diseases, aiming to improve human health and well-being. It integrates medical expertise, technology, and policy to deliver care across diverse populations."),
    ("what is the applications of Artificial Intelligence?", "AI powers autonomous vehicles, virtual assistants, predictive analytics, and personalized recommendations, impacting sectors like healthcare, finance, and transportation."),
    ("Summarize the concept of Articficial Intelligence.", "Artificial Intelligence (AI) involves creating systems capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving. Key subfields include machine learning (enabling data-driven predictions), natural language processing (understanding human language), and computer vision (interpreting visual data), while ethical considerations like fairness and bias mitigation remain crucial. AI enhances efficiency through automation, improves decision-making with data-driven insights, and enables personalized experiences in healthcare, retail, and other sectors. Its applications span autonomous vehicles, virtual assistants, predictive analytics, and recommendation systems, revolutionizing industries like finance, transportation, and healthcare."),
]
