# Graph RAG System

## Problem Statement

Traditional Retrieval-Augmented Generation (RAG) systems often rely on flat document chunk retrieval, which can miss deeper semantic and relational connections between concepts. This project implements a Graph-Based Retrieval-Augmented Generation (Graph-RAG) system that constructs a knowledge graph from a corpus of documents (PDFs), retrieves relevant information using graph traversal and semantic similarity, and generates answers using a powerful LLM.

## Technologies Used

- **LangChain** – For building document loaders, graph transformers, and LLM chains  
- **TogetherAI (LLaMA 3)** – LLM backend for answer generation  
- **Sentence Transformers (MiniLM)** – For semantic embedding and cosine similarity  
- **NetworkX** – For graph construction and traversal  
- **PyMuPDF** – For extracting text from PDFs  
- **Scikit-learn** – For similarity computations  
- **NLTK** – For sentence and word tokenization  

## Directory Structure

```
project/
│
├── main.py                     # Entry point of the project
├── config.py                   # API key setup and model initialization
├── data.py                     # PDF loader and Q&A pairs
├── graph_builder.py            # Constructs entity and sentence-based graph
├── retriever.py                # Retrieves top-k relevant sentences from the graph
├── generator.py                # Generates answers using the LLM
├── evaluator.py                # Evaluates predictions using EM and F1
├── utils.py                    # Preprocessing and scoring utilities
├── data/                       # Folder containing your 5 source PDFs
│   ├── AI.pdf
│   ├── Automobile.pdf
│   ├── finance.pdf
│   ├── Healthcare.pdf
│   ├── Sports.pdf
│
├── requirements.txt            # List of required Python packages
└── README.md                   # Project documentation
```

## How to Run

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the script**:

   ```bash
   python main.py
   ```

## Example Results

- Question: "Summarize the concept of Articficial Intelligence."
  - Answer: Artificial Intelligence (AI) refers to the development of systems that can perform tasks that typically require human intelligence, such as learning, reasoning, and problem-solving, enabling automation and data-driven decision-making across various industries.
