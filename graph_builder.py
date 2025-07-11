from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from config import llm, embedder
from utils import preprocess_text

def construct_graph(documents):
    from langchain_core.documents import Document
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
    from nltk.tokenize import sent_tokenize
    from config import llm, embedder
    from utils import preprocess_text
    from sklearn.metrics.pairwise import cosine_similarity

    graph = NetworkxEntityGraph()
    transformer = LLMGraphTransformer(llm=llm)

    for doc in documents:
        clean_text = preprocess_text(doc.page_content)
        graph_doc = transformer.convert_to_graph_documents([Document(page_content=clean_text)])[0]

        for node in graph_doc.nodes:
            graph._graph.add_node(node.id, type=node.type)
        for edge in graph_doc.relationships:
            graph._graph.add_edge(edge.source.id, edge.target.id, type=edge.type)

    sentences = [sent for doc in documents for sent in sent_tokenize(doc.page_content)]
    embeddings = embedder.encode(sentences)

    for i, sent in enumerate(sentences):
        graph._graph.add_node(f"sentence_{i}", type="sentence", content=sent)

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            if sim > 0.7:
                graph._graph.add_edge(f"sentence_{i}", f"sentence_{j}", type="similarity", weight=sim)

    return graph

