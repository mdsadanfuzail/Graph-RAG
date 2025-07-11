from config import embedder
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_context(graph, question, k=3):
    question_vec = embedder.encode([question])[0]
    sentence_nodes = [
    (n, d['content']) 
    for n, d in graph._graph.nodes(data=True) 
    if d.get('type') == 'sentence' and 'content' in d
    ]
    
    similarities = []
    for node, content in sentence_nodes:
        content_vec = embedder.encode([content])[0]
        sim = cosine_similarity([question_vec], [content_vec])[0][0]
        similarities.append((node, content, sim))
    
    similarities.sort(key=lambda x: x[2], reverse=True)
    top_k = similarities[:k]
    
    context = set()
    for node, content, _ in top_k:
        context.add(content)
        for neighbor in graph._graph.neighbors(node):
            if graph._graph.nodes[neighbor]['type'] == 'sentence':
                context.add(graph._graph.nodes[neighbor]['content'])

    return " ".join(context)
