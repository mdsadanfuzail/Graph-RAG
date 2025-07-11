from data import corpus, questions_answers
from graph_builder import construct_graph
from retriever import retrieve_context
from generator import generate_answer
from evaluator import evaluate

def main():
    graph = construct_graph(corpus)
    predictions = []

    for question, _ in questions_answers:
        context = retrieve_context(graph, question)
        answer = generate_answer(question, context)
        predictions.append(answer)
        print(f"Q: {question}\nA: {answer}\n")

    references = [ans for _, ans in questions_answers]
    em, f1 = evaluate(predictions, references)
    print(f"Exact Match: {em:.2f} | F1 Score: {f1:.2f}")

if __name__ == "__main__":
    main()
