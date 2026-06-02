import torch
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embedding_model

def main():
    model_name = "AITeamVN/Vietnamese_Embedding"
    embedding_model = get_embedding_model(model_name)
    questions = [
        "What is the capital of France?",
        "Who is the president of the United States?",
        "What is the largest mammal?"
    ]
    answers = [
        "The capital of France is Paris.",
        "The president of the United States is Joe Biden.",
        "The largest mammal is the blue whale."
    ]

    question_embeddings = embedding_model.embed_documents(questions)
    answer_embeddings = embedding_model.embed_documents(answers)
    def compute_similarity(embedding_list_1, embedding_list_2):
        similarities = []
        for emb1 in embedding_list_1:
            for emb2 in embedding_list_2:
                similarity = sum(a * b for a, b in zip(emb1, emb2)) / (sum(a * a for a in emb1) ** 0.5 * sum(b * b for b in emb2) ** 0.5)
                similarities.append(similarity)
        return similarities
    similarity = compute_similarity(question_embeddings, answer_embeddings)
    print(similarity)
if __name__ == "__main__":
    main()
