from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load pre-trained Sentence Transformers model
model_name = "roberta-base-nli-mean-tokens"
model = SentenceTransformer(model_name)


@app.route("/calculate_similarity", methods=["POST"])
def calculate_similarity_en():
    data = request.json
    student_answers = data["student_answer"]
    professor_answers = data["professor_answer"]

    # Generate embeddings for English sentences
    embeddings = model.encode([student_answers, professor_answers])

    # Compute cosine similarity between the embeddings
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1])

    similarity_scores = {"similarity_score": similarity_score.item()}

    return jsonify(similarity_scores)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
