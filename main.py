import glob
import re
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# 1. Load transcripts
# ---------------------------------------------------------
raw_documents = []

for filepath in glob.glob("documents/*.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        raw_documents.append(f.read())

print("Loaded", len(raw_documents), "raw transcripts")

# ---------------------------------------------------------
# 2. Extract interviewee answers
# ---------------------------------------------------------
def extract_answers(text):
    answers = []
    # Adjust these labels if your transcripts use different names
    pattern = r"(Interviewee|Participant|Instructor|Respondent)[^:]*:\s*(.*?)(?=(Interviewer|Researcher|Moderator|Question)[^:]*:|$)"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    for m in matches:
        answer = m[1].strip()
        if len(answer) > 50:  # avoid tiny answers
            answers.append(answer)
    return answers

documents = []
for transcript in raw_documents:
    documents.extend(extract_answers(transcript))

print("Extracted", len(documents), "interviewee answers")

# ---------------------------------------------------------
# 3. Feature seeds
# ---------------------------------------------------------
feature_seeds = {
    "Automation": "automation administrative tasks grading repetitive",
    "Workflow Integration": "workflow integration LMS Canvas minimal change",
    "Curriculum Visualization": "curriculum change impact students visualization",
    "Import Export": "import export csv upload download",
    "Calculation Customization": "custom calculations modify metrics formulas",
    "UI Customization": "interface layout visuals graph type customization",
    "Longitudinal Views": "over time progression trend trajectory",
    "Alternative Access": "TA view student view permissions dashboard",
    "Context Explanations": "context explanation definitions interpretation",
    "Student Tagging": "tagging sorting categorizing labeling",
    "Comparisons": "compare performance assignments materials grades"
}

# ---------------------------------------------------------
# 4. Seed vocabulary
# ---------------------------------------------------------
seed_words = []
for phrase in feature_seeds.values():
    seed_words.extend(phrase.split())

vectorizer = CountVectorizer(vocabulary=list(set(seed_words)))

# ---------------------------------------------------------
# 5. Precompute embeddings (disables UMAP)
# ---------------------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = embedder.encode(documents)

# ---------------------------------------------------------
# 6. BERTopic without UMAP/HDBSCAN
# ---------------------------------------------------------
topic_model = BERTopic(
    embedding_model=None,          # disables UMAP
    vectorizer_model=vectorizer,
    calculate_probabilities=False,
    low_memory=True,
    min_topic_size=2,
    verbose=True
)

topics, probs = topic_model.fit_transform(documents, embeddings=document_embeddings)

# ---------------------------------------------------------
# 7. Reduce topics to something readable (10–15 topics)
# ---------------------------------------------------------
topic_model = topic_model.reduce_topics(documents, nr_topics=15)

# ---------------------------------------------------------
# 8. Build topic representations
# ---------------------------------------------------------
topic_info = topic_model.get_topic_info()

topic_representations = {}
valid_topic_ids = []

for topic_id in topic_info.Topic:
    if topic_id == -1:
        continue  # skip outlier topic
    words = topic_model.get_topic(topic_id)
    topic_representations[topic_id] = " ".join([w for w, _ in words])
    valid_topic_ids.append(topic_id)

topic_embeddings = embedder.encode(list(topic_representations.values()))
feature_embeddings = embedder.encode(list(feature_seeds.values()))

similarity_matrix = cosine_similarity(topic_embeddings, feature_embeddings)

# ---------------------------------------------------------
# 9. Map topics → predefined features
# ---------------------------------------------------------
new_labels = {}
for i, topic_id in enumerate(valid_topic_ids):
    best_feature = list(feature_seeds.keys())[np.argmax(similarity_matrix[i])]
    new_labels[topic_id] = best_feature

topic_model.set_topic_labels(new_labels)

# ---------------------------------------------------------
# 10. CLEAN PRINTED SUMMARY
# ---------------------------------------------------------
print("\n\n================ CLEAN TOPIC SUMMARY ================\n")

for topic_id in valid_topic_ids:
    label = new_labels[topic_id]
    words = topic_model.get_topic(topic_id)
    print(f"\n### {label}")
    print("Top Words:", [w for w, _ in words[:10]])

# ---------------------------------------------------------
# 11. Representative quotes per topic
# ---------------------------------------------------------
print("\n\n================ REPRESENTATIVE QUOTES ================\n")

for topic_id in valid_topic_ids:
    label = new_labels[topic_id]
    print(f"\n=== {label} ===")
    docs = topic_model.get_representative_docs(topic_id)
    for d in docs[:3]:  # top 3 quotes
        print("-", d[:300], "...\n")

# ---------------------------------------------------------
# 12. Visualizations
# ---------------------------------------------------------
try:
    fig1 = topic_model.visualize_barchart()
    fig1.show()

    fig2 = topic_model.visualize_hierarchy()
    fig2.show()

    fig3 = topic_model.visualize_topics()
    fig3.show()

except Exception as e:
    print("Visualization not supported in this environment:", e)
