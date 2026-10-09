"""Similarity and redundancy analysis.

Similarity is a continuous text-relatedness score. Redundancy is a separate, contextual
decision that should be confirmed by a person. The optional supervised model learns only
from human-reviewed task pairs.
"""
from __future__ import annotations

import re
from typing import Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report

TEXT_FIELDS = ["task_name", "description"]

def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()

def task_text(task: dict[str, Any] | pd.Series) -> str:
    name = clean_text(task.get("task_name", ""))
    desc = clean_text(task.get("description", ""))
    category = clean_text(task.get("category", ""))
    inputs = clean_text(task.get("inputs", ""))
    outputs = clean_text(task.get("outputs", ""))
    # Weight name and description more heavily by repeating them.
    return f"{name}. {name}. {desc}. {desc}. Category: {category}. Inputs: {inputs}. Outputs: {outputs}"

class SimilarityEngine:
    def __init__(self, method: str = "Sentence Transformers"):
        self.method = method
        self.model = None
        self._sentence_model = None
        self.last_backend = None

    def _load_sentence_model(self):
        if self._sentence_model is None:
            from sentence_transformers import SentenceTransformer
            self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._sentence_model

    def similarity_matrix(self, tasks: pd.DataFrame) -> np.ndarray:
        texts = [task_text(row) for _, row in tasks.iterrows()]
        if not texts:
            return np.zeros((0, 0))
        if self.method == "Sentence Transformers":
            try:
                model = self._load_sentence_model()
                vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                self.last_backend = "Sentence Transformers (all-MiniLM-L6-v2)"
                return np.clip(np.asarray(vectors) @ np.asarray(vectors).T, -1, 1)
            except Exception as exc:
                # Deliberate fallback makes the app usable without model downloads/network.
                self.last_backend = f"TF-IDF fallback (Sentence Transformers unavailable: {type(exc).__name__})"
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        try:
            vectors = vectorizer.fit_transform(texts)
            matrix = cosine_similarity(vectors)
        except ValueError:
            matrix = np.eye(len(texts))
        self.last_backend = self.last_backend or "TF-IDF"
        return matrix

def pair_features(a: pd.Series | dict, b: pd.Series | dict, similarity: float) -> dict[str, float]:
    def val(obj, key):
        return clean_text(obj.get(key, "")).lower()
    cat_a, cat_b = val(a, "category"), val(b, "category")
    in_a, in_b = val(a, "inputs"), val(b, "inputs")
    out_a, out_b = val(a, "outputs"), val(b, "outputs")
    name_a, name_b = val(a, "task_name"), val(b, "task_name")
    desc_a, desc_b = val(a, "description"), val(b, "description")
    return {
        "similarity": float(similarity),
        "same_category": float(bool(cat_a and cat_b and cat_a == cat_b)),
        "same_inputs": float(bool(in_a and in_b and in_a == in_b)),
        "same_outputs": float(bool(out_a and out_b and out_a == out_b)),
        "name_overlap": float(bool(name_a and name_b and (name_a in name_b or name_b in name_a))),
        "description_overlap": float(bool(desc_a and desc_b and (desc_a in desc_b or desc_b in desc_a))),
    }

def redundancy_assessment(features: dict[str, float]) -> tuple[str, str]:
    """Transparent baseline; thresholds are heuristics, not calibrated probabilities."""
    score = features["similarity"]
    context_matches = features["same_inputs"] + features["same_outputs"]
    if score >= 0.82 and context_matches >= 1:
        return "High-priority review", "Strong semantic similarity plus matching inputs or outputs."
    if score >= 0.72 and (context_matches >= 1 or features["name_overlap"]):
        return "Review candidate", "Similar work with at least one contextual overlap."
    if score >= 0.78:
        return "Similar; verify purpose", "Text is highly similar, but evidence of duplicate work is limited."
    return "No strong redundancy signal", "Current signals do not strongly suggest duplicated work."

def build_pair_dataset(tasks: pd.DataFrame, matrix: np.ndarray) -> pd.DataFrame:
    rows = []
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            a, b = tasks.iloc[i], tasks.iloc[j]
            sim = float(matrix[i, j])
            feats = pair_features(a, b, sim)
            assessment, rationale = redundancy_assessment(feats)
            rows.append({
                "task_a_id": int(a["id"]), "task_a_name": a["task_name"],
                "task_b_id": int(b["id"]), "task_b_name": b["task_name"],
                "similarity": sim, **{k: v for k, v in feats.items() if k != "similarity"},
                "assessment": assessment, "rationale": rationale,
            })
    return pd.DataFrame(rows)

FEATURE_COLUMNS = ["similarity", "same_category", "same_inputs", "same_outputs", "name_overlap", "description_overlap"]

def train_review_classifier(reviews: pd.DataFrame, tasks: pd.DataFrame, matrix: np.ndarray):
    """Train a binary classifier from reviewed pairs, returning model + evaluation info.

    The model is intentionally withheld unless both classes have at least 2 examples.
    """
    if reviews.empty:
        return None, {"error": "No reviewed pairs yet. Label task pairs first."}
    indexed = tasks.reset_index(drop=True)
    id_to_idx = {int(row["id"]): i for i, (_, row) in enumerate(indexed.iterrows())}
    feature_rows, labels = [], []
    for _, review in reviews.iterrows():
        a_id, b_id = int(review["task_a_id"]), int(review["task_b_id"])
        if a_id not in id_to_idx or b_id not in id_to_idx:
            continue
        a_idx, b_idx = id_to_idx[a_id], id_to_idx[b_id]
        features = pair_features(indexed.iloc[a_idx], indexed.iloc[b_idx], matrix[a_idx, b_idx])
        feature_rows.append([features[c] for c in FEATURE_COLUMNS])
        labels.append(1 if review["label"] == "Redundant" else 0)
    X = np.asarray(feature_rows, dtype=float)
    y = np.asarray(labels, dtype=int)
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2 or min(counts) < 2:
        return None, {"error": "Need at least 2 reviewed examples in each class: Redundant and not redundant."}
    model = make_pipeline(
        # Scale is not essential for logistic regression, but helps keep coefficients stable.
        __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    )
    info = {}
    if len(y) >= 10 and min(counts) >= 3:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        info = {
            "evaluation_method": "Stratified hold-out split (75% train / 25% test)",
            "test_size": int(len(y_test)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
            "class_names": ["Not redundant", "Redundant"],
        }
        # Refit on all reviewed examples only after recording held-out performance.
        model.fit(X, y)
    else:
        model.fit(X, y)
        info = {
            "evaluation_method": "Training fit only; not a valid estimate of generalization",
            "warning": "Collect more reviewed pairs before interpreting performance metrics.",
            "reviewed_pairs": int(len(y)),
        }
    info["training_examples"] = int(len(y))
    info["positive_examples"] = int(np.sum(y))
    return model, info

def model_predict_pair(model, a, b, similarity: float) -> tuple[int, float]:
    features = pair_features(a, b, similarity)
    x = np.asarray([[features[c] for c in FEATURE_COLUMNS]], dtype=float)
    pred = int(model.predict(x)[0])
    try:
        probability = float(model.predict_proba(x)[0, 1])
    except Exception:
        probability = float(pred)
    return pred, probability
