# Workflow Intelligence Tracker

A local-first Python app for individual productivity. It tracks tasks, imports CSV/Excel files,
compares work semantically, flags *potential* redundancy, captures human reviews, and evaluates
a redundancy classifier when labeled examples are available.

## Features

- Manual task creation and editing
- CSV/XLSX import
- SQLite persistence (`workflow_tracker.db`)
- Semantic similarity using Sentence Transformers (optional at runtime); TF-IDF fallback
- Separate similarity and redundancy assessments
- Human-in-the-loop labels: `Redundant`, `Similar but necessary`, `Unrelated`
- Optional supervised classifier trained from reviewed task pairs
- Evaluation metrics: precision, recall, F1, confusion matrix, cross-validation where feasible
- CSV export of tasks and pair comparisons

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Sentence Transformers may download a model the first time it is used. If unavailable, select
TF-IDF in the app. The app works without a pretrained model.

## Import format

Required column:
- `task_name`

Recommended columns:
- `description`
- `category`
- `status` (`Todo`, `In Progress`, `Done`, `Blocked`)
- `estimated_minutes`
- `actual_minutes`
- `created_at`
- `completed_at`
- `inputs`
- `outputs`
- `workflow_name`

Column names are case-insensitive; spaces and hyphens are normalized. Unknown columns are ignored.
A sample file is provided in `sample_tasks.csv`.

## How similarity differs from redundancy

- **Similarity** means two task descriptions are semantically alike. It is a ranking signal, not a
  claim that the tasks should be removed.
- **Redundancy** means the same work may be performed more than once without a meaningful
  difference in purpose or output. The app uses similarity plus contextual overlap (category,
  inputs, outputs, and description) to flag candidates.
- Human review is the ground truth. Similar-but-necessary pairs should remain distinct.
- The optional classifier learns from reviewed pairs only after enough examples are available.
  It is a decision-support tool, not an automatic task deleter.

## Recommended evaluation

1. Review a representative sample of suggested pairs.
2. Label each pair as `Redundant`, `Similar but necessary`, or `Unrelated`.
3. Evaluate the binary redundancy classifier on labeled examples using a held-out split where
   enough examples exist; inspect precision, recall, F1, and confusion matrix.
4. Avoid evaluating on the same examples used to fit the model.
5. Track false positives especially carefully: wrongly merging necessary work is costly.
6. Test workflow recommendations manually and compare measured completion time before/after.

## Project structure

- `app.py` — Streamlit UI
- `workflow_engine.py` — text similarity, feature extraction, classification, metrics
- `storage.py` — SQLite storage
- `sample_tasks.csv` — example import data
