from __future__ import annotations

from datetime import datetime
import io
import pandas as pd
import streamlit as st
import plotly.express as px

import storage
from workflow_engine import SimilarityEngine, build_pair_dataset, train_review_classifier, model_predict_pair

st.set_page_config(page_title="Workflow Intelligence Tracker", page_icon="🧭", layout="wide")
storage.init_db()

st.title("🧭 Workflow Intelligence Tracker")
st.caption("Track work, compare tasks semantically, and review possible redundancy without automatically deleting or merging tasks.")

with st.sidebar:
    st.header("Analysis settings")
    method = st.selectbox("Similarity engine", ["Sentence Transformers", "TF-IDF"], index=0,
                          help="Sentence Transformers captures semantic meaning. TF-IDF is a lightweight lexical baseline.")
    threshold = st.slider("Similarity review threshold", 0.0, 1.0, 0.70, 0.01)
    st.caption("Threshold controls the candidate list only. Similarity is not proof of redundancy.")

engine = SimilarityEngine(method)
tasks = storage.get_tasks()

def normalize_import(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    aliases = {"name": "task_name", "task": "task_name", "task_description": "description",
               "estimate_minutes": "estimated_minutes", "duration_minutes": "actual_minutes"}
    df = df.rename(columns={c: aliases[c] for c in df.columns if c in aliases})
    allowed = ["task_name", "description", "category", "status", "estimated_minutes",
               "actual_minutes", "created_at", "completed_at", "inputs", "outputs", "workflow_name"]
    for col in allowed:
        if col not in df.columns:
            df[col] = None
    df = df[allowed]
    df["task_name"] = df["task_name"].fillna("").astype(str).str.strip()
    df = df[df["task_name"] != ""]
    df["status"] = df["status"].fillna("Todo").replace("", "Todo")
    for col in ["estimated_minutes", "actual_minutes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

tab_overview, tab_add, tab_import, tab_tasks, tab_analysis, tab_reviews, tab_evaluation = st.tabs([
    "Overview", "Add task", "Import CSV/Excel", "Manage tasks", "Similarity & redundancy", "Review pairs", "Evaluation"
])

with tab_overview:
    tasks = storage.get_tasks()
    reviews = storage.get_reviews()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tasks tracked", len(tasks))
    c2.metric("Completed", int((tasks["status"] == "Done").sum()) if not tasks.empty else 0)
    c3.metric("Pairs reviewed", len(reviews))
    c4.metric("Estimated minutes", int(tasks["estimated_minutes"].fillna(0).sum()) if not tasks.empty else 0)
    if tasks.empty:
        st.info("Start by adding a task manually or importing sample_tasks.csv.")
    else:
        status_counts = tasks["status"].fillna("Unknown").value_counts().rename_axis("Status").reset_index(name="Tasks")
        st.plotly_chart(px.bar(status_counts, x="Status", y="Tasks", title="Tasks by status"), use_container_width=True)
        st.subheader("Recent tasks")
        st.dataframe(tasks[["id", "task_name", "category", "status", "estimated_minutes", "actual_minutes"]].head(10),
                     use_container_width=True, hide_index=True)

with tab_add:
    st.subheader("Create a task")
    with st.form("add_task_form", clear_on_submit=True):
        name = st.text_input("Task name *")
        description = st.text_area("Description", placeholder="What work is performed and why?")
        c1, c2 = st.columns(2)
        category = c1.text_input("Category", placeholder="e.g. Reporting, Learning")
        status = c2.selectbox("Status", ["Todo", "In Progress", "Done", "Blocked"])
        c3, c4 = st.columns(2)
        estimate = c3.number_input("Estimated minutes", min_value=0, value=0, step=5)
        actual = c4.number_input("Actual minutes (0 if unknown)", min_value=0, value=0, step=5)
        inputs = st.text_input("Inputs", placeholder="Files, datasets, notes, systems...")
        outputs = st.text_input("Outputs", placeholder="Report, cleaned dataset, email...")
        workflow = st.text_input("Workflow name", placeholder="Optional grouping")
        submitted = st.form_submit_button("Save task", type="primary")
        if submitted:
            if not name.strip():
                st.error("Task name is required.")
            else:
                storage.add_task({
                    "task_name": name.strip(), "description": description.strip(), "category": category.strip(),
                    "status": status, "estimated_minutes": float(estimate) if estimate else None,
                    "actual_minutes": float(actual) if actual else None, "created_at": datetime.now().isoformat(timespec="seconds"),
                    "completed_at": datetime.now().isoformat(timespec="seconds") if status == "Done" else None,
                    "inputs": inputs.strip(), "outputs": outputs.strip(), "workflow_name": workflow.strip()
                })
                st.success("Task saved.")
                st.rerun()

with tab_import:
    st.subheader("Import tasks")
    st.write("Upload a CSV or Excel file. `task_name` is required; other columns are optional.")
    uploaded = st.file_uploader("Choose CSV or XLSX", type=["csv", "xlsx"])
    if uploaded:
        try:
            raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
            normalized = normalize_import(raw)
            st.write(f"Preview: {len(normalized)} valid rows")
            st.dataframe(normalized.head(20), use_container_width=True, hide_index=True)
            if st.button("Import previewed tasks", type="primary"):
                count = storage.add_tasks_bulk(normalized)
                st.success(f"Imported {count} tasks.")
                st.rerun()
        except Exception as exc:
            st.error(f"Could not read file: {exc}")
    with open("sample_tasks.csv", "rb") as sample_file:
        st.download_button("Download sample CSV", sample_file, file_name="sample_tasks.csv", mime="text/csv")

with tab_tasks:
    st.subheader("Manage existing tasks")
    tasks = storage.get_tasks()
    if tasks.empty:
        st.info("No tasks to manage yet.")
    else:
        selected_id = st.selectbox("Select task ID", tasks["id"].tolist(),
                                   format_func=lambda x: f"#{x} — {tasks.loc[tasks.id == x, 'task_name'].iloc[0]}")
        selected = tasks[tasks["id"] == selected_id].iloc[0]
        with st.form("edit_task_form"):
            edit_name = st.text_input("Task name", value=str(selected["task_name"] or ""))
            edit_desc = st.text_area("Description", value=str(selected["description"] or ""))
            e1, e2 = st.columns(2)
            edit_cat = e1.text_input("Category", value=str(selected["category"] or ""))
            edit_status = e2.selectbox("Status", ["Todo", "In Progress", "Done", "Blocked"],
                                       index=["Todo", "In Progress", "Done", "Blocked"].index(selected["status"])
                                       if selected["status"] in ["Todo", "In Progress", "Done", "Blocked"] else 0)
            e3, e4 = st.columns(2)
            est_value = 0 if pd.isna(selected["estimated_minutes"]) else int(selected["estimated_minutes"])
            act_value = 0 if pd.isna(selected["actual_minutes"]) else int(selected["actual_minutes"])
            edit_est = e3.number_input("Estimated minutes", min_value=0, value=est_value, step=5)
            edit_actual = e4.number_input("Actual minutes", min_value=0, value=act_value, step=5)
            edit_inputs = st.text_input("Inputs", value=str(selected["inputs"] or ""))
            edit_outputs = st.text_input("Outputs", value=str(selected["outputs"] or ""))
            edit_workflow = st.text_input("Workflow name", value=str(selected["workflow_name"] or ""))
            save = st.form_submit_button("Save changes")
            if save:
                storage.update_task(int(selected_id), {
                    "task_name": edit_name.strip(), "description": edit_desc.strip(), "category": edit_cat.strip(),
                    "status": edit_status, "estimated_minutes": float(edit_est) if edit_est else None,
                    "actual_minutes": float(edit_actual) if edit_actual else None,
                    "completed_at": datetime.now().isoformat(timespec="seconds") if edit_status == "Done" else None,
                    "inputs": edit_inputs.strip(), "outputs": edit_outputs.strip(), "workflow_name": edit_workflow.strip()
                })
                st.success("Task updated.")
                st.rerun()
        if st.button("Delete selected task", type="secondary"):
            storage.delete_task(int(selected_id))
            st.warning("Task deleted.")
            st.rerun()
        st.download_button("Export tasks as CSV", tasks.to_csv(index=False).encode("utf-8"),
                           file_name="workflow_tasks.csv", mime="text/csv")

with tab_analysis:
    st.subheader("Compare tasks and find potential overlap")
    tasks = storage.get_tasks()
    if len(tasks) < 2:
        st.info("Add at least two tasks to compare them.")
    else:
        with st.spinner("Calculating task similarities..."):
            matrix = engine.similarity_matrix(tasks)
        st.caption(f"Active backend: {engine.last_backend}")
        pairs = build_pair_dataset(tasks, matrix)
        candidates = pairs[pairs["similarity"] >= threshold].sort_values("similarity", ascending=False)
        c1, c2 = st.columns(2)
        c1.metric("Task pairs compared", len(pairs))
        c2.metric("Pairs above threshold", len(candidates))
        if candidates.empty:
            st.info("No pairs meet the current similarity threshold. Try lowering it.")
        else:
            display = candidates[["task_a_id", "task_a_name", "task_b_id", "task_b_name",
                                  "similarity", "assessment", "rationale"]].copy()
            display["similarity"] = display["similarity"].map(lambda x: f"{x:.1%}")
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.download_button("Export pair analysis CSV", candidates.to_csv(index=False).encode("utf-8"),
                               file_name="workflow_pair_analysis.csv", mime="text/csv")
            st.markdown("#### Inspect a pair")
            pair_options = candidates.to_dict("records")
            chosen_index = st.selectbox(
                "Choose pair",
                range(len(pair_options)),
                format_func=lambda i: f'{pair_options[i]["task_a_name"]} ↔ {pair_options[i]["task_b_name"]} ({pair_options[i]["similarity"]:.0%})'
            )
            chosen = pair_options[chosen_index]
            left, right = st.columns(2)
            with left:
                st.markdown(f"**{chosen['task_a_name']}**")
                a_row = tasks[tasks["id"] == chosen["task_a_id"]].iloc[0]
                st.write(a_row["description"] or "No description")
                st.caption(f"Inputs: {a_row['inputs'] or '—'} | Outputs: {a_row['outputs'] or '—'}")
            with right:
                st.markdown(f"**{chosen['task_b_name']}**")
                b_row = tasks[tasks["id"] == chosen["task_b_id"]].iloc[0]
                st.write(b_row["description"] or "No description")
                st.caption(f"Inputs: {b_row['inputs'] or '—'} | Outputs: {b_row['outputs'] or '—'}")
            st.info(f"Assessment: **{chosen['assessment']}** — {chosen['rationale']}")
            st.warning("Do not merge or remove tasks solely because the similarity score is high. Confirm their purpose, owner, required output, and dependencies.")
            if st.button("Open this pair in Review pairs"):
                st.session_state["review_pair"] = (int(chosen["task_a_id"]), int(chosen["task_b_id"]))
                st.toast("Use the Review pairs tab to record a human label.")

with tab_reviews:
    st.subheader("Human review: teach the system what redundancy means")
    tasks = storage.get_tasks()
    reviews = storage.get_reviews()
    if len(tasks) < 2:
        st.info("Add at least two tasks first.")
    else:
        ids = tasks["id"].tolist()
        default_pair = st.session_state.get("review_pair", (ids[0], ids[1]))
        valid_default = default_pair[0] in ids and default_pair[1] in ids and default_pair[0] != default_pair[1]
        a_default = default_pair[0] if valid_default else ids[0]
        b_default = default_pair[1] if valid_default else ids[1]
        task_map = {int(r["id"]): f'#{r["id"]} — {r["task_name"]}' for _, r in tasks.iterrows()}
        a_id = st.selectbox("Task A", ids, index=ids.index(a_default), format_func=lambda x: task_map[x], key="review_a")
        b_choices = [x for x in ids if x != a_id]
        b_id = st.selectbox("Task B", b_choices, index=b_choices.index(b_default) if b_default in b_choices else 0,
                            format_func=lambda x: task_map[x], key="review_b")
        current_matrix = engine.similarity_matrix(tasks)
        index_map = {int(r["id"]): i for i, (_, r) in enumerate(tasks.iterrows())}
        sim = float(current_matrix[index_map[a_id], index_map[b_id]])
        st.metric("Current similarity", f"{sim:.1%}")
        st.write("Choose the *work relationship*, not just whether the wording looks similar.")
        label = st.radio("Review label", ["Redundant", "Similar but necessary", "Unrelated"], horizontal=False)
        notes = st.text_area("Review notes (optional)", placeholder="Why are these redundant or distinct?")
        if st.button("Save human review", type="primary"):
            storage.save_review(a_id, b_id, sim, label, notes)
            st.success("Review saved. This label can be used to train the optional classifier.")
            st.rerun()
        if not reviews.empty:
            st.markdown("#### Existing reviews")
            st.dataframe(reviews[["task_a_name", "task_b_name", "similarity", "label", "notes", "reviewed_at"]],
                         use_container_width=True, hide_index=True)
            st.download_button("Export review labels CSV", reviews.to_csv(index=False).encode("utf-8"),
                               file_name="workflow_pair_reviews.csv", mime="text/csv")

with tab_evaluation:
    st.subheader("Classifier training and evaluation")
    st.write("The supervised model learns from human-reviewed pairs. `Similar but necessary` and `Unrelated` are both treated as not redundant.")
    tasks = storage.get_tasks()
    reviews = storage.get_reviews()
    if len(tasks) < 2:
        st.info("Add at least two tasks and review task pairs first.")
    else:
        with st.spinner("Preparing current similarity features..."):
            matrix = engine.similarity_matrix(tasks)
        model, info = train_review_classifier(reviews, tasks, matrix)
        if model is None:
            st.warning(info.get("error", "Not enough reviewed data."))
        else:
            st.success("Classifier trained from reviewed examples.")
            st.json(info)
            if info.get("warning"):
                st.warning(info["warning"])
            st.markdown("#### Predict a reviewed-style redundancy risk")
            task_map = {int(r["id"]): f'#{r["id"]} — {r["task_name"]}' for _, r in tasks.iterrows()}
            ids = tasks["id"].tolist()
            pa = st.selectbox("Prediction task A", ids, format_func=lambda x: task_map[x], key="pred_a")
            pb_options = [x for x in ids if x != pa]
            pb = st.selectbox("Prediction task B", pb_options, format_func=lambda x: task_map[x], key="pred_b")
            index_map = {int(r["id"]): i for i, (_, r) in enumerate(tasks.iterrows())}
            pred, prob = model_predict_pair(model, tasks.iloc[index_map[pa]], tasks.iloc[index_map[pb]],
                                            float(matrix[index_map[pa], index_map[pb]]))
            st.metric("Estimated redundancy probability", f"{prob:.1%}")
            st.caption("This probability is only as reliable as the reviewed labels and evaluation data. Review the pair before changing your workflow.")

st.divider()
st.caption("Privacy: data is stored in a local SQLite database beside the app. Do not import confidential work data unless your environment is approved for it.")
