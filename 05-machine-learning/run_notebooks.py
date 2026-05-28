import json
import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS = [
    "Simple-Linear-Regression-v1.ipynb",
    "Mulitple-Linear-Regression-v1.ipynb",
    "Logistic-Regression-v1.ipynb",
    "Multi-class-Classification-V2.ipynb",
    "Decision-tree-classifier-drug-pred-v1.ipynb",
    "Regression-Trees-Taxi-Tip-v1.ipynb",
    "decision-tree-svm-ccFraud-v1.ipynb",
    "KNN-lab-v1.ipynb",
    "Random- Forests -XGBoost-v1.ipynb",
    "K-Means-Customer-Seg-v1.ipynb",
    "Comparing-DBScan-HDBScan-v1.ipynb",
    "PCA-v1.ipynb",
    "tSNE-UMAP-v1.ipynb",
    "Evaluating Classification Models-v1.ipynb",
    "Evaluating-random-forest-v1.ipynb",
    "Evaluating-k-means-clustering-v1.ipynb",
    "Regularization-in-LinearRegression-v1.ipynb",
    "ML-Pipelines-and-GridSearchCV-v1.ipynb",
]

def fix_notebook(nb_dict):
    nb_dict["nbformat"] = 4
    nb_dict["nbformat_minor"] = 5
    for cell in nb_dict.get("cells", []):
        if isinstance(cell.get("source"), list):
            cell["source"] = "".join(cell["source"])
        if cell.get("cell_type") == "code":
            if "execution_count" not in cell:
                cell["execution_count"] = None
            if "outputs" not in cell:
                cell["outputs"] = []
            for output in cell.get("outputs", []):
                if isinstance(output.get("text"), list):
                    output["text"] = "".join(output["text"])
                if isinstance(output.get("data"), dict):
                    for k, v in output["data"].items():
                        if isinstance(v, list):
                            output["data"][k] = "".join(v)
        elif cell.get("cell_type") == "markdown":
            cell.pop("outputs", None)
    return nb_dict

results = []
for nb_name in NOTEBOOKS:
    nb_path = os.path.join(os.getcwd(), nb_name)
    print(f"\n{'='*60}")
    print(f"Processing: {nb_name}")
    try:
        with open(nb_path, encoding="utf-8") as f:
            nb_dict = json.load(f)
        nb_dict = fix_notebook(nb_dict)
        nb = nbformat.from_dict(nb_dict)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": os.getcwd()}})
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"SUCCESS: {nb_name}")
        results.append((nb_name, "SUCCESS"))
    except Exception as e:
        print(f"ERROR: {nb_name} -> {e}")
        results.append((nb_name, f"ERROR: {e}"))

print("\n\n" + "="*60)
print("SUMMARY")
print("="*60)
for name, status in results:
    print(f"{status[:7]:8s} {name}")
