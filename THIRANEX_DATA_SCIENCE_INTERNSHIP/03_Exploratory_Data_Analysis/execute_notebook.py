"""
Notebook Executor Script for Task 3: Exploratory Data Analysis (EDA)
Executes notebooks/Task_3_EDA.ipynb programmatically
and saves executed outputs into the .ipynb file.
"""

import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def execute_notebook():
    project_root = os.path.dirname(os.path.abspath(__file__))
    nb_path = os.path.join(project_root, "notebooks", "Task_3_EDA.ipynb")

    if not os.path.exists(nb_path):
        raise FileNotFoundError(f"Notebook file not found at: {nb_path}")

    print(f"[INFO] Reading notebook from: {nb_path}")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    print("[INFO] Executing notebook cells...")
    try:
        notebook_dir = os.path.join(project_root, "notebooks")
        ep.preprocess(nb, {'metadata': {'path': notebook_dir}})
        print("[SUCCESS] Task 3 Notebook executed successfully without errors!")
    except Exception as e:
        print(f"[ERROR] Notebook execution failed: {e}")
        raise e

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"[VERIFIED] Executed notebook saved to: {nb_path}")


if __name__ == "__main__":
    execute_notebook()
