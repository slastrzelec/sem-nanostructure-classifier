"""Collect code and splits into data/kaggle_code/ for upload as a private Kaggle dataset."""
import hashlib
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent.parent
dst = root / "data" / "kaggle_code"
(dst / "semcls").mkdir(parents=True, exist_ok=True)
files = {p: dst / "semcls" / p.name for p in (root / "src" / "semcls").glob("*.py")}
files[root / "splits" / "splits.json"] = dst / "splits.json"
files[root / "splits" / "splits_v1.json"] = dst / "splits_v1.json"  # E1 runs were trained with this file (random section identical)
files[root / "scripts" / "kaggle_run.py"] = dst / "kaggle_run.py"
files[root / "scripts" / "kaggle_final_eval.py"] = dst / "kaggle_final_eval.py"
files[root / "scripts" / "report_runs.py"] = dst / "report_runs.py"
for s, d in files.items():
    shutil.copyfile(s, d)
    print(f"{hashlib.sha256(d.read_bytes()).hexdigest()[:12]}  {d.relative_to(dst)}")
