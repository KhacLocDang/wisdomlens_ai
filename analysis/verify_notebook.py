import json
import locale
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def safe_print_df(df, rows=5):
    print(json.dumps(df.head(rows).to_dict(orient='records'), ensure_ascii=True, indent=2, default=str))


def row_signature(row):
    return tuple(
        json.dumps(value, ensure_ascii=False, default=str)
        if isinstance(value, (dict, list))
        else value
        for value in row
    )


os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    locale.setlocale(locale.LC_ALL, "C.UTF-8")
except Exception:
    pass

repo_root = Path(__file__).resolve().parent.parent
backend_root = repo_root / 'backend'
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from sqlalchemy import inspect, select
from app.config import get_database_url
from app.database import SessionLocal
from app.models.inquiry import Inquiry

print('Repo root:', repo_root)
print('Database URL loaded:', get_database_url())

with SessionLocal() as db:
    inspector = inspect(db.bind)
    table_names = inspector.get_table_names()
    print('Tables found:', table_names)

    columns = inspector.get_columns('inquiries')
    print('Columns in inquiries:')
    for col in columns:
        print(f" - {col['name']}: {col['type']}")

    result = db.execute(
        select(
            Inquiry.id,
            Inquiry.question,
            Inquiry.summary,
            Inquiry.perspectives,
            Inquiry.buddhism,
            Inquiry.western_philosophy,
            Inquiry.psychology,
            Inquiry.language,
            Inquiry.source,
            Inquiry.model,
            Inquiry.created_at,
        )
    ).all()

columns = [
    'id', 'question', 'summary', 'perspectives',
    'buddhism', 'western_philosophy', 'psychology',
    'language', 'source', 'model', 'created_at'
]

df = pd.DataFrame(result, columns=columns)
print('Rows:', len(df))
print('Columns:', len(df.columns))
safe_print_df(df, 5)

duplicate_rows = int(
    df.apply(row_signature, axis=1).duplicated(keep=False).sum()
)
print('Fully duplicated rows:', duplicate_rows)
question_duplicates = df[df['question'].notna()].duplicated(subset=['question'], keep=False)
print('Potentially duplicated questions:', int(question_duplicates.sum()))

print('Question length (characters):')
df['question_len_chars'] = df['question'].fillna('').str.len()
df['question_len_words'] = df['question'].fillna('').str.split().str.len()
print(df['question_len_chars'].describe().to_string())

print('Summary length (characters):')
df['summary_len_chars'] = df['summary'].fillna('').str.len()
df['summary_len_words'] = df['summary'].fillna('').str.split().str.len()
print(df['summary_len_chars'].describe().to_string())


def normalize_perspective_dict(value):
    if pd.isna(value):
        return {}
    try:
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items() if v is not None}
        return {}
    except Exception:
        return {}


df['perspectives_dict'] = df['perspectives'].map(normalize_perspective_dict)
all_perspective_keys = []
for item in df['perspectives_dict']:
    all_perspective_keys.extend(item.keys())

perspective_counts = pd.Series(all_perspective_keys).value_counts()
print('Perspective counts from perspectives JSONB:')
print(perspective_counts.to_string())

print('Daily counts head:')
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
daily_counts = df.groupby(df['created_at'].dt.floor('D')).size().rename('count')
print(daily_counts.head(10).to_string())

missing_report = pd.DataFrame({
    'column': df.columns,
    'null_count': [int(df[col].isna().sum()) for col in df.columns],
    'null_pct': [round(df[col].isna().mean() * 100, 2) for col in df.columns],
}).sort_values('null_pct', ascending=False)
print('Missing data report:')
print(missing_report.to_string())

initial_observations = {
    'records_total': int(len(df)),
    'records_with_question': int(df['question'].notna().sum()),
    'records_with_summary': int(df['summary'].notna().sum()),
    'available_perspectives': sorted(set(all_perspective_keys)),
    'duplicate_questions': int(df[df['question'].notna()].duplicated(subset=['question'], keep=False).sum()),
    'date_range_start': str(df['created_at'].min()),
    'date_range_end': str(df['created_at'].max()),
}

print('Initial observations:')
print(json.dumps(initial_observations, indent=2, ensure_ascii=False, default=str))
