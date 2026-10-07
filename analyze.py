"""Ask questions about a CSV with an LLM that sees the column profile and the results, never the rows.

1. The CSV is profiled locally: column types, nulls, ranges, top values.
2. The profile and your question go to the model, which writes a pandas function.
3. You read the code. With --run it executes locally in a separate Python process.
4. The result (a small table) goes back to the model for a short written answer.

    pip install pandas
    export OPENAI_API_KEY=...                      # key for the endpoint
    export OPENAI_BASE_URL=https://argolink.io/v1  # optional, this is the default
    python analyze.py data/orders.csv "Which region and channel lost the most revenue from Q2 to Q3?"
    python analyze.py data/orders.csv "Same question" --run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

DEFAULT_BASE_URL = "https://argolink.io/v1"
USER_AGENT = "ai-data-analysis-workflow/1.0"  # some endpoints reject urllib's default agent
# USD per 1M input / output tokens on the default endpoint, checked 2026-10-07.
PRICES = {
    "gpt-6-sol": (0.60, 3.00),
    "gpt-5.6-luna": (0.06, 0.36),
    "kimi-k2.7-code": (0.76, 3.20),
    "deepseek-v4-pro": (0.528, 1.584),
    "deepseek-v4.1-flash": (0.12, 0.48),
    "claude-sonnet-5": (0.80, 4.00),
}
MAX_RESULT_CHARS = 6000

CODE_RULES = """Write one Python function `analyze(df)` that answers the question with pandas.
Rules:
- df is already loaded; columns and types are in the profile. Do not read or write files.
- Use only pandas and numpy (import them inside the function).
- Return a small JSON-serializable dict: lists of records or scalars, at most 50 rows in total.
- Name result keys so a reader understands them without the code.
Reply with the function in one ```python block, then one sentence on the method."""


def load(path: str | Path) -> pd.DataFrame:
    # Only empty cells are missing. pandas' default would also turn strings such as
    # "NA", "N/A" or "null" into NaN, so a region code NA (North America) would vanish.
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    for column in df.columns:
        if re.search(r"(date|time|_at)$", column, re.I):
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def profile(df: pd.DataFrame, sample_rows: int = 0) -> dict:
    """Schema and aggregates only. Raw rows are included only if you ask for them."""
    columns = []
    for name in df.columns:
        series = df[name]
        info = {"name": name, "dtype": str(series.dtype), "nulls": int(series.isna().sum()),
                "distinct": int(series.nunique())}
        if pd.api.types.is_bool_dtype(series):
            info["true_share"] = round(float(series.mean()), 4)
        elif pd.api.types.is_numeric_dtype(series):
            info.update(min=float(series.min()), max=float(series.max()), mean=round(float(series.mean()), 4))
        elif pd.api.types.is_datetime64_any_dtype(series):
            info.update(min=str(series.min().date()), max=str(series.max().date()))
        elif info["distinct"] <= 50:
            info["top"] = {str(k): int(v) for k, v in series.value_counts().head(8).items()}
        else:
            info["example_lengths"] = [len(str(v)) for v in series.dropna().head(3)]
        columns.append(info)
    result = {"rows": len(df), "columns": columns}
    if sample_rows:
        result["sample_rows"] = json.loads(df.head(sample_rows).to_json(orient="records", date_format="iso"))
    return result


def code_messages(question: str, data_profile: dict) -> list[dict]:
    return [
        {"role": "system", "content": CODE_RULES},
        {"role": "user", "content": f"Profile of the data:\n{json.dumps(data_profile, indent=1)}\n\nQuestion: {question}"},
    ]


def answer_messages(question: str, code: str, result: str) -> list[dict]:
    return [
        {"role": "system", "content": "Answer the question from the analysis result in at most 120 words. "
                                      "Quote the numbers you rely on. Say so if the result cannot answer it."},
        {"role": "user", "content": f"Question: {question}\n\nCode that produced the result:\n{code}\n\nResult:\n{result}"},
    ]


def extract_code(reply: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", reply, re.S)
    code = match.group(1) if match else reply
    if "def analyze" not in code:
        raise ValueError("the reply contains no analyze(df) function")
    return code.strip() + "\n"


RUNNER = """
import json, sys
sys.path.insert(0, sys.argv[3])
from analyze import load  # the same loading rules as the profile
df = load(sys.argv[1])
namespace = {}
exec(open(sys.argv[2]).read(), namespace)
print(json.dumps(namespace["analyze"](df), default=str))
"""


def run_code(code_path: str | Path, csv_path: str | Path, timeout: int = 120) -> str:
    """Run analyze(df) in a separate process with the API key removed from its environment.
    This is isolation from your key, not a sandbox: read the code before you run it."""
    env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY",)}
    here = str(Path(__file__).resolve().parent)
    done = subprocess.run([sys.executable, "-c", RUNNER, str(csv_path), str(code_path), here],
                          capture_output=True, text=True, timeout=timeout, env=env)
    if done.returncode != 0:
        raise RuntimeError(f"analysis failed:\n{done.stderr[-2000:]}")
    return done.stdout.strip()[:MAX_RESULT_CHARS]


def chat(messages: list[dict], model: str, effort: str | None = None) -> tuple[str, dict]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("set OPENAI_API_KEY to the key for your endpoint")
    body = {"model": model, "messages": messages, "max_completion_tokens": 8000}
    if effort:
        body["reasoning_effort"] = effort
    base = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    request = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(), method="POST")
    request.add_header("Authorization", f"Bearer {key}")
    request.add_header("Content-Type", "application/json")
    request.add_header("User-Agent", USER_AGENT)
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{error.code}: {error.read().decode(errors='replace')[:300]}") from None
    return data["choices"][0]["message"]["content"] or "", data.get("usage", {})


def cost(model: str, usages: list[dict]) -> float | None:
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return sum(u.get("prompt_tokens", 0) * price_in + u.get("completion_tokens", 0) * price_out for u in usages) / 1e6


def generate(csv_path: str | Path, question: str, model: str = "gpt-6-sol", effort: str | None = None,
             sample_rows: int = 0, out_dir: str | Path = "analysis") -> dict:
    """Steps 1-2: profile locally, ask the model for analyze(df), save it for review."""
    data_profile = profile(load(csv_path), sample_rows)
    reply, usage = chat(code_messages(question, data_profile), model, effort)
    code = extract_code(reply)
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    code_path = folder / f"{len(list(folder.glob('*.py'))) + 1:03d}_analysis.py"
    code_path.write_text(f"# Question: {question}\n# Model: {model}\n{code}")
    return {"code_path": code_path, "code": code, "usages": [usage]}


def answer(csv_path: str | Path, code_path: str | Path, model: str = "gpt-6-sol", effort: str | None = None) -> dict:
    """Steps 3-4: run the reviewed file locally, then ask for a written answer from its result."""
    source = Path(code_path).read_text()
    question = source.splitlines()[0].removeprefix("# Question: ") if source.startswith("# Question: ") else ""
    result = run_code(code_path, csv_path)
    text, usage = chat(answer_messages(question, source, result), model, effort)
    return {"result": result, "answer": text, "usages": [usage]}


def report(model: str, usages: list[dict]) -> str:
    total = cost(model, usages)
    tokens_in = sum(u.get("prompt_tokens", 0) for u in usages)
    tokens_out = sum(u.get("completion_tokens", 0) for u in usages)
    return f"[{model}] {tokens_in} in / {tokens_out} out" + (f", ${total:.5f}" if total is not None else "")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask an LLM about a CSV without sending its rows")
    parser.add_argument("csv")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--model", default="gpt-6-sol")
    parser.add_argument("--effort", help="reasoning_effort for models that take one, e.g. low")
    parser.add_argument("--run", action="store_true", help="run the generated code right away (skips your review)")
    parser.add_argument("--run-file", help="run a saved, reviewed analysis file and get the written answer")
    parser.add_argument("--sample-rows", type=int, default=0, help="also send the first N rows (off by default)")
    parser.add_argument("--show-profile", action="store_true", help="print what would be sent and stop")
    args = parser.parse_args()

    if args.show_profile:
        print(json.dumps(profile(load(args.csv), args.sample_rows), indent=1))
        return 0
    if args.run_file:
        done = answer(args.csv, args.run_file, args.model, args.effort)
        print(f"Result: {done['result']}\n\n{done['answer']}")
        print("\n" + report(args.model, done["usages"]), file=sys.stderr)
        return 0
    if not args.question:
        parser.error("give a question, or --run-file with a saved analysis")
    made = generate(args.csv, args.question, args.model, args.effort, args.sample_rows)
    print(made["code"])
    usages = made["usages"]
    if args.run:
        done = answer(args.csv, made["code_path"], args.model, args.effort)
        print(f"Result: {done['result']}\n\n{done['answer']}")
        usages += done["usages"]
    else:
        print(f"Saved to {made['code_path']}. Read it, then: python analyze.py {args.csv} --run-file {made['code_path']}")
    print("\n" + report(args.model, usages), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
