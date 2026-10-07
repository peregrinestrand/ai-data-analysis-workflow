"""Offline tests: a fake urlopen stands in for the chat endpoint; profiling and the
analysis subprocess run for real on data/orders.csv."""
import io
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import analyze  # noqa: E402

CSV = ROOT / "data" / "orders.csv"
EXAMPLE = ROOT / "examples" / "001_q2_vs_q3_by_segment.py"


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def fake_api(monkeypatch, replies, sent):
    def urlopen(request, timeout=None):
        sent.append((request.full_url, json.loads(request.data)))
        content = replies.pop(0)
        return FakeResponse(json.dumps({
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 400}}).encode())

    monkeypatch.setattr(analyze.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("OPENAI_API_KEY", "test")


def test_profile_carries_no_rows_unless_asked():
    df = analyze.load(CSV)
    text = json.dumps(analyze.profile(df))
    assert "O-00001" not in text and "2026-06-15T" not in text
    assert "O-00001" in json.dumps(analyze.profile(df, sample_rows=2))


def test_region_code_na_is_not_read_as_missing():
    region = next(c for c in analyze.profile(analyze.load(CSV))["columns"] if c["name"] == "region")
    assert region["nulls"] == 0 and region["top"]["NA"] == 592


def test_types_are_detected_for_the_model():
    columns = {c["name"]: c for c in analyze.profile(analyze.load(CSV))["columns"]}
    assert columns["order_date"]["dtype"].startswith("datetime64")
    assert columns["returned"]["dtype"] == "bool"
    assert columns["unit_price"]["max"] == 990.0


def test_extract_code_needs_an_analyze_function():
    reply = "Here you go:\n```python\ndef analyze(df):\n    return {'rows': len(df)}\n```\nCounts rows."
    assert analyze.extract_code(reply).startswith("def analyze(df):")
    with pytest.raises(ValueError):
        analyze.extract_code("```python\nprint('hi')\n```")


def test_reference_analysis_finds_the_planted_drop():
    result = json.loads(analyze.run_code(EXAMPLE, CSV))
    worst = result["largest_q2_to_q3_revenue_drops"][0]
    assert (worst["region"], worst["channel"], worst["change_pct"]) == ("APAC", "partner", -74.0)


def test_generated_code_runs_without_the_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    code = tmp_path / "probe.py"
    code.write_text("def analyze(df):\n    import os\n    return {'key': os.environ.get('OPENAI_API_KEY'), 'rows': len(df)}\n")
    assert json.loads(analyze.run_code(code, CSV)) == {"key": None, "rows": 2400}


def test_failing_code_is_reported(tmp_path):
    code = tmp_path / "broken.py"
    code.write_text("def analyze(df):\n    return df['no_such_column'].sum()\n")
    with pytest.raises(RuntimeError, match="no_such_column"):
        analyze.run_code(code, CSV)


def test_generate_then_answer_from_the_reviewed_file(monkeypatch, tmp_path):
    sent = []
    code_reply = "```python\n" + EXAMPLE.read_text().split("\n", 2)[2] + "```\nPivot by quarter."
    fake_api(monkeypatch, [code_reply, "APAC partner revenue fell 74.0% from Q2 to Q3."], sent)

    made = analyze.generate(CSV, "Which segment lost the most revenue from Q2 to Q3?", out_dir=tmp_path)
    assert sent[0][0] == "https://argolink.io/v1/chat/completions"
    assert "O-00001" not in json.dumps(sent[0][1])          # no raw rows in the prompt
    assert made["code_path"].read_text().startswith("# Question: Which segment")

    done = analyze.answer(CSV, made["code_path"])
    assert '"change_pct": -74.0' in sent[1][1]["messages"][1]["content"]  # the result, not the rows
    assert done["answer"].startswith("APAC partner")
    assert analyze.report("gpt-6-sol", made["usages"] + done["usages"]) == "[gpt-6-sol] 1000 in / 800 out, $0.00300"
