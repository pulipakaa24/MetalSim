"""The GPU dashboard's progress parser is format-agnostic: counters, fractions, percentages, totals from
the command line, and a fallback when nothing is recognisable."""
import importlib.util, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("gpu_dashboard", os.path.join(ROOT, "scripts", "gpu_dashboard.py"))
gd = importlib.util.module_from_spec(spec); spec.loader.exec_module(gd)


def test_counter_with_total_from_cli():
    cur, tot, pct, name, src = gd.parse_progress(["it  898 steps 1 | ep_ret 3.0"], "python -m x --iters 1500")
    assert (cur, tot, name, src) == (898, 1500, "it", "counter") and abs(pct - 59.87) < 0.1


def test_counter_with_bare_total_and_anomaly_lines_skipped():
    lines = ["terms it 10 {...}", "[anomaly] it 11: joint limit violated by 0.2 rad"]
    cur, tot, *_ = gd.parse_progress(lines, "bash runs/il3/run_train.sh flat preset 1500 0")
    assert (cur, tot) == (10, 1500)


def test_inline_total_fraction_and_percent():
    assert gd.parse_progress(["iteration 12/500"], "")[:2] == (12, 500)
    assert gd.parse_progress(["frames: 40/200 [00:10<00:40]"], "")[:2] == (40, 200)
    assert gd.parse_progress(["progress 37.5% of frames"], "")[2] == 37.5
    assert gd.parse_progress([" 62%|######    | 620/1000"], "")[:2] == (620, 1000)
    assert gd.parse_progress(["busy processes (cpu > 40 %):"], "")[2] is None      # not a progress line


def test_no_signal():
    assert gd.parse_progress(["saved policy to runs/x.pt", "hello"], "") == (None, None, None, None, None)


def test_rate_and_eta():
    st = {}
    assert gd.rate_and_eta(st, "k", 100, 1000, 0.0) == (None, None)
    rate, eta = gd.rate_and_eta(st, "k", 160, 1000, 30.0)
    assert abs(rate - 2.0) < 1e-9 and abs(eta - 420.0) < 1e-6
    assert gd.rate_and_eta(st, "k", 5, 1000, 40.0) == (None, None)      # restart resets the window
