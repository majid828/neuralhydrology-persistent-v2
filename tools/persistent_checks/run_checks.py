"""Executable checks of the v2 persistent-state code (repository source extracted verbatim; NumPy stand-in for torch).
Usage (from the repository root, no PyTorch needed): python3 -I tools/persistent_checks/run_checks.py ."""
import ast
import io
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = __file__.rsplit("/", 1)[0]
exec(open(f"{HERE}/harness.py").read())

RESULTS = []


def report(test_id, name, passed, detail):
    RESULTS.append((test_id, name, "PASS" if passed else "FAIL", detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {test_id} {name}\n        {detail}")


def per_basin(calls):
    d = defaultdict(list)
    for c in calls:
        d[c["basin"]].append(c)
    return d


def stats(calls):
    by = per_basin(calls)
    out = dict(none=0, transfers=0, contiguous=0, gap=0, overlap=0, wrong_basin=0, step_mismatch=0)
    for b, cs in by.items():
        cum = 0
        for c in cs:
            if c["h0"] is None:
                out["none"] += 1
                cum = 0
            else:
                out["transfers"] += 1
                last_t, hb, steps = c["h0"]
                out["wrong_basin"] += int(hb != b)
                delta = c["t_first"] - last_t
                out["contiguous" if delta == 1 else ("gap" if delta > 1 else "overlap")] += 1
                out["step_mismatch"] += int(steps != cum)
            cum += c["T"]
    out["internal_jumps"] = sum(c["internal_jumps"] > 0 for c in calls)
    out["static_ok"] = all(c["xs_shape"] == (1, 1) and float(c["xs"][0, 0]) == c["basin"] for c in calls)
    out["basins"] = len(by)
    return out


def coverage(ds, seen, L):
    """Multiset of target values expected (every finite target of every window) vs. scored by the loss."""
    expected = Counter()
    for _, (basin, idx) in ds.lookup_table.items():
        y = np.asarray(ds._y[basin]["1D"]).reshape(-1)
        for t in range(idx[0] - L + 1, idx[0] + 1):
            if np.isfinite(y[t]):
                expected[float(y[t])] += 1
    got = Counter(float(v) for arr in seen for v in arr)
    n_all = sum(expected.values())
    return n_all, sum(got.values()), (sum((got & expected).values()) if got == expected else -1)


print("=" * 100)
print("Repository:", REPO)
print("=" * 100)

# ------------------------------------------------------------------ daily config L=2, B=8
L, B = 2, 8
series = make_series({"A": 400, "B": 300, "C": 250})
ds = build_dataset(series, seq_len=L, stride=L, is_train=True)
tr = make_trainer(ds, B, L, seed=555)
(c1, s1), (c2, s2) = run_epochs(tr, [1, 2])
st1, st2 = stats(c1), stats(c2)

report("V1", "State continuity between consecutive batches of the same basin",
       st1["transfers"] == st1["contiguous"] and st1["wrong_basin"] == 0 and st1["step_mismatch"] == 0
       and st1["internal_jumps"] == 0,
       f"{st1['transfers']} hand-offs, {st1['contiguous']} contiguous; wrong basin {st1['wrong_basin']}; "
       f"internal jumps {st1['internal_jumps']}")
report("V1b", "Static attributes reattached for the right basin", st1["static_ok"], "x_s per call == basin vector")
report("V2", "Zero state only at a basin's first batch in each epoch",
       st1["none"] == st1["basins"] and st2["none"] == st2["basins"],
       f"epoch 1: {st1['none']} zero-state calls / {st1['basins']} basins; epoch 2: {st2['none']} / {st2['basins']}")
order1, order2 = [c["basin"] for c in c1], [c["basin"] for c in c2]
chrono = all(all(np.diff([c["t_first"] for c in cs]) > 0) for cs in per_basin(c1).values())
report("V3", "Chronological within basin, randomly interleaved across basins, different each epoch",
       chrono and order1 != order2 and sum(a != b for a, b in zip(order1, order1[1:])) > 3,
       f"chronological {chrono}; epochs differ {order1 != order2}")
tr_b = make_trainer(ds, B, L, seed=1)
((cb, _),) = run_epochs(tr_b, [1])
same = {b: [(c['t_first'], c['h0']) for c in cs] for b, cs in per_basin(c1).items()} == \
       {b: [(c['t_first'], c['h0']) for c in cs] for b, cs in per_basin(cb).items()}
report("V4", "Per-basin state sequence independent of interleaving", same and [c["basin"] for c in cb] != order1,
       f"identical per-basin (start, state) sequences across seeds: {same}")

n_all, n_seen, n_distinct = coverage(ds, s1, L)
per_call = Counter(len(x) for x in s1)
report("V5", "Loss covers every valid step of the flattened B*L stream (Eq. 17)",
       n_seen == n_all and n_distinct == n_all,
       f"targets scored in epoch 1: {n_seen} of {n_all}, each valid target exactly once: {n_distinct == n_all}; "
       f"loss calls by number of targets: {dict(per_call)}")

Lh, Bh = 24, 16
ds_h = build_dataset(make_series({"A": 24 * 120, "B": 24 * 90}), seq_len=Lh, stride=Lh, is_train=True)
tr_h = make_trainer(ds_h, Bh, Lh, seed=555)
((ch, sh),) = run_epochs(tr_h, [1])
na, ns_, nd = coverage(ds_h, sh, Lh)
sth = stats(ch)
report("V5b", "Same for hourly / 15-min configuration (L=24, B=16)", ns_ == na and nd == na and
       sth["contiguous"] == sth["transfers"], f"{ns_}/{na} targets scored, each exactly once: {nd == na}; {sth['contiguous']}/{sth['transfers']} "
       f"contiguous hand-offs")

proc = defaultdict(Counter)
for c in c1:
    for t in range(int(c["t_first"]), int(c["t_last"]) + 1):
        proc[c["basin"]][t] += 1
never = {b: sorted(set(range(n)) - set(proc[float(i)])) for i, (b, n) in enumerate([("A", 400), ("B", 300), ("C", 250)])}
report("V6", "No step processed twice; only the incomplete final segment is left out (Sec. 2.2)",
       max(max(v.values()) for v in proc.values()) == 1 and all(len(v) < L for v in never.values()),
       f"max repeats 1; unprocessed steps per basin: { {b: v for b, v in never.items()} }")

# ------------------------------------------------------------------ gaps
EVENTS.clear()
sg_series = make_series({"A": 400, "B": 300}, target_gap={"A": (100, 130)}, input_nan={"B": 151})
ds_g = build_dataset(sg_series, seq_len=L, stride=L, is_train=True)
tr_g = make_trainer(ds_g, B, L, seed=555)
((cg, sg),) = run_epochs(tr_g, [1])
stg = stats(cg)
target_gap_processed = all(t in {x for c in cg if c["basin"] == 0.0 for x in range(int(c["t_first"]), int(c["t_last"]) + 1)}
                           for t in range(100, 130))
n_updates = EVENTS.count("step")
n_batches = len(cg)
report("V7", "Target gap: state runs through the gap's forcing; forcing gap: reset, never a jump",
       stg["internal_jumps"] == 0 and stg["gap"] == 0 and target_gap_processed and stg["none"] == stg["basins"] + 1,
       f"internal jumps {stg['internal_jumps']}; hand-offs across a gap {stg['gap']}; target-gap steps 100-129 fed "
       f"to the model: {target_gap_processed}; zero-state starts {stg['none']} (2 basins + 1 reset after the NaN "
       f"forcing day); {n_batches} batches, {n_updates} updates ({n_batches - n_updates} batches without targets "
       f"skipped)")

# ------------------------------------------------------------------ order of operations
EVENTS.clear()
tr_e = make_trainer(build_dataset(make_series({"A": 64}), seq_len=L, stride=L, is_train=True), B, L)
run_epochs(tr_e, [1])
patt = ["forward", "zero_grad", "backward", "step", "cache"]
report("V8", "Per batch: forward -> zero_grad -> backward -> step -> cache detached state (Eq. 13, Alg. 1 step 16)",
       EVENTS == patt * (len(EVENTS) // 5), f"{len(EVENTS) // 5} batches; pattern {EVENTS[:5]}")

# ------------------------------------------------------------------ ablation
tr_a = make_trainer(ds, B, L, seed=555, ablate=True)
((ca, sa),) = run_epochs(tr_a, [1])
report("V9", "Ablation 'ablate_state_carry': every batch starts from zero, nothing else changes",
       all(c["h0"] is None for c in ca) and len(ca) == len(c1) and coverage(ds, sa, L)[2] == n_all,
       f"{sum(c['h0'] is None for c in ca)}/{len(ca)} zero-state calls; same batches and loss coverage")

# ------------------------------------------------------------------ config validation
def cfg_ns(**kw):
    c = _NS()
    defaults = dict(persistent_state=True, model="persistentlstm", use_frequencies=[], non_overlapping_sequences=True,
                    seq_length=2, predict_last_n=2, seq_stride=2, eval_warmup_steps=60,
                    eval_warmup_from_preceding_period=True, persistent_state_init_file=None, ablate_state_carry=False)
    defaults.update(kw)
    for k, v in defaults.items():
        setattr(c, k, v)
    return c


cases = [("valid daily config", cfg_ns(), False),
         ("valid 15-min config", cfg_ns(seq_length={"15min": 24}, predict_last_n={"15min": 24}, seq_stride=24,
                                        eval_warmup_steps=2880, use_frequencies=["15min"]), False),
         ("overlapping windows", cfg_ns(non_overlapping_sequences=False), True),
         ("predict_last_n < seq_length", cfg_ns(predict_last_n=1), True),
         ("seq_stride 3 with L=2", cfg_ns(seq_stride=3), True),
         ("warm-up 61 not multiple of L", cfg_ns(eval_warmup_steps=61), True),
         ("init file + preceding warm-up", cfg_ns(persistent_state_init_file="x.p"), True),
         ("persistent_state with cudalstm", cfg_ns(model="cudalstm"), True),
         ("non-persistent cudalstm run", cfg_ns(persistent_state=False, model="cudalstm"), False)]
outcomes = []
for name, c, should_raise in cases:
    try:
        NS["validate_persistent_config"](c)
        raised = False
    except ValueError:
        raised = True
    outcomes.append((name, raised == should_raise))
report("V10", "Configuration guards reject settings that break the method",
       all(ok for _, ok in outcomes), "; ".join(f"{n}: {'ok' if ok else 'WRONG'}" for n, ok in outcomes))

# ------------------------------------------------------------------ evaluation
def make_tester(L, B, warmup, ablate=False):
    te = object.__new__(NS["RegressionTester"])
    c = _NS()
    c.batch_size, c.persistent_state, c.model = B, True, "persistentlstm"
    c.predict_last_n, c.seq_length, c.non_overlapping_sequences, c.seq_stride = L, L, True, L
    c.eval_warmup_steps, c.ablate_state_carry = warmup, ablate
    te.cfg, te.device, te._eval_drop_last = c, "cpu", False
    te._persistent_hidden_eval, te._persistent_state_end_eval = None, None
    te._persistent_init_states, te._persistent_basin_eval = {}, "A"
    te.loss_obj = RecordingMSE(L)
    return te


def run_eval(te, ds_e):
    model = CounterLSTM()
    preds, obs, dates, losses, _ = te._evaluate(model, te._get_chrono_loader(ds_e), ["1D"])
    return model, preds, obs, dates


# warm-up taken from the preceding period: the dataset starts W steps before the period start, targets there are NaN
for (Le, Be, W, n_period) in [(2, 8, 60, 395), (2, 8, 90, 395), (24, 16, 720, 24 * 60)]:
    n = W + n_period
    ser = make_series({"A": n}, target_gap={"A": (0, W)})
    ds_e = build_dataset(ser, seq_len=Le, stride=Le, is_train=False)
    te = make_tester(Le, Be, W)
    model, preds, obs, dates = run_eval(te, ds_e)
    se = stats(model.calls)
    d = dates["1D"].reshape(-1)
    first_scored = int((d[0] - np.datetime64("1990-01-01")).astype(int))
    scored = len(d)
    seen = sum(len(x) for x in te.loss_obj.seen)
    finite_obs = int(np.isfinite(obs["1D"]).sum())
    report(f"V11 (L={Le},B={Be},W={W})", "Evaluation: continuous state, warm-up before the period, full period scored",
           se["none"] == 1 and se["contiguous"] == se["transfers"] and first_scored == W
           and scored == (n_period // Le) * Le and seen == finite_obs,
           f"{se['transfers']} contiguous hand-offs; first scored step = period start (index {first_scored}); "
           f"{scored} of {n_period} period steps scored (incomplete final window excluded: {n_period % Le}); "
           f"validation loss saw {seen} of {finite_obs} finite targets")

# offsets used by basedataset for the preceding-period warm-up (same expression as in the code)
offset_ok = []
for freq, W, start, expect in [("1D", 60, "2012-10-01", "2012-08-02"), ("1D", 90, "2012-10-01", "2012-07-03"),
                               ("1h", 720, "2012-10-01 00:00", "2012-09-01 00:00"),
                               ("15min", 2880, "2012-10-01 00:00", "2012-09-01 00:00")]:
    offset = (24 - 24) * to_offset(freq) + W * to_offset(freq)
    offset_ok.append(pd.Timestamp(start) - offset == pd.Timestamp(expect))
report("V12", "Preceding-period warm-up offsets for the four manuscript configs", all(offset_ok),
       "test_start - W*freq = 2012-08-02 (snow), 2012-07-03 (nonstationary), 2012-09-01 (hourly, 15-min)")

# NaN forcing at evaluation: state must not stay NaN for the rest of the basin
ser = make_series({"A": 200}, input_nan={"A": 37})
ds_e = build_dataset(ser, seq_len=2, stride=2, is_train=False)
te = make_tester(2, 8, 0)
model, preds, obs, dates = run_eval(te, ds_e)
yh = preds["1D"].reshape(-1)
nan_idx = np.where(np.isnan(yh))[0]
report("V13", "NaN forcing at evaluation: state reset after the affected batch, later predictions finite",
       len(nan_idx) > 0 and nan_idx.max() < 48 and np.all(np.isfinite(yh[48:])),
       f"NaN predictions only at indices {nan_idx.min()}..{nan_idx.max()} (the batch holding the NaN forcing day 37); "
       f"all {len(yh) - 48} later predictions finite")

# save final state of a validation stream, initialize the test stream from it
ser_val = make_series({"A": 100})
ds_v = build_dataset(ser_val, seq_len=2, stride=2, is_train=False)
te_v = make_tester(2, 8, 10)
model_v, *_ = run_eval(te_v, ds_v)
saved = {"A": {"h": np.asarray(te_v._persistent_hidden_eval[0]), "c": np.asarray(te_v._persistent_hidden_eval[1]),
               "last_date": te_v._persistent_state_end_eval}}
ser_test = make_series({"A": 60}, start=str(np.datetime64("1990-01-01") + 100))
ser_test["A"]["t"] = ser_test["A"]["t"] + 100
ds_t = build_dataset(ser_test, seq_len=2, stride=2, is_train=False)
te_t = make_tester(2, 8, 10)
te_t._persistent_init_states = saved
model_t, preds_t, _, dates_t = run_eval(te_t, ds_t)
first = model_t.calls[0]
report("V14", "Saved end-of-period state initializes the next period (operational use, Sec. 1)",
       first["h0"] is not None and first["h0"][0] == 99 and first["h0"][2] == 100 and int(preds_t["1D"].reshape(-1)[0]) == 101
       and dates_t["1D"].reshape(-1)[0] == np.datetime64("1990-04-11"),
       f"test stream's first call receives the saved state (last step {first['h0'][0]:.0f}, {first['h0'][2]:.0f} steps); "
       f"warm-up skipped; first scored date {dates_t['1D'].reshape(-1)[0]}")

te_g = make_tester(2, 8, 10)
te_g._persistent_init_states = {"A": dict(saved["A"], last_date=np.datetime64("1990-03-01"))}
model_g, *_ = run_eval(te_g, ds_t)
report("V14b", "A saved state that does not end right before the stream is ignored", model_g.calls[0]["h0"] is None,
       "non-contiguous saved state -> zero state + warm-up")

te_ab = make_tester(2, 8, 10, ablate=True)
model_ab, *_ = run_eval(te_ab, ds_v)
batch_starts = [c for c in model_ab.calls if int(c["t_first"]) % 16 == 0]
report("V15", "Ablation in evaluation: every flattened batch (16 steps) starts from zero",
       len(batch_starts) == int(np.ceil(100 / 16)) and all(c["h0"] is None for c in batch_starts),
       f"{len(batch_starts)} batches, zero-state at every batch start: {all(c['h0'] is None for c in batch_starts)} "
       f"(within a batch the warm-up slice hands its state to the scored slice, as in the normal stream)")

# ------------------------------------------------------------------ 15-min time zones
src = (REPO / "neuralhydrology/datasetzoo/fifteenminutecamelsus.py").read_text()
tree = ast.parse(src)
code = {n.targets[0].id if isinstance(n, ast.Assign) else n.name: ast.get_source_segment(src, n)
        for n in tree.body if (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "USGS_TZ_UTC_OFFSET_HOURS")
        or (isinstance(n, ast.FunctionDef) and n.name == "load_15min_us_discharge")}
tz_ns = {"pd": pd, "np": np, "Path": Path, "LOGGER": logging.getLogger("tz")}
exec(code["USGS_TZ_UTC_OFFSET_HOURS"], tz_ns)
exec(code["load_15min_us_discharge"], tz_ns)
import tempfile
tmp = Path(tempfile.mkdtemp())
(tmp / "usgs_15min_rdb_by_year").mkdir()
rdb = ("# comment\nagency_cd\tsite_no\tdatetime\ttz_cd\t1234_00060\t1234_00060_cd\n5s\t15s\t20d\t6s\t14n\t10s\n"
       "USGS\t12035000\t2012-03-11 01:45\tPST\t10\tA\nUSGS\t12035000\t2012-03-11 03:00\tPDT\t11\tA\n"
       "USGS\t12035000\t2012-11-04 01:45\tPDT\t20\tA\nUSGS\t12035000\t2012-11-04 01:00\tPST\t21\tA\n")
(tmp / "usgs_15min_rdb_by_year" / "12035000_2012.rdb").write_text(rdb)
q = tz_ns["load_15min_us_discharge"](tmp, "12035000", to_utc=True).dropna()
expect = {pd.Timestamp("2012-03-11 09:45"): 10, pd.Timestamp("2012-03-11 10:00"): 11,
          pd.Timestamp("2012-11-04 08:45"): 20, pd.Timestamp("2012-11-04 09:00"): 21}
got = {k: float(v) for k, v in q["qobs_cfs"].items()}
report("V16", "15-min discharge converted from local time (tz_cd) to UTC, DST-safe", got == expect,
       f"PST 01:45 -> 09:45 UTC, PDT 03:00 -> 10:00 UTC, PDT 01:45 -> 08:45 UTC, PST 01:00 -> 09:00 UTC: {got == expect}")

print("=" * 100)
print("SUMMARY")
for r in RESULTS:
    print(f"  {r[2]:4s}  {r[0]:24s} {r[1]}")
print(f"{sum(r[2] == 'PASS' for r in RESULTS)}/{len(RESULTS)} passed")
