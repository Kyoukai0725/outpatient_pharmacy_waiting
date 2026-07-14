"""2024 machine add/remove changes: observed before/after comparison vs model prediction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase2.intervention import _y_probs, expected_y, minutes_to_quartile

DATA = ROOT / "data"
PHASE2 = DATA / "phase2"
CHANGE_FILE = ROOT / "raw" / "machine_changes" / "2024机器变更情况.xlsx"
DISPENSE_FILE = ROOT / "raw" / "dispense" / "门诊药房药品调配时间统计-20251226.xlsx"

MACHINE_SEC = 17.0
DEFAULT_MANUAL_SEC = 30.0


def _normalize_ym_text(s) -> str:
    """Normalize Excel time cells; recover 2024.10 when float-truncated to 2024.1."""
    if isinstance(s, (int, float, np.integer, np.floating)) and not pd.isna(s):
        y = int(s)
        m = int(round((float(s) - y) * 100))
        if m == 0:
            m = 1
        return f"{y}.{m:02d}"
    text = str(s).strip().strip("'\" ").replace("-", ".")
    if "." in text:
        y, m = text.split(".")[:2]
        return f"{int(float(y))}.{int(float(m)):02d}"
    return text


def parse_ym(s) -> pd.Timestamp:
    text = _normalize_ym_text(s)
    y, m = text.split(".")[:2]
    return pd.Timestamp(year=int(y), month=int(m), day=1)


def load_changes() -> pd.DataFrame:
    # Keep time as string so 2024.10 is not float-truncated to 2024.1
    adds = pd.read_excel(CHANGE_FILE, sheet_name="新增", dtype={"时间": str})
    dels = pd.read_excel(CHANGE_FILE, sheet_name="删除", dtype={"时间": str})
    adds["材料编号"] = adds["材料编号"].astype(str).str.strip()
    dels["材料编号"] = dels["材料编号"].astype(str).str.strip()
    adds["时间"] = adds["时间"].map(_normalize_ym_text)
    dels["时间"] = dels["时间"].map(_normalize_ym_text)
    adds["change_type"] = "add_to_machine"
    dels["change_type"] = "remove_from_machine"
    return pd.concat([adds, dels], ignore_index=True)


def resolve_seconds(change_type: str, drugid: str, disp: pd.DataFrame, items: pd.DataFrame) -> tuple[float, float]:
    """Return (pre_sec, post_sec) for the change."""
    row = disp[disp["材料编号"] == drugid]
    item_sec = None
    sub = items[items["drugid"] == drugid]
    if len(sub):
        item_sec = float(sub["调配秒数_最终"].iloc[0])

    if change_type == "add_to_machine":
        # pre: manual, post: machine 17
        man = DEFAULT_MANUAL_SEC
        if len(row) and pd.notna(row["单品项人工调配时间/秒"].iloc[0]):
            man = float(row["单品项人工调配时间/秒"].iloc[0])
        elif item_sec and item_sec > 20:
            # if somehow still manual in table
            man = item_sec
        return man, MACHINE_SEC

    # remove from machine: pre machine 17, post manual
    man = DEFAULT_MANUAL_SEC
    if len(row) and pd.notna(row["单品项人工调配时间/秒"].iloc[0]):
        man = float(row["单品项人工调配时间/秒"].iloc[0])
    elif item_sec and item_sec > 20:
        man = item_sec
    return MACHINE_SEC, man


def summarize_period(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"n": 0}
    return {
        "n": int(len(df)),
        "wait_mean": round(float(df["候药时长_分钟"].mean()), 3),
        "wait_median": round(float(df["候药时长_分钟"].median()), 3),
        "Y_mean": round(float(df["Y0"].mean()), 3),
        "P_Q4": round(float((df["Y0"] == 3).mean()), 3),
        "call_mean": round(float(df["叫号等待_分钟"].mean()), 3),
    }


def before_after_stats(pre: pd.DataFrame, post: pd.DataFrame) -> dict:
    if len(pre) < 5 or len(post) < 5:
        return {
            "delta_wait": None,
            "delta_Y": None,
            "delta_PQ4": None,
            "ttest_p": None,
            "mannwhitney_p": None,
            "boot_ci95_delta_wait": None,
            "obs_wait_significant": None,
            "obs_Y_significant": None,
            "insufficient_n": True,
        }
    delta_wait = float(post["候药时长_分钟"].mean() - pre["候药时长_分钟"].mean())
    delta_Y = float(post["Y0"].mean() - pre["Y0"].mean())
    delta_PQ4 = float((post["Y0"] == 3).mean() - (pre["Y0"] == 3).mean())
    tt = stats.ttest_ind(pre["候药时长_分钟"], post["候药时长_分钟"], equal_var=False)
    mw = stats.mannwhitneyu(pre["候药时长_分钟"], post["候药时长_分钟"], alternative="two-sided")
    rng = np.random.default_rng(42)
    boots_w, boots_y = [], []
    for _ in range(2000):
        boots_w.append(
            post["候药时长_分钟"].sample(frac=1, replace=True, random_state=rng.integers(1e9)).mean()
            - pre["候药时长_分钟"].sample(frac=1, replace=True, random_state=rng.integers(1e9)).mean()
        )
        boots_y.append(
            post["Y0"].sample(frac=1, replace=True, random_state=rng.integers(1e9)).mean()
            - pre["Y0"].sample(frac=1, replace=True, random_state=rng.integers(1e9)).mean()
        )
    boots_w = np.asarray(boots_w)
    boots_y = np.asarray(boots_y)
    ci_w = [float(x) for x in np.quantile(boots_w, [0.025, 0.975])]
    ci_y = [float(x) for x in np.quantile(boots_y, [0.025, 0.975])]
    sig_w = (tt.pvalue < 0.05) or (mw.pvalue < 0.05) or not (ci_w[0] <= 0 <= ci_w[1])
    sig_y = not (ci_y[0] <= 0 <= ci_y[1])
    return {
        "delta_wait": round(delta_wait, 4),
        "delta_Y": round(delta_Y, 4),
        "delta_PQ4": round(delta_PQ4, 4),
        "ttest_p": float(tt.pvalue),
        "mannwhitney_p": float(mw.pvalue),
        "boot_ci95_delta_wait": ci_w,
        "boot_ci95_delta_Y": ci_y,
        "obs_wait_significant": bool(sig_w),
        "obs_Y_significant": bool(sig_y),
        "insufficient_n": False,
    }


def model_effect(
    sample: pd.DataFrame,
    items: pd.DataFrame,
    drugid: str,
    pre_sec: float,
    post_sec: float,
    cpts: dict,
    thr: dict,
    wait_by_y: dict,
) -> dict:
    ids = sample["处方编号"].unique()
    if len(ids) == 0:
        return {"n": 0}
    grouped = {k: g for k, g in items[items["处方编号"].isin(ids)].groupby("处方编号")}
    idx = sample.set_index("处方编号", drop=False)
    rows = []
    for rid in ids:
        if rid not in grouped:
            continue
        row = idx.loc[rid]
        g = grouped[rid]
        drugs = g["drugid"].astype(str).to_numpy()
        base = g["调配秒数_最终"].astype(float).to_numpy().copy()
        secs_old = base.copy()
        secs_new = base.copy()
        for i, did in enumerate(drugs):
            if did == drugid:
                secs_old[i] = pre_sec
                secs_new[i] = post_sec
        d0_old = minutes_to_quartile(secs_old.sum() / 60, thr["D0"])
        d0_new = minutes_to_quartile(secs_new.sum() / 60, thr["D0"])
        w0 = int(row["W0"])
        ey_old = expected_y(cpts, row, w0, d0_old)
        ey_new = expected_y(cpts, row, w0, d0_new)
        py_old = _y_probs(cpts, row, w0, d0_old)
        py_new = _y_probs(cpts, row, w0, d0_new)
        wait_old = sum(py_old[k] * wait_by_y[k] for k in range(4))
        wait_new = sum(py_new[k] * wait_by_y[k] for k in range(4))
        rows.append(
            {
                "disp_drop": (secs_old.sum() - secs_new.sum()) / 60,
                "d0_old": d0_old,
                "d0_new": d0_new,
                "ey_old": ey_old,
                "ey_new": ey_new,
                "wait_old": wait_old,
                "wait_new": wait_new,
                "pq4_old": float(py_old[3]),
                "pq4_new": float(py_new[3]),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return {"n": 0}
    return {
        "n": int(len(df)),
        "pre_sec": pre_sec,
        "post_sec": post_sec,
        "delta_sec": pre_sec - post_sec,
        "mean_disp_drop_min": round(float(df["disp_drop"].mean()), 4),
        "frac_D0_drop": round(float((df["d0_new"] < df["d0_old"]).mean()), 4),
        "pred_wait_drop_min": round(float((df["wait_old"] - df["wait_new"]).mean()), 4),
        "pred_EY_drop": round(float((df["ey_old"] - df["ey_new"]).mean()), 4),
        "pred_PQ4_drop": round(float((df["pq4_old"] - df["pq4_new"]).mean()), 4),
        "model_near_zero": bool(
            abs(float((df["wait_old"] - df["wait_new"]).mean())) < 0.2
            and abs(float((df["ey_old"] - df["ey_new"]).mean())) < 0.1
        ),
    }


def align(obs: dict, model: dict) -> dict:
    if obs.get("insufficient_n") or model.get("n", 0) == 0:
        return {"status": "insufficient_data"}
    obs_sig = bool(obs.get("obs_wait_significant"))
    model_nz = not bool(model.get("model_near_zero", True))
    # Alignment: observed nonsignificant ↔ model near zero; or observed significant improvement ↔ model same-direction improvement
    if not obs_sig and model.get("model_near_zero"):
        verdict = "aligned_both_nonsignificant"
        ok = True
    elif obs_sig and model_nz and np.sign(-obs["delta_wait"]) == np.sign(model["pred_wait_drop_min"]):
        verdict = "aligned_both_significant_same_sign"
        ok = True
    elif obs_sig and model.get("model_near_zero"):
        verdict = "mismatch_obs_sig_model_near_zero"
        ok = False
    elif (not obs_sig) and model_nz:
        verdict = "mismatch_obs_ns_model_nonzero"
        ok = False
    else:
        verdict = "mixed"
        ok = False
    return {
        "ok": ok,
        "verdict": verdict,
        "obs_wait_significant": obs_sig,
        "obs_delta_wait": obs.get("delta_wait"),
        "model_pred_wait_drop": model.get("pred_wait_drop_min"),
        "model_pred_EY_drop": model.get("pred_EY_drop"),
        "model_near_zero": model.get("model_near_zero"),
    }


def main() -> None:
    print("[1] Loading change log and data...")
    changes = load_changes()
    disp = pd.read_excel(DISPENSE_FILE)
    disp["材料编号"] = disp["材料编号"].astype(str).str.strip()

    rx = pd.read_parquet(DATA / "rx_level.parquet")
    disc = pd.read_parquet(PHASE2 / "rx_discretized.parquet")
    items = pd.read_parquet(
        DATA / "item_level.parquet",
        columns=["处方编号", "drugid", "调配秒数_最终", "是否机器_最终"],
    )
    with (PHASE2 / "calibrated_cpts.json").open(encoding="utf-8") as f:
        cpts = json.load(f)
    with (PHASE2 / "discretize_thresholds.json").open(encoding="utf-8") as f:
        thr = json.load(f)

    d_all = disc.merge(rx[["处方编号", "日期"]], on="处方编号")
    d_all["日期_dt"] = pd.to_datetime(d_all["日期"])
    wait_by_y = d_all.groupby("Y0")["候药时长_分钟"].mean().to_dict()

    results = []
    print("[2] Per-drug validation...")
    for _, ch in changes.iterrows():
        drugid = ch["材料编号"]
        ctype = ch["change_type"]
        change_m = parse_ym(ch["时间"])
        pre_start = change_m - pd.DateOffset(months=2)
        pre_end = change_m
        post_start = change_m + pd.DateOffset(months=1)
        post_end = change_m + pd.DateOffset(months=3)

        has = set(items.loc[items["drugid"] == drugid, "处方编号"])
        d = d_all[d_all["处方编号"].isin(has)].copy()
        pre = d[(d["日期_dt"] >= pre_start) & (d["日期_dt"] < pre_end)]
        post = d[(d["日期_dt"] >= post_start) & (d["日期_dt"] < post_end)]

        pre_sec, post_sec = resolve_seconds(ctype, drugid, disp, items)
        obs_ba = before_after_stats(pre, post)
        # Model: counterfactual on post sample (factual ≈ post seconds, CF = pre seconds)
        model = model_effect(post if len(post) >= 5 else pre, items, drugid, pre_sec, post_sec, cpts, thr, wait_by_y)
        al = align(obs_ba, model)

        rec = {
            "drugid": drugid,
            "name": str(ch["名称"]),
            "change_type": ctype,
            "change_month": str(ch["时间"]),
            "reason": str(ch.get("原因", "")),
            "pre_window": f"{pre_start.date()} ~ {pre_end.date()}",
            "post_window": f"{post_start.date()} ~ {post_end.date()}",
            "pre_sec": pre_sec,
            "post_sec": post_sec,
            "freq_total": int((items["drugid"] == drugid).sum()),
            "observed_pre": summarize_period(pre),
            "observed_post": summarize_period(post),
            "observed_effect": obs_ba,
            "model_effect": model,
            "alignment": al,
        }
        results.append(rec)
        print(
            f"  {ctype} {drugid} {ch['时间']}: "
            f"n_pre={len(pre)} n_post={len(post)} "
            f"sec {pre_sec}→{post_sec} | "
            f"obs Δwait={obs_ba.get('delta_wait')} sig={obs_ba.get('obs_wait_significant')} | "
            f"pred wait↓={model.get('pred_wait_drop_min')} | {al.get('verdict')}"
        )

    # pooled: all adds / all removes
    def pool(ctype: str) -> dict:
        subset = [r for r in results if r["change_type"] == ctype]
        ok = sum(1 for r in subset if r["alignment"].get("ok"))
        ns = sum(1 for r in subset if r["alignment"].get("verdict") == "aligned_both_nonsignificant")
        return {
            "n_drugs": len(subset),
            "aligned_ok": ok,
            "aligned_both_nonsignificant": ns,
            "drugs": [r["drugid"] for r in subset],
        }

    summary = {
        "source": str(CHANGE_FILE.name),
        "rule": "Add to machine: manual sec→17s; remove: 17s→manual sec; Rx containing drug only; ~2 months before/after change month",
        "n_changes": len(results),
        "pool_add": pool("add_to_machine"),
        "pool_remove": pool("remove_from_machine"),
        "n_aligned_ok": sum(1 for r in results if r["alignment"].get("ok")),
        "n_aligned_both_ns": sum(
            1 for r in results if r["alignment"].get("verdict") == "aligned_both_nonsignificant"
        ),
        "cases": results,
    }

    out_path = PHASE2 / "machine_change_2024_validation.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(f"Changes={summary['n_changes']}, aligned OK={summary['n_aligned_ok']}, both nonsignificant={summary['n_aligned_both_ns']}")
    print("saved:", out_path)


if __name__ == "__main__":
    main()
