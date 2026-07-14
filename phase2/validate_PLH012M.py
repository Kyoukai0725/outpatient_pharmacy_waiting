"""PLH012M Mar 2024 location change: observed DiD vs model counterfactual prediction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase2.intervention import expected_y, minutes_to_quartile, prob_long_wait
from phase2.queue_propagation import attach_staff, propagate_call_wait_minutes

DATA = ROOT / "data"
PHASE2 = DATA / "phase2"
DRUG = "PLH012M"


def main() -> None:
    rx = pd.read_parquet(DATA / "rx_level.parquet")
    items = pd.read_parquet(
        DATA / "item_level.parquet",
        columns=["处方编号", "drugid", "是否机器_最终", "调配秒数_最终", "货架区域_最终"],
    )
    disc = pd.read_parquet(PHASE2 / "rx_discretized.parquet")
    staff = pd.read_parquet(PHASE2 / "daily_staff.parquet")
    with (PHASE2 / "calibrated_cpts.json").open(encoding="utf-8") as f:
        cpts = json.load(f)
    with (PHASE2 / "discretize_thresholds.json").open(encoding="utf-8") as f:
        thr = json.load(f)
    with (PHASE2 / "queue_model.json").open(encoding="utf-8") as f:
        qmodel = json.load(f)

    has = set(items.loc[items["drugid"] == DRUG, "处方编号"])
    d = disc.merge(rx[["处方编号", "日期"]], on="处方编号", how="left")
    d = attach_staff(d, staff)
    d["has"] = d["处方编号"].isin(has)
    d["日期_dt"] = pd.to_datetime(d["日期"])

    pre = d[(d["日期_dt"] >= "2024-02-01") & (d["日期_dt"] < "2024-03-01") & d["has"]].copy()
    post = d[(d["日期_dt"] >= "2024-04-01") & (d["日期_dt"] < "2024-06-01") & d["has"]].copy()
    mar = d[(d["日期_dt"] >= "2024-03-01") & (d["日期_dt"] < "2024-04-01") & d["has"]].copy()
    ctrl_pre = d[(d["日期_dt"] >= "2024-02-01") & (d["日期_dt"] < "2024-03-01") & (~d["has"])]
    ctrl_post = d[(d["日期_dt"] >= "2024-04-01") & (d["日期_dt"] < "2024-06-01") & (~d["has"])]

    print("=== Observed samples ===")
    print(f"pre Feb={len(pre)}, Mar={len(mar)}, post Apr-May={len(post)}")
    print(f"ctrl pre/post={len(ctrl_pre)}/{len(ctrl_post)}")

    obs = {
        "target_pre_n": int(len(pre)),
        "target_post_n": int(len(post)),
        "target_mar_n": int(len(mar)),
        "target_pre_wait_mean": float(pre["候药时长_分钟"].mean()),
        "target_post_wait_mean": float(post["候药时长_分钟"].mean()),
        "target_mar_wait_mean": float(mar["候药时长_分钟"].mean()) if len(mar) else None,
        "target_pre_wait_med": float(pre["候药时长_分钟"].median()),
        "target_post_wait_med": float(post["候药时长_分钟"].median()),
        "target_delta_wait": float(post["候药时长_分钟"].mean() - pre["候药时长_分钟"].mean()),
        "ctrl_delta_wait": float(ctrl_post["候药时长_分钟"].mean() - ctrl_pre["候药时长_分钟"].mean()),
        "did_wait": float(
            (post["候药时长_分钟"].mean() - pre["候药时长_分钟"].mean())
            - (ctrl_post["候药时长_分钟"].mean() - ctrl_pre["候药时长_分钟"].mean())
        ),
        "target_pre_Y": float(pre["Y0"].mean()),
        "target_post_Y": float(post["Y0"].mean()),
        "did_Y": float(
            (post["Y0"].mean() - pre["Y0"].mean())
            - (ctrl_post["Y0"].mean() - ctrl_pre["Y0"].mean())
        ),
        "target_pre_PQ4": float((pre["Y0"] == 3).mean()),
        "target_post_PQ4": float((post["Y0"] == 3).mean()),
        "did_PQ4": float(
            ((post["Y0"] == 3).mean() - (pre["Y0"] == 3).mean())
            - ((ctrl_post["Y0"] == 3).mean() - (ctrl_pre["Y0"] == 3).mean())
        ),
        "target_pre_call": float(pre["叫号等待_分钟"].mean()),
        "target_post_call": float(post["叫号等待_分钟"].mean()),
        "did_call": float(
            (post["叫号等待_分钟"].mean() - pre["叫号等待_分钟"].mean())
            - (ctrl_post["叫号等待_分钟"].mean() - ctrl_pre["叫号等待_分钟"].mean())
        ),
    }
    print(json.dumps(obs, indent=2, ensure_ascii=False))

    rng = np.random.default_rng(42)
    boots_w, boots_y = [], []
    for _ in range(2000):
        boots_w.append(
            (
                post["候药时长_分钟"].sample(frac=1, replace=True, random_state=rng.integers(1e9)).mean()
                - pre["候药时长_分钟"].sample(frac=1, replace=True, random_state=rng.integers(1e9)).mean()
            )
            - (
                ctrl_post["候药时长_分钟"]
                .sample(n=5000, replace=True, random_state=rng.integers(1e9))
                .mean()
                - ctrl_pre["候药时长_分钟"]
                .sample(n=5000, replace=True, random_state=rng.integers(1e9))
                .mean()
            )
        )
        boots_y.append(
            (
                post["Y0"].sample(frac=1, replace=True, random_state=rng.integers(1e9)).mean()
                - pre["Y0"].sample(frac=1, replace=True, random_state=rng.integers(1e9)).mean()
            )
            - (
                ctrl_post["Y0"].sample(n=5000, replace=True, random_state=rng.integers(1e9)).mean()
                - ctrl_pre["Y0"].sample(n=5000, replace=True, random_state=rng.integers(1e9)).mean()
            )
        )
    boots_w = np.asarray(boots_w)
    boots_y = np.asarray(boots_y)
    did_wait_ci = [float(x) for x in np.quantile(boots_w, [0.025, 0.975])]
    did_y_ci = [float(x) for x in np.quantile(boots_y, [0.025, 0.975])]
    print("DiD wait 95% CI:", did_wait_ci)
    print("DiD Y 95% CI:", did_y_ci)

    hyps = {
        "H1_底层人工42→中层人工30": (42.0, 30.0),
        "H2_底层人工42→机器17": (42.0, 17.0),
        "H3_慢区走动37→快区24": (37.0, 24.0),
        "H4_人工均值31→机器17": (31.0, 17.0),
        "H5_底层42→中层近端24": (42.0, 24.0),
    }

    post_ids = post["处方编号"].unique()
    item_sub = items[items["处方编号"].isin(post_ids)]
    grouped = {k: g for k, g in item_sub.groupby("处方编号", sort=False)}
    post_index = post.set_index("处方编号", drop=False)

    all_post = d[(d["日期_dt"] >= "2024-04-01") & (d["日期_dt"] < "2024-06-01")]
    frac = float(all_post["has"].mean())
    print("Apr-May Rx containing drug share:", round(frac, 5))

    def evaluate_hyp(pre_sec: float, post_sec: float, label: str) -> dict:
        save_per_target = (pre_sec - post_sec) / 60.0
        system_saved = frac * save_per_target
        rows = []
        for rid in post_ids:
            if rid not in grouped:
                continue
            row = post_index.loc[rid]
            g = grouped[rid]
            drugs = g["drugid"].astype(str).to_numpy()
            base = g["调配秒数_最终"].astype(float).to_numpy().copy()
            secs_pre = base.copy()
            secs_post = base.copy()
            for i, did in enumerate(drugs):
                if did == DRUG:
                    secs_pre[i] = pre_sec
                    secs_post[i] = post_sec
            disp_pre = secs_pre.sum() / 60.0
            disp_post = secs_post.sum() / 60.0
            d0_pre = minutes_to_quartile(disp_pre, thr["D0"])
            d0_post = minutes_to_quartile(disp_post, thr["D0"])
            w_min = float(row["叫号等待_分钟"])
            cap = (
                float(row["产能代理"])
                if "产能代理" in row.index and pd.notna(row["产能代理"])
                else None
            )
            # Factual ≈ post-adjustment; counterfactual = still pre-adjustment (slower system)
            w_pre = propagate_call_wait_minutes(
                w_min, -system_saved, int(row["Peak0"]), int(row["Load0"]), qmodel, capacity=cap
            )
            w0_pre = minutes_to_quartile(w_pre, thr["W0"])
            ey_post = expected_y(cpts, row, int(row["W0"]), d0_post)
            ey_pre = expected_y(cpts, row, w0_pre, d0_pre)
            ey_pre_fw = expected_y(cpts, row, int(row["W0"]), d0_pre)
            ey_post_fw = expected_y(cpts, row, int(row["W0"]), d0_post)
            py3_post = prob_long_wait(cpts, row, int(row["W0"]), d0_post)
            py3_pre = prob_long_wait(cpts, row, w0_pre, d0_pre)
            rows.append(
                {
                    "disp_saved": disp_pre - disp_post,
                    "d0_pre": d0_pre,
                    "d0_post": d0_post,
                    "w_pre": w_pre,
                    "w_obs": w_min,
                    "ey_pre": ey_pre,
                    "ey_post": ey_post,
                    "ey_pre_fw": ey_pre_fw,
                    "ey_post_fw": ey_post_fw,
                    "py3_pre": py3_pre,
                    "py3_post": py3_post,
                }
            )
        df = pd.DataFrame(rows)
        return {
            "hyp": label,
            "pre_sec": pre_sec,
            "post_sec": post_sec,
            "delta_sec": pre_sec - post_sec,
            "system_saved_min": round(system_saved, 6),
            "n": int(len(df)),
            "mean_disp_saved_min": round(float(df["disp_saved"].mean()), 4),
            "frac_D0_drop": round(float((df["d0_post"] < df["d0_pre"]).mean()), 4),
            "pred_EY_reduction_queue": round(float((df["ey_pre"] - df["ey_post"]).mean()), 4),
            "pred_EY_reduction_fixedW": round(
                float((df["ey_pre_fw"] - df["ey_post_fw"]).mean()), 4
            ),
            "pred_PQ4_reduction": round(float((df["py3_pre"] - df["py3_post"]).mean()), 4),
            "pred_call_wait_reduction_min": round(float((df["w_pre"] - df["w_obs"]).mean()), 4),
        }

    preds = []
    print("\n=== Model predictions ===")
    for name, (a, b) in hyps.items():
        r = evaluate_hyp(a, b, name)
        preds.append(r)
        print(r)

    # Match: observed improvement = -did (did is post-pre-ctrl; negative = improvement)
    obs_y_improve = -obs["did_Y"]
    obs_pq4_improve = -obs["did_PQ4"]
    obs_wait_improve = -obs["did_wait"]
    obs_call_improve = -obs["did_call"]

    ranking = []
    for r in preds:
        # Rank by absolute error of EY and P(Q4)
        err = abs(r["pred_EY_reduction_queue"] - obs_y_improve) + abs(
            r["pred_PQ4_reduction"] - obs_pq4_improve
        )
        ranking.append({**r, "abs_err_EY_PQ4": round(err, 4)})
    ranking = sorted(ranking, key=lambda x: x["abs_err_EY_PQ4"])

    out = {
        "drug": DRUG,
        "name": "罗沙司他胶囊 50mg*3 (爱瑞卓)",
        "current_slot": "F01 / 药包机 / 17s（调配表为当前状态）",
        "periods": {"pre": "2024-02", "transition": "2024-03", "post": "2024-04~05"},
        "observed": obs,
        "did_wait_ci95": did_wait_ci,
        "did_Y_ci95": did_y_ci,
        "observed_improvements": {
            "wait_min": round(obs_wait_improve, 4),
            "Y_level": round(obs_y_improve, 4),
            "P_Q4": round(obs_pq4_improve, 4),
            "call_wait_min": round(obs_call_improve, 4),
        },
        "model_predictions": preds,
        "best_matching_hypotheses": ranking[:3],
        "interpretation": {
            "note": "Historical bin seconds unknown; use assumed seconds for counterfactual; observed DiD detrends with other Rx",
            "best_hyp": ranking[0]["hyp"] if ranking else None,
        },
    }

    out_path = PHASE2 / "PLH012M_validation.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n=== Comparison summary ===")
    print(f"Observed DiD wait improvement: {obs_wait_improve:.3f} min  CI={did_wait_ci}")
    print(f"Observed DiD Y-level improvement: {obs_y_improve:.3f}  CI={[-did_y_ci[1], -did_y_ci[0]]}")
    print(f"Observed DiD P(Q4) improvement: {obs_pq4_improve:.3f}")
    print(f"Observed DiD call-wait improvement: {obs_call_improve:.3f} min")
    print("\nClosest hypotheses:")
    for r in ranking[:3]:
        print(
            f"  {r['hyp']}: pred_EY↓={r['pred_EY_reduction_queue']:.3f} "
            f"(obs {obs_y_improve:.3f}), pred_PQ4↓={r['pred_PQ4_reduction']:.3f} "
            f"(obs {obs_pq4_improve:.3f}), err={r['abs_err_EY_PQ4']:.3f}"
        )
    print("saved:", out_path)


if __name__ == "__main__":
    main()
