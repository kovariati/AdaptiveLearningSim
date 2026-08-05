from __future__ import annotations

import argparse
import json
import math
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import run_taskenv as core

MAX_MI = "maximum_response_information_gain"
TWO_CORNER = "two_corner_robust_response_information_gain"
INTERLEAVED = "interleaved_median_item"
BLOCKED = core.REFERENCE_POLICY
KEY_POLICIES = (BLOCKED, INTERLEAVED, MAX_MI, TWO_CORNER)


def _interval(v: np.ndarray) -> tuple[float, float]:
    return core.simulation_replication_interval(np.asarray(v, dtype=float))


def observed_endpoint_tables(existing_results: Path, out: Path) -> None:
    rep = pd.read_csv(existing_results / "replicate_results_order_averaged.csv")
    rows = []
    for (world, policy), sub in rep.groupby(["world_model", "policy"], sort=True):
        vals = sub["simulated_benefit_delayed_observed"].to_numpy(float)
        lo, hi = _interval(vals)
        rows.append({
            "world_model": world,
            "policy": policy,
            "n_independent_replications": int(len(vals)),
            "delayed_observed_benefit_mean": float(vals.mean()),
            "simulation_interval_low": lo,
            "simulation_interval_high": hi,
            "positive_replication_fraction": float(np.mean(vals > 0)),
        })
    sm = pd.DataFrame(rows)
    sm.to_csv(out / "rq1_delayed_observed_policy_summary.csv", index=False)

    rank_rows = []
    for world, sub in sm[sm.policy.isin(core.FEASIBLE_POLICIES)].groupby("world_model"):
        s = sub.sort_values("delayed_observed_benefit_mean", ascending=False).reset_index(drop=True)
        for i, r in s.iterrows():
            rank_rows.append({"world_model": world, "policy": r.policy, "observed_score_rank": i + 1,
                              "delayed_observed_benefit_mean": r.delayed_observed_benefit_mean})
    ranks = pd.DataFrame(rank_rows)
    ranks.to_csv(out / "rq1_delayed_observed_rank_summary.csv", index=False)

    crows = []
    for world, sub in rep.groupby("world_model"):
        a = sub[sub.policy == MAX_MI].set_index("replicate")
        b = sub[sub.policy == TWO_CORNER].set_index("replicate")
        common = a.index.intersection(b.index)
        vals = a.loc[common, "delayed_observed"].to_numpy(float) - b.loc[common, "delayed_observed"].to_numpy(float)
        lo, hi = _interval(vals)
        crows.append({"world_model": world, "contrast": "maximum_response_MI_minus_two_corner",
                      "n_independent_replications": int(len(vals)), "mean_difference": float(vals.mean()),
                      "simulation_interval_low": lo, "simulation_interval_high": hi,
                      "positive_replication_fraction": float(np.mean(vals > 0))})
    pd.DataFrame(crows).to_csv(out / "rq1_delayed_observed_mi_pairwise_contrast.csv", index=False)

    # Replication-level rank frequencies for the common observed endpoint.
    rr = []
    for world, subw in rep[rep.policy.isin(core.FEASIBLE_POLICIES)].groupby("world_model"):
        counts = {p: {i: 0 for i in range(1, len(core.FEASIBLE_POLICIES) + 1)} for p in core.FEASIBLE_POLICIES}
        for rid, sr in subw.groupby("replicate"):
            ordered = sr.sort_values("simulated_benefit_delayed_observed", ascending=False).policy.tolist()
            for rank, pol in enumerate(ordered, 1): counts[pol][rank] += 1
        nrep = subw.replicate.nunique()
        for pol in core.FEASIBLE_POLICIES:
            rr.append({"world_model": world, "policy": pol, "n_replications": nrep,
                       "rank1_frequency": counts[pol][1] / nrep,
                       "top3_frequency": sum(counts[pol][i] for i in (1,2,3)) / nrep,
                       "mean_replication_rank": sum(i*counts[pol][i] for i in counts[pol]) / nrep})
    pd.DataFrame(rr).to_csv(out / "rq1_delayed_observed_replication_rank_frequencies.csv", index=False)

    # Primary common expected-response endpoint. This is the exact conditional
    # response expectation under the shared evaluation mapping and therefore
    # avoids an additional Bernoulli measurement layer.
    ref_exp = rep[rep.policy == BLOCKED][["world_model","replicate","delayed_expected"]].rename(columns={"delayed_expected":"ref_delayed_expected"})
    ex = rep.merge(ref_exp, on=["world_model","replicate"], validate="many_to_one")
    ex["expected_benefit"] = ex["delayed_expected"] - ex["ref_delayed_expected"]
    erows=[]
    for (world,policy),sub in ex.groupby(["world_model","policy"],sort=True):
        vals=sub.expected_benefit.to_numpy(float); lo,hi=_interval(vals)
        erows.append({"world_model":world,"policy":policy,"n_independent_replications":len(vals),
                      "delayed_expected_benefit_mean":float(vals.mean()),"simulation_interval_low":lo,"simulation_interval_high":hi,
                      "percentage_point_difference":float(100*vals.mean()),
                      "expected_additional_correct_per_100_items":float(100*vals.mean())})
    esm=pd.DataFrame(erows); esm.to_csv(out / "rq1_delayed_expected_policy_summary.csv",index=False)

    ec=[]
    for world,sub in ex.groupby("world_model"):
        a=sub[sub.policy==MAX_MI].set_index("replicate").expected_benefit
        b=sub[sub.policy==TWO_CORNER].set_index("replicate").expected_benefit
        vals=(a-b).dropna().to_numpy(float); lo,hi=_interval(vals)
        ec.append({"world_model":world,"contrast":"maximum_response_MI_minus_two_corner","n_independent_replications":len(vals),
                   "mean_difference":float(vals.mean()),"simulation_interval_low":lo,"simulation_interval_high":hi,
                   "percentage_point_difference":float(100*vals.mean()),"expected_additional_correct_per_100_items":float(100*vals.mean()),
                   "within_0.001_indifference_band":bool(abs(vals.mean())<=0.001),
                   "within_0.002_indifference_band":bool(abs(vals.mean())<=0.002)})
    pd.DataFrame(ec).to_csv(out / "rq1_delayed_expected_mi_pairwise_contrast.csv",index=False)

    rr_exp=[]
    for world,subw in ex[ex.policy.isin(core.FEASIBLE_POLICIES)].groupby("world_model"):
        nrep=subw.replicate.nunique(); counts={p:{i:0 for i in range(1,len(core.FEASIBLE_POLICIES)+1)} for p in core.FEASIBLE_POLICIES}
        for rid,sr in subw.groupby("replicate"):
            ordered=sr.sort_values("expected_benefit",ascending=False).policy.tolist()
            for rank,pol in enumerate(ordered,1): counts[pol][rank]+=1
        for pol in core.FEASIBLE_POLICIES:
            rr_exp.append({"world_model":world,"policy":pol,"n_replications":nrep,
                           "rank1_frequency":counts[pol][1]/nrep,"top3_frequency":sum(counts[pol][i] for i in (1,2,3))/nrep,
                           "mean_replication_rank":sum(i*counts[pol][i] for i in counts[pol])/nrep})
    pd.DataFrame(rr_exp).to_csv(out / "rq1_delayed_expected_replication_rank_frequencies.csv",index=False)

    # Challenge-world observed endpoint.
    ch = pd.read_csv(existing_results / "challenge_dependent_learning_order_replicates.csv")
    ref = ch[ch.policy == BLOCKED][["replicate", "order_id", "delayed_observed"]].rename(columns={"delayed_observed":"ref"})
    cp = ch.merge(ref, on=["replicate","order_id"], validate="many_to_one")
    cp["observed_benefit"] = cp.delayed_observed - cp.ref
    ravg = cp.groupby(["policy","replicate"], as_index=False).observed_benefit.mean()
    cr=[]
    for pol, s in ravg.groupby("policy"):
        vals=s.observed_benefit.to_numpy(float); lo,hi=_interval(vals)
        cr.append({"policy":pol,"n_independent_replications":len(vals),"delayed_observed_benefit_mean":vals.mean(),
                   "simulation_interval_low":lo,"simulation_interval_high":hi,"positive_replication_fraction":np.mean(vals>0)})
    pd.DataFrame(cr).to_csv(out / "challenge_delayed_observed_policy_summary.csv", index=False)
    a=ravg[ravg.policy==MAX_MI].set_index("replicate").observed_benefit
    b=ravg[ravg.policy==TWO_CORNER].set_index("replicate").observed_benefit
    vals=(a-b).dropna().to_numpy(float); lo,hi=_interval(vals)
    pd.DataFrame([{"contrast":"maximum_response_MI_minus_two_corner","n_independent_replications":len(vals),
                   "mean_difference":vals.mean(),"simulation_interval_low":lo,"simulation_interval_high":hi,
                   "positive_replication_fraction":np.mean(vals>0)}]).to_csv(out / "challenge_delayed_observed_mi_pairwise_contrast.csv", index=False)


def empirical_anchor_audit(reference_zip: Path, calibration_zip: Path, out: Path) -> None:
    with zipfile.ZipFile(calibration_zip) as z:
        agg = pd.read_csv(z.open("aggregate_metrics.csv"))
        gap = pd.read_csv(z.open("gap_metrics.csv"))
    with zipfile.ZipFile(reference_zip) as z:
        q = pd.read_csv(z.open("selected_skill_gap_sample_quantiles.csv"))

    test = agg[(agg.split == "test") & (agg.sample_class == "all")].copy()
    test.to_csv(out / "ednet_heldout_bkt_bktf_performance.csv", index=False)

    # Compact absolute-gain table, making the modest predictive increment explicit.
    piv = test.set_index("model")
    perf = pd.DataFrame([{
        "metric": "micro_weighted_log_loss", "BKT": float(piv.loc["BKT","micro_weighted_log_loss"]),
        "BKT_F": float(piv.loc["BKT-F","micro_weighted_log_loss"]),
        "BKT_F_minus_BKT": float(piv.loc["BKT-F","micro_weighted_log_loss"] - piv.loc["BKT","micro_weighted_log_loss"]),
    },{
        "metric": "micro_weighted_auc", "BKT": float(piv.loc["BKT","micro_weighted_auc"]),
        "BKT_F": float(piv.loc["BKT-F","micro_weighted_auc"]),
        "BKT_F_minus_BKT": float(piv.loc["BKT-F","micro_weighted_auc"] - piv.loc["BKT","micro_weighted_auc"]),
    },{
        "metric": "micro_weighted_brier", "BKT": float(piv.loc["BKT","micro_weighted_brier"]),
        "BKT_F": float(piv.loc["BKT-F","micro_weighted_brier"]),
        "BKT_F_minus_BKT": float(piv.loc["BKT-F","micro_weighted_brier"] - piv.loc["BKT","micro_weighted_brier"]),
    }])
    perf.to_csv(out / "ednet_heldout_bkt_bktf_absolute_difference.csv", index=False)

    # Held-out gap-bin predictive diagnostics for the empirically fitted models.
    gt = gap[gap.split == "test"].copy()
    grows=[]
    for (model, gb), s in gt.groupby(["model","gap_bin"]):
        w=s.n_events.to_numpy(float); n=w.sum()
        if n <= 0: continue
        grows.append({"model":model,"gap_bin":gb,"n_events":int(n),
                      "event_rate_weighted":float(np.average(s.event_rate, weights=w)),
                      "mean_prediction_weighted":float(np.average(s.mean_prediction, weights=w)),
                      "log_loss_weighted":float(np.average(s.log_loss, weights=w)),
                      "brier_weighted":float(np.average(s.brier, weights=w)),
                      "auc_weighted":float(np.average(s.auc.fillna(0.5), weights=w))})
    pd.DataFrame(grows).to_csv(out / "ednet_heldout_gap_bin_predictive_diagnostics.csv", index=False)

    q["gap_days"] = q.gap_ms / 86400000.0
    q.to_csv(out / "ednet_gap_quantiles_days.csv", index=False)
    qq=q.pivot(index="skill_id",columns="probability",values="gap_days")
    sup=[]
    for sid,r in qq.iterrows():
        sup.append({"skill_id":sid,"q90_gap_days":r.get(0.90,np.nan),"q95_gap_days":r.get(0.95,np.nan),"q99_gap_days":r.get(0.99,np.nan),
                    "q90_below_30d":bool(r.get(0.90,np.inf)<30),"q95_below_30d":bool(r.get(0.95,np.inf)<30),"q99_below_30d":bool(r.get(0.99,np.inf)<30)})
    sdf=pd.DataFrame(sup); sdf.to_csv(out / "ednet_30day_retention_support_by_skill.csv",index=False)
    pd.DataFrame([{
        "n_skills":len(sdf),"skills_q90_below_30d":int(sdf.q90_below_30d.sum()),
        "skills_q95_below_30d":int(sdf.q95_below_30d.sum()),"skills_q99_below_30d":int(sdf.q99_below_30d.sum()),
        "interpretation":"30 days lies in the upper empirical gap tail for most skills; it is not wholly out of support but is partly extrapolative."
    }]).to_csv(out / "ednet_30day_retention_support_summary.csv",index=False)

    validity = pd.DataFrame([
        {"world_model":"binary_bktf",
         "empirical_status":"benchmark world derived from an EdNet-fitted BKT-F precursor and subsequently modified by empirical-Bayes shrinkage and marginal-accuracy-anchored item shifts",
         "precursor_disjoint_learner_test_metrics":"available",
         "final_world_untouched_end_to_end_test":"not available",
         "reason":"the final benchmark parameterization includes shrinkage and item-shift construction that are not evaluated as one end-to-end model on a fully untouched learner sample",
         "equal_empirical_support_claimed":False},
        {"world_model":"continuous_latent_state_dynamical",
         "empirical_status":"designed moment-matched structural stress construction; not independently fitted to EdNet sequences",
         "precursor_disjoint_learner_test_metrics":"not applicable",
         "final_world_untouched_end_to_end_test":"not available",
         "reason":"designed stress construction",
         "equal_empirical_support_claimed":False},
        {"world_model":"four_state_semimarkov",
         "empirical_status":"designed moment-matched structural stress construction; not independently fitted to EdNet sequences",
         "precursor_disjoint_learner_test_metrics":"not applicable",
         "final_world_untouched_end_to_end_test":"not available",
         "reason":"designed stress construction",
         "equal_empirical_support_claimed":False},
    ])
    validity.to_csv(out / "base_world_empirical_validity_status.csv", index=False)


def _order_worker(payload):
    inputs, world, rep, orders, n_learners, n_steps, delayed_days, root_seed, grid = payload
    tables = core.build_policy_tables(inputs, grid)
    seed = root_seed + 73000 + rep * 100003
    rows=[]
    for oi, order in enumerate(orders):
        a = core.simulate_policy(world, MAX_MI, inputs, seed, n_learners, n_steps, delayed_days, order, oi, policy_tables=tables, policy_inputs=inputs)
        b = core.simulate_policy(world, TWO_CORNER, inputs, seed, n_learners, n_steps, delayed_days, order, oi, policy_tables=tables, policy_inputs=inputs)
        rows.append({"world_model":world,"replicate":rep,"order_id":oi,"seed":seed,
                     "max_mi_delayed_latent":float(a.delayed_latent.mean()),"two_corner_delayed_latent":float(b.delayed_latent.mean()),
                     "latent_contrast_max_minus_two":float(a.delayed_latent.mean()-b.delayed_latent.mean()),
                     "max_mi_delayed_observed":float(a.delayed_observed.mean()),"two_corner_delayed_observed":float(b.delayed_observed.mean()),
                     "observed_contrast_max_minus_two":float(a.delayed_observed.mean()-b.delayed_observed.mean())})
    return rows


def order_replication_factorial(inputs: core.Inputs, out: Path, args) -> None:
    orders = core.generate_skill_orders(inputs.n_skills, args.order_count, args.root_seed + 61000)
    tasks=[(inputs,w,r,orders,args.order_learners,args.practice_steps,args.delayed_days,args.root_seed,args.info_grid_size)
           for w in core.WORLDS for r in range(args.order_replicates)]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs=[ex.submit(_order_worker,t) for t in tasks]
        for j,f in enumerate(as_completed(futs),1):
            rows.extend(f.result())
            if j%5==0 or j==len(futs): print(f"order x replication tasks {j}/{len(futs)} completed",flush=True)
    d=pd.DataFrame(rows).sort_values(["world_model","replicate","order_id"])
    d.to_csv(out / "order20_x_replication_mi_contrasts.csv",index=False)
    srows=[]
    orows=[]
    for world, sw in d.groupby("world_model"):
        # Primary estimand averages each independent replication over all prespecified orders.
        for metric in ("latent_contrast_max_minus_two","observed_contrast_max_minus_two"):
            rv=sw.groupby("replicate")[metric].mean().to_numpy(float); lo,hi=_interval(rv)
            ov=sw.groupby("order_id")[metric].mean()
            srows.append({"world_model":world,"endpoint":metric,"n_independent_replications":len(rv),"n_orders":sw.order_id.nunique(),
                          "mean_contrast":float(rv.mean()),"simulation_interval_low":lo,"simulation_interval_high":hi,
                          "positive_replication_fraction":float(np.mean(rv>0)),"positive_order_fraction_after_replication_average":float(np.mean(ov.to_numpy()>0)),
                          "minimum_order_mean":float(ov.min()),"maximum_order_mean":float(ov.max())})
            for oid,val in ov.items():
                vv=sw[sw.order_id==oid][metric].to_numpy(float); l2,h2=_interval(vv)
                orows.append({"world_model":world,"endpoint":metric,"order_id":int(oid),"mean_contrast":float(val),
                              "simulation_interval_low":l2,"simulation_interval_high":h2,"positive_replication_fraction":float(np.mean(vv>0))})
    pd.DataFrame(srows).to_csv(out / "order20_x_replication_mi_contrast_summary.csv",index=False)
    pd.DataFrame(orows).to_csv(out / "order20_x_replication_by_order_summary.csv",index=False)


def _challenge_grid_worker(payload):
    inputs,cfg,rep,orders,n_learners,n_steps,delayed_days,root_seed,grid=payload
    tables=core.build_policy_tables(inputs,grid)
    seed=root_seed+83000+rep*100003+int(round(cfg.target_success*1000))*7+int(round(cfg.width*1000))*11+int(round(cfg.floor*1000))*13
    rows=[]
    for oi,order in enumerate(orders):
        vals={}
        for pol in KEY_POLICIES:
            df=core.simulate_policy("continuous_latent_trait",pol,inputs,seed,n_learners,n_steps,delayed_days,order,oi,
                                    policy_tables=tables,policy_inputs=inputs,learning_effect=cfg)
            vals[pol]={"latent":float(df.delayed_latent.mean()),"observed":float(df.delayed_observed.mean()),
                       "expected":float(df.delayed_expected.mean()),
                       "challenge":float(df.mean_challenge_score.mean()),"power":float(df.mean_effective_learning_power.mean()),
                       "effective_p":float(df.mean_effective_learning_probability.mean())}
        ref=vals[BLOCKED]
        for pol in KEY_POLICIES:
            rows.append({"target_success":cfg.target_success,"width":cfg.width,"floor":cfg.floor,"replicate":rep,"order_id":oi,
                         "policy":pol,"latent_benefit":vals[pol]["latent"]-ref["latent"],"observed_benefit":vals[pol]["observed"]-ref["observed"],
                         "expected_benefit":vals[pol]["expected"]-ref["expected"],
                         "mean_challenge_score":vals[pol]["challenge"],"mean_effective_learning_power":vals[pol]["power"],
                         "mean_effective_learning_probability":vals[pol]["effective_p"]})
    return rows


def challenge_grid(inputs: core.Inputs, out: Path, args) -> None:
    configs=[core.LearningEffectConfig(mode="challenge_zone",target_success=t,width=w,floor=f)
             for t in args.challenge_targets for w in args.challenge_widths for f in args.challenge_floors]
    orders=core.generate_skill_orders(inputs.n_skills,args.challenge_orders,args.root_seed+81000)
    tasks=[(inputs,cfg,r,orders,args.challenge_learners,args.practice_steps,args.delayed_days,args.root_seed,args.info_grid_size)
           for cfg in configs for r in range(args.challenge_replicates)]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs=[ex.submit(_challenge_grid_worker,t) for t in tasks]
        for j,f in enumerate(as_completed(futs),1):
            rows.extend(f.result())
            if j%10==0 or j==len(futs): print(f"challenge grid tasks {j}/{len(futs)} completed",flush=True)
    d=pd.DataFrame(rows)
    d.to_csv(out / "challenge_parameter_grid_replicates.csv",index=False)
    rep=d.groupby(["target_success","width","floor","policy","replicate"],as_index=False).agg(
        latent_benefit=("latent_benefit","mean"),observed_benefit=("observed_benefit","mean"),expected_benefit=("expected_benefit","mean"),
        mean_challenge_score=("mean_challenge_score","mean"),mean_effective_learning_power=("mean_effective_learning_power","mean"),
        mean_effective_learning_probability=("mean_effective_learning_probability","mean"))
    rows=[]
    for keys,s in rep.groupby(["target_success","width","floor","policy"]):
        t,w,f,p=keys
        lv=s.latent_benefit.to_numpy(float); ov=s.observed_benefit.to_numpy(float); ev=s.expected_benefit.to_numpy(float); ll,lh=_interval(lv); ol,oh=_interval(ov); el,eh=_interval(ev)
        rows.append({"target_success":t,"width":w,"floor":f,"policy":p,"n_independent_replications":len(s),
                     "latent_benefit_mean":lv.mean(),"latent_interval_low":ll,"latent_interval_high":lh,
                     "observed_benefit_mean":ov.mean(),"observed_interval_low":ol,"observed_interval_high":oh,
                     "expected_benefit_mean":ev.mean(),"expected_interval_low":el,"expected_interval_high":eh,
                     "mean_challenge_score":s.mean_challenge_score.mean(),"mean_effective_learning_power":s.mean_effective_learning_power.mean(),
                     "mean_effective_learning_probability":s.mean_effective_learning_probability.mean()})
    sm=pd.DataFrame(rows)
    sm.to_csv(out / "challenge_parameter_grid_summary.csv",index=False)
    # Explicit ordering region for the three substantive policies.
    rr=[]
    for keys,s in sm.groupby(["target_success","width","floor"]):
        x=s.set_index("policy")
        ranking=x.loc[[INTERLEAVED,MAX_MI,TWO_CORNER]].sort_values("latent_benefit_mean",ascending=False).index.tolist()
        rr.append({"target_success":keys[0],"width":keys[1],"floor":keys[2],"latent_rank_order":" > ".join(ranking),
                   "max_mi_minus_two_corner":float(x.loc[MAX_MI,"latent_benefit_mean"]-x.loc[TWO_CORNER,"latent_benefit_mean"]),
                   "best_policy":ranking[0],"both_information_negative_vs_blocked":bool((x.loc[MAX_MI,"latent_benefit_mean"]<0) and (x.loc[TWO_CORNER,"latent_benefit_mean"]<0))})
    pd.DataFrame(rr).to_csv(out / "challenge_parameter_grid_rank_region.csv",index=False)
    # Common expected-response endpoint is the primary cross-world estimand; the
    # observed Bernoulli score is retained as a measurement-layer sensitivity.
    rep_primary=rep.copy()
    rr_exp=[]
    for keys,s in sm.groupby(["target_success","width","floor"]):
        x=s.set_index("policy")
        candidate=[BLOCKED,INTERLEAVED,MAX_MI,TWO_CORNER]
        ranking=x.loc[candidate].sort_values("expected_benefit_mean",ascending=False).index.tolist()
        top, second = ranking[0], ranking[1]
        cell=rep_primary[(rep_primary.target_success==keys[0])&(rep_primary.width==keys[1])&(rep_primary.floor==keys[2])]
        a=cell[cell.policy==top].set_index("replicate").expected_benefit
        b=cell[cell.policy==second].set_index("replicate").expected_benefit
        dv=(a-b).dropna().to_numpy(float); dl,dh=_interval(dv)
        rr_exp.append({"target_success":keys[0],"width":keys[1],"floor":keys[2],
                       "expected_rank_order":" > ".join(ranking),"mean_best_minus_runner_up":float(dv.mean()),
                       "simulation_interval_low":dl,"simulation_interval_high":dh,
                       "resolved_best":bool(dl>0),"best_policy_by_mean":top,
                       "both_information_negative_vs_blocked":bool((x.loc[MAX_MI,"expected_benefit_mean"]<0) and (x.loc[TWO_CORNER,"expected_benefit_mean"]<0))})
    pd.DataFrame(rr_exp).to_csv(out / "challenge_parameter_grid_expected_rank_region.csv",index=False)

    rr_obs=[]
    for keys,s in sm.groupby(["target_success","width","floor"]):
        x=s.set_index("policy"); candidate=[BLOCKED,INTERLEAVED,MAX_MI,TWO_CORNER]
        ranking=x.loc[candidate].sort_values("observed_benefit_mean",ascending=False).index.tolist()
        rr_obs.append({"target_success":keys[0],"width":keys[1],"floor":keys[2],
                       "observed_rank_order":" > ".join(ranking),"best_policy_by_mean":ranking[0],
                       "both_information_negative_vs_blocked":bool((x.loc[MAX_MI,"observed_benefit_mean"]<0) and (x.loc[TWO_CORNER,"observed_benefit_mean"]<0))})
    pd.DataFrame(rr_obs).to_csv(out / "challenge_parameter_grid_observed_rank_region.csv",index=False)



def _retention_worker(payload):
    inputs, world, horizon, rep, orders, n_learners, n_steps, root_seed, grid = payload
    tables = core.build_policy_tables(inputs, grid)
    seed = root_seed + 91000 + rep * 100003 + int(round(horizon * 100)) * 17 + list(core.WORLDS).index(world) * 1009
    rows = []
    for oi, order in enumerate(orders):
        vals = {}
        for pol in KEY_POLICIES:
            df = core.simulate_policy(
                world, pol, inputs, seed, n_learners, n_steps, horizon, order, oi,
                policy_tables=tables, policy_inputs=inputs,
            )
            vals[pol] = {
                "latent": float(df.delayed_latent.mean()),
                "observed": float(df.delayed_observed.mean()),
            }
        ref = vals[BLOCKED]
        for pol in KEY_POLICIES:
            rows.append({
                "world_model": world,
                "retention_days": float(horizon),
                "replicate": int(rep),
                "order_id": int(oi),
                "policy": pol,
                "latent_benefit": vals[pol]["latent"] - ref["latent"],
                "observed_benefit": vals[pol]["observed"] - ref["observed"],
            })
    return rows


def retention_horizon_sensitivity(inputs: core.Inputs, out: Path, args) -> None:
    orders = core.generate_skill_orders(inputs.n_skills, args.retention_orders, args.root_seed + 90000)
    tasks = [
        (inputs, world, horizon, rep, orders, args.retention_learners, args.practice_steps, args.root_seed, args.info_grid_size)
        for world in core.WORLDS
        for horizon in args.retention_horizons
        for rep in range(args.retention_replicates)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_retention_worker, t) for t in tasks]
        for j, f in enumerate(as_completed(futs), 1):
            rows.extend(f.result())
            if j % 10 == 0 or j == len(futs):
                print(f"retention horizon tasks {j}/{len(futs)} completed", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out / "retention_horizon_sensitivity_replicates.csv", index=False)
    rep = d.groupby(["world_model", "retention_days", "policy", "replicate"], as_index=False).agg(
        latent_benefit=("latent_benefit", "mean"), observed_benefit=("observed_benefit", "mean")
    )
    srows = []
    for keys, s in rep.groupby(["world_model", "retention_days", "policy"]):
        world, horizon, policy = keys
        lv = s.latent_benefit.to_numpy(float)
        ov = s.observed_benefit.to_numpy(float)
        ll, lh = _interval(lv)
        ol, oh = _interval(ov)
        srows.append({
            "world_model": world,
            "retention_days": float(horizon),
            "policy": policy,
            "n_independent_replications": len(s),
            "latent_benefit_mean": float(lv.mean()),
            "latent_interval_low": ll,
            "latent_interval_high": lh,
            "observed_benefit_mean": float(ov.mean()),
            "observed_interval_low": ol,
            "observed_interval_high": oh,
        })
    pd.DataFrame(srows).to_csv(out / "retention_horizon_sensitivity_summary.csv", index=False)

def scale_grid_cleanup(existing_results: Path, out: Path) -> None:
    d=pd.read_csv(existing_results / "parameter_stress_scale_grid_summary.csv")
    x=d[d.stress_scale != 1.0].copy()
    keep=[c for c in ["analysis_mode","world_model","policy","n_parameter_draws","stress_scale","grid_role","reference_policy","orders_per_draw","learners_per_order","mean_over_draws","positive_draw_fraction"] if c in x.columns]
    # Add range from raw files where available; do not report q025/q975 for n=20.
    rows=[]
    for _,r in x[keep].iterrows():
        fn=existing_results / f"parameter_stress_scale_{r.stress_scale:.1f}_draw_order_results.csv"
        mn=mx=math.nan
        if fn.exists():
            raw=pd.read_csv(fn)
            s=raw[(raw.world_model==r.world_model)&(raw.policy==r.policy)]
            if len(s):
                # average orders within draw first; choose the released reference-specific column.
                value_col = "benefit_vs_interleaved" if r.reference_policy == "interleaved_median_item" else "benefit_vs_blocked"
                v=s.groupby("parameter_draw")[value_col].mean()
                mn=float(v.min()); mx=float(v.max())
        row=r.to_dict(); row["minimum_draw_mean"]=mn; row["maximum_draw_mean"]=mx
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "parameter_stress_scale_exploratory_summary_no_tail.csv",index=False)



def _schedule_worker(payload):
    inputs, world, budget, items_per_day, rep, orders, n_learners, delayed_days, root_seed, grid = payload
    tables=core.build_policy_tables(inputs,grid)
    seed=root_seed+131000+rep*100003+budget*37+items_per_day*101+list(core.WORLDS).index(world)*1009
    rows=[]
    for oi,order in enumerate(orders):
        vals={}
        for pol in KEY_POLICIES:
            df=core.simulate_policy(world,pol,inputs,seed,n_learners,budget,delayed_days,order,oi,
                                    items_per_day=items_per_day,within_minutes=5.0,
                                    policy_tables=tables,policy_inputs=inputs)
            vals[pol]={"expected":float(df.delayed_expected.mean()),"observed":float(df.delayed_observed.mean())}
        ref=vals[BLOCKED]
        for pol in KEY_POLICIES:
            rows.append({"world_model":world,"practice_budget":budget,"items_per_day":items_per_day,
                         "replicate":rep,"order_id":oi,"policy":pol,
                         "expected_benefit":vals[pol]["expected"]-ref["expected"],
                         "observed_benefit":vals[pol]["observed"]-ref["observed"]})
    return rows


def practice_schedule_sensitivity(inputs: core.Inputs, out: Path, args) -> None:
    orders=core.generate_skill_orders(inputs.n_skills,args.schedule_orders,args.root_seed+129000)
    tasks=[(inputs,w,b,d,r,orders,args.schedule_learners,args.delayed_days,args.root_seed,args.info_grid_size)
           for w in core.WORLDS for b in args.schedule_budgets for d in args.schedule_items_per_day
           for r in range(args.schedule_replicates)]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs=[ex.submit(_schedule_worker,t) for t in tasks]
        for j,f in enumerate(as_completed(futs),1):
            rows.extend(f.result())
            if j%10==0 or j==len(futs): print(f"schedule tasks {j}/{len(futs)} completed",flush=True)
    d=pd.DataFrame(rows); d.to_csv(out/'practice_schedule_sensitivity_replicates.csv',index=False)
    rep=d.groupby(['world_model','practice_budget','items_per_day','policy','replicate'],as_index=False).agg(
        expected_benefit=('expected_benefit','mean'),observed_benefit=('observed_benefit','mean'))
    sm=[]; ctr=[]
    for keys,g in rep.groupby(['world_model','practice_budget','items_per_day','policy']):
        w,b,ipd,p=keys; v=g.expected_benefit.to_numpy(float); lo,hi=_interval(v)
        sm.append({'world_model':w,'practice_budget':b,'items_per_day':ipd,'policy':p,'n_independent_replications':len(v),
                   'expected_benefit_mean':float(v.mean()),'simulation_interval_low':lo,'simulation_interval_high':hi})
    for keys,g in rep.groupby(['world_model','practice_budget','items_per_day']):
        w,b,ipd=keys; a=g[g.policy==MAX_MI].set_index('replicate').expected_benefit; z=g[g.policy==TWO_CORNER].set_index('replicate').expected_benefit
        v=(a-z).dropna().to_numpy(float); lo,hi=_interval(v)
        ctr.append({'world_model':w,'practice_budget':b,'items_per_day':ipd,'contrast':'maximum_response_MI_minus_two_corner',
                    'mean_difference':float(v.mean()),'simulation_interval_low':lo,'simulation_interval_high':hi,
                    'positive_replication_fraction':float(np.mean(v>0))})
    pd.DataFrame(sm).to_csv(out/'practice_schedule_sensitivity_summary.csv',index=False)
    pd.DataFrame(ctr).to_csv(out/'practice_schedule_mi_contrast.csv',index=False)


def _normalization_worker(payload):
    inputs,norm,rep,orders,n_learners,n_steps,delayed_days,root_seed,grid=payload
    cfg=core.LearningEffectConfig(mode='challenge_zone',target_success=0.70,width=0.18,floor=0.25,normalization=norm)
    tables=core.build_policy_tables(inputs,grid)
    seed=root_seed+151000+rep*100003+(0 if norm=='uniform_bins' else 7919)
    rows=[]
    for oi,order in enumerate(orders):
        vals={}
        for pol in KEY_POLICIES:
            df=core.simulate_policy('continuous_latent_trait',pol,inputs,seed,n_learners,n_steps,delayed_days,order,oi,
                                    policy_tables=tables,policy_inputs=inputs,learning_effect=cfg)
            vals[pol]=float(df.delayed_expected.mean())
        ref=vals[BLOCKED]
        for pol in KEY_POLICIES:
            rows.append({'normalization':norm,'replicate':rep,'order_id':oi,'policy':pol,
                         'expected_benefit':vals[pol]-ref})
    return rows


def challenge_normalization_sensitivity(inputs: core.Inputs, out: Path, args) -> None:
    orders=core.generate_skill_orders(inputs.n_skills,args.normalization_orders,args.root_seed+149000)
    tasks=[(inputs,norm,r,orders,args.normalization_learners,args.practice_steps,args.delayed_days,args.root_seed,args.info_grid_size)
           for norm in ['uniform_bins','empirical_frequency'] for r in range(args.normalization_replicates)]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs=[ex.submit(_normalization_worker,t) for t in tasks]
        for f in as_completed(futs): rows.extend(f.result())
    d=pd.DataFrame(rows); d.to_csv(out/'challenge_normalization_sensitivity_replicates.csv',index=False)
    rep=d.groupby(['normalization','policy','replicate'],as_index=False).expected_benefit.mean()
    sm=[]
    for (norm,p),g in rep.groupby(['normalization','policy']):
        v=g.expected_benefit.to_numpy(float); lo,hi=_interval(v)
        sm.append({'normalization':norm,'policy':p,'n_independent_replications':len(v),'expected_benefit_mean':float(v.mean()),
                   'simulation_interval_low':lo,'simulation_interval_high':hi,'positive_replication_fraction':float(np.mean(v>0))})
    pd.DataFrame(sm).to_csv(out/'challenge_normalization_sensitivity_summary.csv',index=False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--skill-parameters',type=Path,required=True)
    ap.add_argument('--item-effects',type=Path,required=True)
    ap.add_argument('--existing-results',type=Path,required=True)
    ap.add_argument('--reference-bundle',type=Path,required=True)
    ap.add_argument('--calibration-bundle',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--root-seed',type=int,default=20260801)
    ap.add_argument('--practice-steps',type=int,default=100)
    ap.add_argument('--delayed-days',type=float,default=30.0)
    ap.add_argument('--info-grid-size',type=int,default=4097)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--order-count',type=int,default=20)
    ap.add_argument('--order-replicates',type=int,default=20)
    ap.add_argument('--order-learners',type=int,default=200)
    ap.add_argument('--challenge-replicates',type=int,default=20)
    ap.add_argument('--challenge-learners',type=int,default=150)
    ap.add_argument('--challenge-orders',type=int,default=2)
    ap.add_argument('--challenge-targets',type=float,nargs='+',default=[0.60,0.70,0.80])
    ap.add_argument('--challenge-widths',type=float,nargs='+',default=[0.12,0.18,0.25])
    ap.add_argument('--challenge-floors',type=float,nargs='+',default=[0.10,0.25,0.50])
    ap.add_argument('--retention-horizons',type=float,nargs='+',default=[7.0,14.0,30.0])
    ap.add_argument('--retention-replicates',type=int,default=10)
    ap.add_argument('--retention-learners',type=int,default=150)
    ap.add_argument('--retention-orders',type=int,default=3)
    ap.add_argument('--schedule-budgets',type=int,nargs='+',default=[50,100,200])
    ap.add_argument('--schedule-items-per-day',type=int,nargs='+',default=[1,5,10])
    ap.add_argument('--schedule-replicates',type=int,default=10)
    ap.add_argument('--schedule-learners',type=int,default=120)
    ap.add_argument('--schedule-orders',type=int,default=3)
    ap.add_argument('--normalization-replicates',type=int,default=20)
    ap.add_argument('--normalization-learners',type=int,default=150)
    ap.add_argument('--normalization-orders',type=int,default=2)
    ap.add_argument('--skip-order',action='store_true')
    ap.add_argument('--skip-challenge',action='store_true')
    ap.add_argument('--skip-retention',action='store_true')
    ap.add_argument('--skip-schedule',action='store_true')
    ap.add_argument('--skip-normalization',action='store_true')
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    inputs=core.load_inputs(a.skill_parameters,a.item_effects)
    observed_endpoint_tables(a.existing_results,a.output_dir)
    empirical_anchor_audit(a.reference_bundle,a.calibration_bundle,a.output_dir)
    scale_grid_cleanup(a.existing_results,a.output_dir)
    if not a.skip_order: order_replication_factorial(inputs,a.output_dir,a)
    if not a.skip_challenge: challenge_grid(inputs,a.output_dir,a)
    if not a.skip_retention: retention_horizon_sensitivity(inputs,a.output_dir,a)
    if not a.skip_schedule: practice_schedule_sensitivity(inputs,a.output_dir,a)
    if not a.skip_normalization: challenge_normalization_sensitivity(inputs,a.output_dir,a)
    manifest={"analysis":"extended_sensitivity","root_seed":a.root_seed,"order_count":a.order_count,"order_replicates":a.order_replicates,
              "order_learners":a.order_learners,"challenge_replicates":a.challenge_replicates,"challenge_learners":a.challenge_learners,
              "challenge_orders":a.challenge_orders,"challenge_targets":a.challenge_targets,"challenge_widths":a.challenge_widths,
              "challenge_floors":a.challenge_floors,"retention_horizons":a.retention_horizons,"retention_replicates":a.retention_replicates,
              "retention_learners":a.retention_learners,"retention_orders":a.retention_orders}
    (a.output_dir/'extended_sensitivity_analysis_manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
