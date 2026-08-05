from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd
import run_taskenv as core


def bin_audit(item_path: Path, n_bins: int = 9) -> pd.DataFrame:
    item = pd.read_csv(item_path)
    item = item[item['eligible_item'].astype(bool)].copy()
    rows=[]
    for skill in sorted(item.skill_id.astype(int).unique()):
        sub=item[item.skill_id.astype(int).eq(skill)].sort_values(['shrunk_item_shift','question_id']).reset_index(drop=True)
        test_mask=core._stable_item_holdout(int(skill),sub['question_id'])
        for split,ss in [('practice',sub.loc[~test_mask].copy()),('test',sub.loc[test_mask].copy())]:
            ss=ss.sort_values(['shrunk_item_shift','interaction_rows','question_id'],ascending=[True,False,True]).reset_index(drop=True)
            labels=np.floor(np.arange(len(ss))*n_bins/len(ss)).astype(int); labels=np.minimum(labels,n_bins-1)
            for b in range(n_bins):
                sb=ss.iloc[np.flatnonzero(labels==b)]
                w=np.sqrt(np.maximum(sb.interaction_rows.to_numpy(float),1.0))
                raw_se=np.maximum(sb.item_shift_se.to_numpy(float),1e-12)
                prior=np.maximum(sb.item_prior_variance.to_numpy(float),1e-12)
                shrink=prior/(prior+raw_se**2)
                posterior_sd=raw_se*np.sqrt(shrink)
                weighted_rms_raw_se=math.sqrt(float(np.average(raw_se**2,weights=w)))
                weighted_mean_raw_se=math.sqrt(float(np.sum((w**2)*(raw_se**2))))/float(np.sum(w))
                weighted_mean_eb_posterior_sd=math.sqrt(float(np.sum((w**2)*(posterior_sd**2))))/float(np.sum(w))
                rows.append({
                    'skill_id':int(skill),'split':split,'bin':b,'n_items':len(sb),
                    'weighted_rms_raw_item_se_stress_amplitude':weighted_rms_raw_se,
                    'weighted_mean_raw_item_se_reference':weighted_mean_raw_se,
                    'weighted_mean_eb_posterior_sd_reference':weighted_mean_eb_posterior_sd,
                    'rms_to_weighted_mean_raw_se_ratio':weighted_rms_raw_se/weighted_mean_raw_se,
                    'rms_to_weighted_mean_eb_posterior_sd_ratio':weighted_rms_raw_se/weighted_mean_eb_posterior_sd,
                })
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--item-effects',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--bins',type=int,default=9); a=ap.parse_args()
    d=bin_audit(a.item_effects,a.bins); a.output.parent.mkdir(parents=True,exist_ok=True); d.to_csv(a.output,index=False)
    x=d.rms_to_weighted_mean_raw_se_ratio.to_numpy(float)
    y=d.rms_to_weighted_mean_eb_posterior_sd_ratio.to_numpy(float)
    s=pd.DataFrame([
        {'comparison':'legacy_RMS_stress / weighted-mean raw-SE','n_cells':len(x),'mean_ratio':x.mean(),'median_ratio':np.median(x),'q95_ratio':np.quantile(x,.95),'max_ratio':x.max()},
        {'comparison':'legacy_RMS_stress / weighted-mean EB-posterior-SD','n_cells':len(y),'mean_ratio':y.mean(),'median_ratio':np.median(y),'q95_ratio':np.quantile(y,.95),'max_ratio':y.max()},
    ])
    s.to_csv(a.output.with_name('item_bin_uncertainty_audit_summary.csv'),index=False)
    print(s.to_string(index=False))
if __name__=='__main__': main()
