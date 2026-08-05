from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd
import run_taskenv as m

def subset_skill(inputs:m.Inputs,k:int)->m.Inputs:
    return replace(inputs,
        skill_ids=inputs.skill_ids[k:k+1],p_init=inputs.p_init[k:k+1],p_learn=inputs.p_learn[k:k+1],
        slip=inputs.slip[k:k+1],guess=inputs.guess[k:k+1],lam=inputs.lam[k:k+1],
        p_learn_base=inputs.p_learn_base[k:k+1],lam_base=inputs.lam_base[k:k+1],
        practice_item_shifts=inputs.practice_item_shifts[k:k+1],practice_item_counts=inputs.practice_item_counts[k:k+1],
        practice_item_shift_se=inputs.practice_item_shift_se[k:k+1],test_item_shifts=inputs.test_item_shifts[k:k+1],
        test_item_counts=inputs.test_item_counts[k:k+1],test_item_shift_se=inputs.test_item_shift_se[k:k+1],
        parameter_transformed_se={name:arr[k:k+1] for name,arr in inputs.parameter_transformed_se.items()},
        holdout_manifest=inputs.holdout_manifest[inputs.holdout_manifest.skill_id.eq(inputs.skill_ids[k])].copy())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--skill-parameters',type=Path,required=True)
    ap.add_argument('--item-effects',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--learners',type=int,default=300000)
    ap.add_argument('--sigma-forgetting',type=float,nargs='+',default=[0.20,0.35,0.50])
    a=ap.parse_args()
    all_rows=[]
    for sigma_forgetting in a.sigma_forgetting:
        full=m.load_inputs(a.skill_parameters,a.item_effects,9,0.0,float(sigma_forgetting),0.70)
        for k,skill_id in enumerate(full.skill_ids):
            x=subset_skill(full,k);n=a.learners;ids=np.arange(n,dtype=np.int64);rr=np.arange(n);sk=np.zeros(n,dtype=np.int32);it=np.zeros(n,dtype=np.int32)
            for wi,world in enumerate(m.WORLDS):
                seed=20260801+int(round(sigma_forgetting*1000))*100003+int(skill_id)*1009+wi*17
                ws=m.init_world(world,x,n,seed,ids);before=m.world_latent(world,ws)[:,0].copy();m.apply_learning(world,ws,rr,sk,it,seed,1,ids);after=m.world_latent(world,ws)[:,0]
                gain=float(np.mean(after-before));gain_target=float((1-x.p_init[0])*x.p_learn[0])
                ws2=m.init_world(world,x,n,seed+99991,ids);m.advance_all_to_time(world,ws2,30.0,seed+99991,77,ids);ret=float(m.world_latent(world,ws2)[:,0].mean());ret_target=float(x.p_init[0]*np.exp(-x.lam[0]*30.0))
                all_rows.append({'sigma_forgetting':float(sigma_forgetting),'skill_id':int(skill_id),'world_model':world,'mc_learners':n,'actual_one_step_gain_mc':gain,'gain_target':gain_target,'gain_abs_error_mc':abs(gain-gain_target),'actual_retention_30d_mc':ret,'retention_target':ret_target,'retention_abs_error_mc':abs(ret-ret_target)})
    d=pd.DataFrame(all_rows);a.output.parent.mkdir(parents=True,exist_ok=True);d.to_csv(a.output,index=False)
    summary=d.groupby(['sigma_forgetting','world_model']).agg(max_gain_abs_error_mc=('gain_abs_error_mc','max'),mean_gain_abs_error_mc=('gain_abs_error_mc','mean'),max_retention_abs_error_mc=('retention_abs_error_mc','max'),mean_retention_abs_error_mc=('retention_abs_error_mc','mean')).reset_index()
    summary.to_csv(a.output.with_name('transition_mc_validation_summary.csv'),index=False)
    print(summary.to_string(index=False)); print('RUNTIME-TRANSITION-MC-VALIDATION: COMPLETE')
if __name__=='__main__': main()
