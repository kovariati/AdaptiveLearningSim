from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import estimate_empirical_bayes_hyperparameters as eb
import build_empirical_calibration_inputs as build
import run_taskenv as sim

FLOORS=(0.35,0.55,0.75)
POLICIES=(sim.REFERENCE_POLICY,'interleaved_median_item','balanced_mastery','maximum_skill_uncertainty','maximum_response_information_gain')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--skill-fits',type=Path,required=True)
    ap.add_argument('--selected-items',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--replicates',type=int,default=5)
    ap.add_argument('--learners',type=int,default=100)
    ap.add_argument('--orders',type=int,default=3)
    ap.add_argument('--root-seed',type=int,default=20260801)
    a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    fits=pd.read_csv(a.skill_fits);items=pd.read_csv(a.selected_items)
    rows=[]; cal_rows=[]
    for floor in FLOORS:
        obj=eb.estimate(fits,0.48,floor)
        cfg={k:v for k,v in obj.items() if k!='audit_rows'}
        skill=build.build_skill_shrinkage(fits,cfg)
        item=build.build_item_effects(items,skill,cfg)
        vdir=a.output_dir/f'floor_{floor:.2f}';vdir.mkdir(exist_ok=True)
        skill.to_csv(vdir/'input_skill_parameter_shrinkage.csv',index=False)
        item.to_csv(vdir/'input_item_effect_audit.csv',index=False)
        skill_guess=skill[skill.parameter.eq('guess')]
        cal_rows.append({'guess_boundary_se_floor':floor,'mean_shrunk_guess':skill_guess.shrunk_value.mean(),'min_shrunk_guess':skill_guess.shrunk_value.min(),'max_shrunk_guess':skill_guess.shrunk_value.max(),'sd_shrunk_guess':skill_guess.shrunk_value.std(ddof=1)})
        inputs=sim.load_inputs(vdir/'input_skill_parameter_shrinkage.csv',vdir/'input_item_effect_audit.csv',9,0.0,0.35,0.70)
        orders=sim.generate_skill_orders(inputs.n_skills,a.orders,a.root_seed)
        tables=sim.build_policy_tables(inputs,2049)
        for r in range(a.replicates):
            seed=a.root_seed+9000000+int(round(floor*100))*100000+r*100003
            for oid,order in enumerate(orders):
                for world in sim.WORLDS:
                    ref=sim.simulate_policy(world,sim.REFERENCE_POLICY,inputs,seed,a.learners,100,30.0,order,oid,policy_tables=tables)['delayed_latent'].mean()
                    for policy in POLICIES:
                        if policy==sim.REFERENCE_POLICY: benefit=0.0
                        else: benefit=float(sim.simulate_policy(world,policy,inputs,seed,a.learners,100,30.0,order,oid,policy_tables=tables)['delayed_latent'].mean()-ref)
                        rows.append({'guess_boundary_se_floor':floor,'replicate':r,'order_id':oid,'world_model':world,'policy':policy,'simulated_benefit':benefit})
    raw=pd.DataFrame(rows); raw.to_csv(a.output_dir/'guess_boundary_sensitivity_replicates.csv',index=False)
    draw=raw.groupby(['guess_boundary_se_floor','replicate','world_model','policy'],as_index=False)['simulated_benefit'].mean()
    summary=draw.groupby(['guess_boundary_se_floor','world_model','policy'],as_index=False).agg(mean_simulated_benefit=('simulated_benefit','mean'),sd_across_replicates=('simulated_benefit','std'))
    summary.to_csv(a.output_dir/'guess_boundary_sensitivity_summary.csv',index=False)
    pd.DataFrame(cal_rows).to_csv(a.output_dir/'guess_boundary_calibration_summary.csv',index=False)
    print('BOUNDARY-SENSITIVITY: PASS')
if __name__=='__main__': main()
