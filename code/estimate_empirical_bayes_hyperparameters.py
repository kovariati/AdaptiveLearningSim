from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logit

PROB_PARAMS = {
    'p_init':'se_p_init_approx', 'p_learn':'se_p_learn_approx',
    'slip':'se_slip_approx', 'guess':'se_guess_approx'
}

def transformed_values(fits: pd.DataFrame, parameter: str, guess_boundary: float):
    raw = fits[parameter].to_numpy(float)
    if parameter == 'lambda_per_day':
        y = np.log(np.maximum(raw, 1e-12))
        se = fits['se_lambda_per_day_approx'].to_numpy(float) / np.maximum(raw, 1e-12)
        boundary = np.zeros(len(raw), dtype=bool)
    else:
        y = logit(np.clip(raw, 1e-10, 1-1e-10))
        se = fits[PROB_PARAMS[parameter]].to_numpy(float) / np.maximum(raw*(1-raw), 1e-12)
        boundary = (raw >= guess_boundary) if parameter == 'guess' else np.zeros(len(raw), dtype=bool)
    return raw, y, se, boundary

def reml_normal_normal(y: np.ndarray, se: np.ndarray) -> tuple[float,float]:
    if len(y) < 3:
        raise ValueError('At least three non-boundary skills are required')
    def objective(tau2: float) -> float:
        v = se*se + tau2
        w = 1.0/v
        mu = float(np.sum(w*y)/np.sum(w))
        q = float(np.sum(w*(y-mu)**2))
        # Restricted negative log-likelihood up to an additive constant.
        return 0.5*(float(np.sum(np.log(v))) + float(np.log(np.sum(w))) + q)
    upper = max(float(np.var(y, ddof=1))*20.0, 1.0)
    res = minimize_scalar(objective, bounds=(1e-12, upper), method='bounded', options={'xatol':1e-14})
    tau2 = max(float(res.x), 1e-12)
    v = se*se + tau2
    w = 1.0/v
    mu = float(np.sum(w*y)/np.sum(w))
    return mu, tau2

def estimate(fits: pd.DataFrame, guess_boundary: float, boundary_se_floor: float) -> dict:
    fits = fits.loc[fits['model'].eq('BKT-F')].copy().sort_values('skill_id')
    params = {}
    audit=[]
    for parameter in ['p_init','p_learn','slip','guess','lambda_per_day']:
        raw,y,se,boundary = transformed_values(fits, parameter, guess_boundary)
        use = ~boundary
        mu,tau2 = reml_normal_normal(y[use], se[use])
        params[parameter]={'global_transformed_mean':mu,'between_skill_variance':tau2}
        audit.append({
            'parameter':parameter,'n_skills':len(y),'n_used_for_hyperparameter_fit':int(use.sum()),
            'n_boundary_excluded':int(boundary.sum()),'global_transformed_mean':mu,
            'between_skill_variance':tau2,'guess_boundary_threshold':guess_boundary if parameter=='guess' else np.nan,
            'boundary_transformed_se_floor':boundary_se_floor if parameter=='guess' else np.nan,
        })
    return {
      'method':'normal-normal empirical-Bayes shrinkage; REML hyperparameters on transformed scale; boundary-adjacent guess fits excluded from hyperparameter estimation and assigned a pre-specified transformed-SE floor during shrinkage',
      'parameters':params,
      'boundary_handling':{
        'guess_boundary_threshold':guess_boundary,
        'guess_boundary_transformed_se_floor':boundary_se_floor,
        'rationale':'BKT-F guess estimates at the optimizer upper boundary have unreliable local-Hessian standard errors. They are excluded from hyperparameter estimation and down-weighted during shrinkage. This is a benchmark calibration rule, not a claim of a standard universal EB procedure.'
      },
      'audit_rows':audit,
      'item_calibration':{
        'guess_endpoint_cap':0.47,
        'continuity_correction_successes':0.5,
        'continuity_correction_trials':1.0,
        'item_prior_variance':'interaction-weighted variance of raw shifts minus interaction-weighted mean squared item-shift standard error',
        'item_recentring':'interaction-times-shrinkage weighted mean raw shift',
        'holdout':'difficulty-stratified deterministic 20 percent by item identity within skill'
      }
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--skill-fits',type=Path,required=True)
    ap.add_argument('--out-json',type=Path,required=True)
    ap.add_argument('--out-audit-csv',type=Path,required=True)
    ap.add_argument('--guess-boundary',type=float,default=0.48)
    ap.add_argument('--guess-boundary-se-floor',type=float,default=0.55)
    a=ap.parse_args()
    obj=estimate(pd.read_csv(a.skill_fits),a.guess_boundary,a.guess_boundary_se_floor)
    a.out_json.parent.mkdir(parents=True,exist_ok=True)
    a.out_json.write_text(json.dumps({k:v for k,v in obj.items() if k!='audit_rows'},indent=2,sort_keys=True),encoding='utf-8')
    pd.DataFrame(obj['audit_rows']).to_csv(a.out_audit_csv,index=False)
    print('Wrote',a.out_json,'and',a.out_audit_csv)
if __name__=='__main__': main()
