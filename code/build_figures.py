from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'figures'
WORK = ROOT
NEW = ROOT / 'results'

WORLD_LABEL = {
    'binary_bktf': 'Binary BKT-F',
    'continuous_latent_trait': 'Continuous latent-state',
    'four_state_semimarkov': 'Four-state semi-Markov',
}
POLICY_LABEL = {
    'interleaved_median_item': 'Interleaved curriculum',
    'random_skill_item_bin': 'Random skill-item bin',
    'balanced_mastery': 'Balanced mastery',
    'least_mastery_target_item': 'Least mastery',
    'maximum_skill_uncertainty': 'Maximum skill uncertainty',
    'maximum_response_information_gain': 'Maximum response MI',
    'two_corner_robust_response_information_gain': 'Two-corner response MI',
    'target_success_070': 'Target success 0.70',
}
WORLDS = list(WORLD_LABEL)
POLICIES = list(POLICY_LABEL)

def wrap(s, n=32):
    return '\n'.join(textwrap.wrap(s, width=n))

def architecture():
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    ax.set_axis_off()
    boxes = [
        (0.03, 0.58, 0.27, 0.29, 'Empirical anchor',
         'EdNet KT1 single-skill sequences\nPrecursor BKT/BKT-F calibration\nEmpirical-Bayes shrinkage\nMarginal-accuracy item shifts'),
        (0.365, 0.58, 0.27, 0.29, 'Learner worlds',
         'Derived binary BKT-F benchmark\nContinuous latent-state construction\nFour-state semi-Markov construction\nChallenge-dependent learning extension'),
        (0.70, 0.58, 0.27, 0.29, 'Policy evaluation',
         'Frozen BKT belief estimator\nFeasible sequencing policies\nShared response mapping\nExpected-response primary endpoint'),
        (0.115, 0.12, 0.31, 0.27, 'Experimental controls',
         'Practice/evaluation action-identity split\nAbsolute-time synchronization\nPaired common random numbers\nCurriculum-order and schedule sensitivities'),
        (0.575, 0.12, 0.31, 0.27, 'Robustness assessment',
         'Simulation-replication uncertainty\nStructural and parameter sensitivity\nShared expected-response endpoint\nScope for prospective validation'),
    ]
    for x, y, w, h, title, body in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012', fill=False, linewidth=1.3)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h - 0.055, title, ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(x + 0.018, y + h - 0.105, body, ha='left', va='top', fontsize=9.5, linespacing=1.25)
    arrows = [
        ((0.30,0.725),(0.365,0.725)), ((0.635,0.725),(0.70,0.725)),
        ((0.50,0.58),(0.31,0.39)), ((0.835,0.58),(0.73,0.39)),
        ((0.425,0.255),(0.575,0.255))
    ]
    for start,end in arrows:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=14,linewidth=1.2))
    ax.set_title('AdaptiveLearningSim: reproducible policy-evaluation benchmark', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT/'Fig1_architecture.png', dpi=240, bbox_inches='tight')
    plt.close(fig)

def response_diagnostics():
    d = pd.read_csv(ROOT/'results'/'response_moment_diagnostics.csv')
    g = d.groupby(['world_model','split'], as_index=False)['monte_carlo_abs_error'].max()
    fig, ax = plt.subplots(figsize=(7.6,4.8))
    x=np.arange(len(WORLDS)); width=0.36
    for j, split in enumerate(['practice','test']):
        vals=[g[(g.world_model==w)&(g.split==split)].monte_carlo_abs_error.iloc[0] for w in WORLDS]
        label='Practice actions' if split=='practice' else 'Evaluation actions'
        ax.bar(x+(j-0.5)*width, vals, width=width, label=label)
    ax.set_xticks(x, [WORLD_LABEL[w] for w in WORLDS], rotation=12, ha='right')
    ax.set_ylabel('Maximum Monte Carlo response-moment error')
    ax.set_title('Initial response-target diagnostics')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT/'Fig2_response_moment_diagnostics.png', dpi=240, bbox_inches='tight')
    plt.close(fig)

def benefit_heatmap():
    d=pd.read_csv(NEW/'rq1_delayed_expected_policy_summary.csv')
    d=d[d.policy.isin(POLICIES)]
    M=np.array([[d[(d.policy==p)&(d.world_model==w)].delayed_expected_benefit_mean.iloc[0] for w in WORLDS] for p in POLICIES])
    fig,ax=plt.subplots(figsize=(8.2,5.8))
    im=ax.imshow(M,aspect='auto')
    ax.set_xticks(range(3), [WORLD_LABEL[w] for w in WORLDS], rotation=18, ha='right')
    ax.set_yticks(range(len(POLICIES)), [POLICY_LABEL[p] for p in POLICIES])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j,i,f'{M[i,j]:+.4f}',ha='center',va='center',fontsize=7.5)
    ax.set_title('Delayed expected-response benefit relative to blocked curriculum')
    fig.colorbar(im,ax=ax,label='Expected-response benefit')
    fig.subplots_adjust(left=0.30, right=0.90, bottom=0.24, top=0.90)
    fig.savefig(OUT/'Fig3_expected_policy_benefit_heatmap.png', dpi=240, bbox_inches='tight')
    plt.close(fig)

def rank_trajectories():
    d=pd.read_csv(NEW/'rq1_delayed_expected_policy_summary.csv')
    d=d[d.policy.isin(POLICIES)].copy()
    d['rank']=d.groupby('world_model')['delayed_expected_benefit_mean'].rank(method='min',ascending=False).astype(int)
    fig,ax=plt.subplots(figsize=(8.4,5.8))
    x=np.arange(3)
    for p in POLICIES:
        y=[int(d[(d.policy==p)&(d.world_model==w)].iloc[0]['rank']) for w in WORLDS]
        ax.plot(x,y,marker='o',linewidth=1.2,label=POLICY_LABEL[p])
    ax.set_xticks(x,[WORLD_LABEL[w] for w in WORLDS])
    ax.set_ylabel('Mean policy rank (1 = highest expected benefit)')
    ax.invert_yaxis(); ax.set_ylim(8.5,0.5)
    ax.set_title('Policy ranks on the delayed expected-response endpoint')
    ax.legend(fontsize=7.5,ncol=2,loc='upper center',bbox_to_anchor=(0.5,-0.16))
    fig.tight_layout()
    fig.savefig(OUT/'Fig4_expected_policy_rank_trajectories.png',dpi=240,bbox_inches='tight')
    plt.close(fig)

def order_factorial():
    conf_obs=pd.read_csv(ROOT/'results'/'rq1_delayed_observed_mi_pairwise_contrast.csv')
    expd=pd.read_csv(ROOT/'results'/'order20_x_replication_mi_contrast_summary.csv')
    latent_five={
        'binary_bktf':(0.001667,0.000435,0.002899),
        'continuous_latent_trait':(-0.000166,-0.000331,-0.000001),
        'four_state_semimarkov':(-0.000530,-0.001558,0.000498)}
    rows=[]
    for w in WORLDS:
        r=conf_obs[conf_obs.world_model==w].iloc[0]
        rows.append((w,'Observed','Five-order design',r.mean_difference,r.simulation_interval_low,r.simulation_interval_high))
        e=expd[(expd.world_model==w)&(expd.endpoint=='observed_contrast_max_minus_two')].iloc[0]
        rows.append((w,'Observed','20-order × 20-replication',e.mean_contrast,e.simulation_interval_low,e.simulation_interval_high))
        m=latent_five[w]
        rows.append((w,'Latent','Five-order design',*m))
        e=expd[(expd.world_model==w)&(expd.endpoint=='latent_contrast_max_minus_two')].iloc[0]
        rows.append((w,'Latent','20-order × 20-replication',e.mean_contrast,e.simulation_interval_low,e.simulation_interval_high))
    F=pd.DataFrame(rows,columns=['world','endpoint','design','mean','lo','hi'])
    pairs=[(w,ep) for w in WORLDS for ep in ['Observed','Latent']]
    labels=[f'{WORLD_LABEL[w]} — {ep}' for w,ep in pairs]
    fig,ax=plt.subplots(figsize=(8.2,5.6))
    for di,design in enumerate(['Five-order design','20-order × 20-replication']):
        xs=[];ys=[];xl=[];xh=[]
        for idx,(w,ep) in enumerate(pairs):
            r=F[(F.world==w)&(F.endpoint==ep)&(F.design==design)].iloc[0]
            xs.append(r['mean']);ys.append(idx+(-0.12 if di==0 else 0.12));xl.append(r['mean']-r.lo);xh.append(r.hi-r['mean'])
        ax.errorbar(xs,ys,xerr=np.vstack([xl,xh]),fmt='o',capsize=3,label=design)
    ax.axvline(0,linewidth=1)
    ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis()
    ax.set_xlabel('Maximum response MI minus two-corner heuristic')
    ax.set_title('Leading-policy contrast under two curriculum-order designs')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT/'Fig5_order_replication_factorial.png',dpi=240,bbox_inches='tight')
    plt.close(fig)



def challenge_ranges():
    d=pd.read_csv(ROOT/'results'/'challenge_parameter_grid_summary.csv')
    policies=['interleaved_median_item','maximum_response_information_gain','two_corner_robust_response_information_gain']
    labels={'interleaved_median_item':'Interleaved curriculum','maximum_response_information_gain':'Maximum response MI','two_corner_robust_response_information_gain':'Two-corner response MI'}
    targets=sorted(d.target_success.unique())
    fig,ax=plt.subplots(figsize=(8.2,5.2))
    for pol in policies:
        means=[]; low=[]; high=[]
        for t in targets:
            x=d[(d.target_success==t)&(d.policy==pol)].expected_benefit_mean.to_numpy(float)
            means.append(float(x.mean())); low.append(float(x.min())); high.append(float(x.max()))
        means=np.array(means); low=np.array(low); high=np.array(high)
        ax.errorbar(targets,means,yerr=np.vstack([means-low,high-means]),marker='o',capsize=4,label=labels[pol])
    ax.axhline(0,linewidth=1)
    ax.set_xlabel('Challenge target success probability')
    ax.set_ylabel('Expected-response benefit vs blocked curriculum')
    ax.set_title('Challenge-dependent learning: range across width and floor settings')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT/'Fig6_challenge_expected_ranges.png',dpi=240,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':
    architecture(); response_diagnostics(); benefit_heatmap(); rank_trajectories(); order_factorial(); challenge_ranges()
