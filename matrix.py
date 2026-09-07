import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
import pandas as pd
from IPython.display import display

def plot_gen_imp(ax, scores, labels, title, xlabel, c_gen='#2196F3', c_imp='#F44336'):
    """
    Plot genuine/impostor histograms + KDE on a given Axes object.
    
    Parameters:
    -----------
    ax     : matplotlib.axes.Axes -> The subplot where the data will be drawn.
    scores : numpy.ndarray        -> 1D array of similarity/matching scores.
    labels : numpy.ndarray        -> 1D array of ground truth (1 for Genuine, 0 for Impostor).
    title  : str                  -> Title of the subplot.
    xlabel : str                  -> Label for the X-axis (e.g., 'Cosine Similarity').
    """
    
    gen = scores[labels == 1]
    imp = scores[labels == 0]
    min_val = np.min(scores)
    max_val = np.max(scores)
    bins = np.linspace(min_val, max_val, 55)
    ax.hist(imp, bins=bins, density=True, alpha=0.45, color=c_imp, label=f'Impostor (n={len(imp):,})')
    ax.hist(gen, bins=bins, density=True, alpha=0.45, color=c_gen, label=f'Genuine  (n={len(gen):,})')

    x = np.linspace(min_val, max_val, 300)
    
    try:
        ax.plot(x, stats.gaussian_kde(gen)(x), color=c_gen, lw=2.5)
        ax.plot(x, stats.gaussian_kde(imp)(x), color=c_imp, lw=2.5)
    except np.linalg.LinAlgError:
        pass 

    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel(xlabel) 
    ax.set_ylabel('Probability Density')
    ax.legend(framealpha=0.7)
    ax.grid(True, alpha=0.25)

def plot_roc_curve(sim_scores_A, dist_scores_B, labels, save_path='roc_curve.png'):
    """
    Computes and plots an ROC curve comparing a similarity-based method (A) 
    and a distance-based method (B), then saves the plot to disk.
    """
    
    fig, ax = plt.subplots(figsize=(7, 6))

    for scores, lbls, name, color in [
        (sim_scores_A, labels, 'Method A (Cosine)', '#1565C0'),
        # ------------------------------------------------------------------------------------------------------------------
        # similarity larger=better
        # GEN-> 0.95, IMP-> 0.12 similarity say 0.95 > 0.12). This perfectly matches the examiner's golden rule.

        # distance (small is better)
        # GEN 2 photos are identical then distance is 0.05
        # IMP 2 photos are different then distance is 4.80
        # (since we want large values for genuine, we use negative values)
        # so -0.05 is greater than -4.80 (match) because the 0.05 is closer to 0 and GEN
        # ------------------------------------------------------------------------------------------------------------------

        (-dist_scores_B, labels, 'Method B (Euclidean)', '#B71C1C'),
    ]:
        fpr, tpr, _ = roc_curve(lbls, scores, pos_label=1)
        
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f'{name}  (AUC = {roc_auc:.4f})')

    # 6. BASELINE: Draw a diagonal dashed black line ('k--') representing a random-guessing classifier (AUC = 0.5)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
    
    ax.set_xlabel('False Match Rate (FMR)')
    ax.set_ylabel('True Match Rate (TMR = 1 − FNMR)') 
    ax.set_title('ROC Curve — Method A vs. Method B', fontweight='bold')
    
    
    ax.legend(loc='lower right', framealpha=0.8)
    ax.grid(True, alpha=0.25)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f" Saved: {save_path}")
    
    return fig, ax

# D Prime
def compute_dprime(scores, labels):
    """
    Computes the d-prime (d') sensitivity index to measure the statistical 
    separation between genuine and impostor distribution curves.
    
    A higher d' value implies less overlap and better classification performance.
    """
    gen = scores[labels == 1]  
    imp = scores[labels == 0]  
    
    mu_g, mu_i = gen.mean(), imp.mean()
    
    # Calculate the average variance of both groups,
    pooled_std = np.sqrt(0.5 * (gen.var() + imp.var()))
    
    # to prevent division by zero.
    return (mu_g - mu_i) / pooled_std if pooled_std > 0 else np.inf

# ── EER ───────────────────────────────────────────────────────────────────────
def compute_eer(scores, labels):
    """Equal Error Rate: threshold where FMR ≈ FNMR.
    Returns (eer_percent, threshold).
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)# FMR , TMR, thresholds
    fnr = 1.0 - tpr                          # FNMR = 1 - TMR
    idx = np.argmin(np.abs(fpr - fnr))       # closest cross-over point 0 is perfect match
    eer = (fpr[idx] + fnr[idx]) / 2.0       # if 0 is not reachable take the average at crossing
    return eer * 100.0, float(thresholds[idx])


# ── TMR @ fixed FMR ───────────────────────────────────────────────────────────
def compute_tmr_at_fmr(scores, labels, target_fmr):
    """TMR (%) at the highest operating point where FMR ≤ target_fmr.
    Returns (tmr_percent, threshold).
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)# FMR , TMR, thresholds
    valid = np.where(fpr <= target_fmr)[0] 
    # valid allowed error rate EX: FMR  <= 0.01% not allowed for 0.01% impostor to enter the system
    if len(valid) == 0:# too save even gen blocked
        return 0.0, float(thresholds[0])
    idx = valid[-1] # last save point still within budget
    return float(tpr[idx]) * 100.0, float(thresholds[idx])


# ── Rank-1 Identification Rate ────────────────────────────────────────────────
def compute_rank1(score_matrix, probe_ids, gallery_ids): # 1-to-Many (1:N) identification mode
    """Rank-1 TPIR: fraction of probes correctly identified at Rank-1.
    For each probe, the gallery image with the HIGHEST similarity score
    is taken as the system's decision (similarity — higher = better).
    """
    """
    imagine gallery_ids as 2d array and rank-1 go through it compare with p_idx 
    to find best match score in diagonal  
    0 1 2 3 4 5
    1 1 0 0 0 0 
    2 0 1 0 0 0
    3 0 0 1 0 0
    4 0 0 0 1 0
    5 0 0 0 0 1

    if img 2 sreach until find best similarity best score 
    """
    correct = 0
    for p_idx, p_id in enumerate(probe_ids):
        best_g_idx = np.argmax(score_matrix[p_idx])
        # go through all images and find best match for each probe image
        if gallery_ids[best_g_idx] == p_id: # LAST CHACK: compares the id of the best match with the id of the probe image
            # if correct add 1 to correct counter
            correct += 1
    return correct / len(probe_ids) * 100.0




def calculate_and_print_metrics(sim_scores, dist_scores, labels, true_labels, pred_labels, pred_dist_labels, feature_type=""):
    print(f"\n---  Project Requirements Metrics ({feature_type.upper()}) ---")
    
    # --- Method A: Cosine Similarity ---
    print("\n[Method A: Cosine Similarity]")
    eer_A, _ = compute_eer(sim_scores, labels)
    dprime_A = compute_dprime(sim_scores, labels)
    tmr_1_A, _ = compute_tmr_at_fmr(sim_scores, labels, 0.01)
    tmr_01_A, _ = compute_tmr_at_fmr(sim_scores, labels, 0.0001)
    acc_A = accuracy_score(true_labels, pred_labels)
    
    print(f"Accuracy:        {acc_A * 100:.2f}%")
    print(f"EER:             {eer_A:.2f}%")
    print(f"D-prime:         {dprime_A:.4f}")
    print(f"TMR @ FMR=1%:    {tmr_1_A:.2f}%")
    print(f"TMR @ FMR=0.01%: {tmr_01_A:.2f}%")

    # --- Method B: Euclidean Distance ---
    print("\n[Method B: Euclidean Distance]")
    # Invert distances so higher score = better match for the ROC/EER functions
    inv_dist_scores = -np.array(dist_scores) 
    
    eer_B, _ = compute_eer(inv_dist_scores, labels)
    dprime_B = compute_dprime(inv_dist_scores, labels)
    tmr_1_B, _ = compute_tmr_at_fmr(inv_dist_scores, labels, 0.01)
    tmr_01_B, _ = compute_tmr_at_fmr(inv_dist_scores, labels, 0.0001)
    acc_B = accuracy_score(true_labels, pred_dist_labels)
    
    print(f"Accuracy:        {acc_B * 100:.2f}%")
    print(f"EER:             {eer_B:.2f}%")
    print(f"D-prime:         {dprime_B:.4f}")
    print(f"TMR @ FMR=1%:    {tmr_1_B:.2f}%")
    print(f"TMR @ FMR=0.01%: {tmr_01_B:.2f}%")




