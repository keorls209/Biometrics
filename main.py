import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import numpy as np
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from matcher import (
    load_biometric_data, 
    match_single_face, 
    match_with_svm,
    match_with_optimized_fused_svm
)
from matrix import (
    calculate_and_print_metrics,
    compute_eer,
    compute_dprime,
    compute_tmr_at_fmr,
    plot_roc_curve,
    plot_gen_imp
)

def run_feature_fusion_experiment(data_path):
    print(f"\n{'='*60}")
    print(f" Strategy 1: Feature-Level Fusion (PCA + LBP)")
    print(f"{'='*60}")
    
    pca_train, pca_test, y_train, y_test = load_biometric_data(data_path, feature_type="pca")
    lbp_train, lbp_test, _, _ = load_biometric_data(data_path, feature_type="lbp")
    
    y_pred, model = match_with_optimized_fused_svm(pca_train, pca_test,
                                                   lbp_train, lbp_test,
                                                   y_train, y_test)

def process_all_matches(train_features, test_features, train_labels, test_labels):
    sim_scores_A, dist_scores_B, binary_labels = [], [], []
    true_labels, pred_A, pred_B = [], [], []
    
    for i in range(len(test_features)):
        res = match_single_face(test_features[i], test_labels[i], train_features, train_labels)
        
        sim_scores_A.extend(res['cosine']['genuine_scores'] + res['cosine']['impostor_scores'])
        dist_scores_B.extend(res['euclidean']['genuine_scores'] + res['euclidean']['impostor_scores'])
        binary_labels.extend([1]*len(res['cosine']['genuine_scores']) + [0]*len(res['cosine']['impostor_scores']))
        
        true_labels.append(test_labels[i])
        pred_A.append(res['cosine']['predicted_id'])
        pred_B.append(res['euclidean']['predicted_id']) 
        
    return (np.array(sim_scores_A), np.array(dist_scores_B), np.array(binary_labels), 
            np.array(true_labels), np.array(pred_A), np.array(pred_B))

def run_full_experiment(feature_type, data_path):
    print(f"\n{'='*60}\n Evaluation for: {feature_type.upper()} Features\n{'='*60}")
    
    try:
        X_train, X_test, y_train, y_test = load_biometric_data(data_path=data_path, feature_type=feature_type)
        (sA, dB, lbls, t_lbls, pA, pB) = process_all_matches(X_train, X_test, y_train, y_test)
        
       
        calculate_and_print_metrics(sA, dB, lbls, t_lbls, pA, pB, feature_type)
        
       
        y_pred_svm, _, svm_scores, svm_labels = match_with_svm(X_train, X_test, y_train, y_test)

        from sklearn.metrics import accuracy_score as _acc
        eer_svm,    _ = compute_eer(svm_scores, svm_labels)
        dprime_svm    = compute_dprime(svm_scores, svm_labels)
        tmr_1_svm,  _ = compute_tmr_at_fmr(svm_scores, svm_labels, 0.01)
        tmr_01_svm, _ = compute_tmr_at_fmr(svm_scores, svm_labels, 0.0001)
        acc_svm       = _acc(y_test, y_pred_svm) * 100

        print("\n[Method C: SVM]")
        print(f"Accuracy:        {acc_svm:.2f}%")
        print(f"EER:             {eer_svm:.2f}%")
        print(f"D-prime:         {dprime_svm:.4f}")
        print(f"TMR @ FMR=1%:    {tmr_1_svm:.2f}%")
        print(f"TMR @ FMR=0.01%: {tmr_01_svm:.2f}%")
        
       
        roc_path = os.path.join(BASE_DIR, f"roc_{feature_type}.png")
        plot_roc_curve(sA, dB, lbls, save_path=roc_path)
        print(f" ROC Curve saved to: {roc_path}")

      
        gen_imp_path = os.path.join(BASE_DIR, f"gen_imp_{feature_type}.png")
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_gen_imp(ax, sA, lbls, title=f"Gen-Imp Distribution ({feature_type.upper()})", xlabel="Cosine Similarity")
        plt.tight_layout()
        plt.savefig(gen_imp_path)
        plt.close()
        print(f" Gen-Imp Curve saved to: {gen_imp_path}")
        
    except Exception as e:
        print(f" Error in {feature_type}: {e}")


def main():
    PROJECT_PATH = os.path.join(BASE_DIR, "data")
    
    run_full_experiment("pca", PROJECT_PATH)
    run_full_experiment("lbp", PROJECT_PATH)
    run_feature_fusion_experiment(PROJECT_PATH)
    run_full_experiment("cnn", PROJECT_PATH)

if __name__ == "__main__":
    main()