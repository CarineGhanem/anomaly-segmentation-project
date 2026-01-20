# Step 2: Test Different Temperatures (Super Fast!)
import os
import pickle
import numpy as np
import torch
from argparse import ArgumentParser
from sklearn.metrics import average_precision_score
from ood_metrics.metrics import fpr_at_95_tpr



def compute_anomaly_score(logits, temperature, method='msp'):
    """
    Compute anomaly score with temperature scaling
    
    Args:
        logits: Model output logits (numpy array)
        temperature: Temperature value for scaling
        method: 'msp', 'max_logit', or 'entropy'
    """
    # Convert to torch tensor
    logits_tensor = torch.from_numpy(logits)
    
    # Apply temperature scaling
    scaled_logits = logits_tensor / temperature
    
    if method == 'entropy':
        probabilities = torch.nn.functional.softmax(scaled_logits, dim=1)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-6), dim=1)
        anomaly_result = entropy.squeeze(0).numpy()
    
    elif method == 'msp':
        probabilities = torch.nn.functional.softmax(scaled_logits, dim=1)
        anomaly_result = 1.0 - np.max(probabilities.squeeze(0).numpy(), axis=0)
    
    else:  # max_logit
        max_logit = np.max(scaled_logits.numpy(), axis=1)
        anomaly_result = -max_logit.squeeze(0)
    
    return anomaly_result


def compute_metrics(anomaly_score_list, ood_gts_list):
    """
    Compute AUPRC and FPR@95TPR metrics
    """
    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)
    
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)
    
    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]
    
    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))
    
    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)
    
    return prc_auc, fpr


def main():
    parser = ArgumentParser()
    parser.add_argument('--logits_file', required=True, help='Path to saved logits pickle file')
    parser.add_argument('--method', default='msp', choices=['msp', 'max_logit', 'entropy'], 
                        help='Anomaly scoring method')
    parser.add_argument('--temperatures', nargs='+', type=float, 
                        default=[0.5, 0.75, 1.0, 1.1, 1.5, 2.0], 
                        help='List of temperatures to test')
    parser.add_argument('--output_file', default='temperature_results.txt', 
                        help='Output file for results')
    args = parser.parse_args()
    
    # Load saved logits
    print(f"Loading logits from {args.logits_file}...")
    with open(args.logits_file, 'rb') as f:
        data = pickle.load(f)
    
    logits_list = data['logits']
    ground_truths = data['ground_truths']
    
    print(f"Loaded {len(logits_list)} samples")
    
    # Extract dataset name from file
    dataset_name = os.path.basename(args.logits_file).replace('_logits.pkl', '')
    
    # Test different temperatures
    results = []
    best_auprc = 0
    best_temp = 1.0
    
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name} | Method: {args.method}")
    print(f"Testing {len(args.temperatures)} different temperatures...")
    print(f"{'='*80}")
    print(f"{'Temperature':<15} {'AUPRC (%)':<15} {'FPR@95TPR (%)':<15}")
    print(f"{'-'*80}")
    
    for temp in args.temperatures:
        # Compute anomaly scores for all samples with this temperature
        anomaly_score_list = []
        
        for logits in logits_list:
            anomaly_score = compute_anomaly_score(logits, temp, args.method)
            anomaly_score_list.append(anomaly_score)
        
        # Compute metrics
        prc_auc, fpr = compute_metrics(anomaly_score_list, ground_truths)
        
        results.append({
            'temperature': temp,
            'auprc': prc_auc * 100.0,
            'fpr95': fpr * 100.0
        })
        
        print(f"{temp:<15.2f} {prc_auc*100.0:<15.2f} {fpr*100.0:<15.2f}")
        
        if prc_auc > best_auprc:
            best_auprc = prc_auc
            best_temp = temp
    
    print(f"{'-'*80}")
    print(f"✓ Best Temperature: {best_temp:.2f} | Best AUPRC: {best_auprc*100.0:.2f}%")
    print(f"{'='*80}\n")
    
    # Save results to file
    with open(args.output_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Dataset: {dataset_name} | Method: {args.method}\n")
        f.write(f"{'='*80}\n")
        f.write(f"{'Temperature':<15} {'AUPRC (%)':<15} {'FPR@95TPR (%)':<15}\n")
        f.write(f"{'-'*80}\n")
        for res in results:
            f.write(f"{res['temperature']:<15.2f} {res['auprc']:<15.2f} {res['fpr95']:<15.2f}\n")
        f.write(f"{'-'*80}\n")
        f.write(f"Best Temperature: {best_temp:.2f} | Best AUPRC: {best_auprc*100.0:.2f}%\n")
        f.write(f"{'='*80}\n")
    
    print(f"✓ Results saved to {args.output_file}")


if __name__ == '__main__':
    main()