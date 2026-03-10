import os
import pandas as pd
import numpy as np
import argparse
import json

# The EXACT 51 classes of RFMiD 2.0 in fixed column order.
# This order is the absolute contract between the data pipeline and the model head.
CLASS_NAMES = [
    'WNL', 'AH', 'AION', 'ARMD', 'BRVO', 'CB', 'CF', 'CL', 'CME', 'CNV', 'CRAO', 'CRS',
    'CRVO', 'CSR', 'CWS', 'CSC', 'DN', 'DR', 'EDN', 'ERM', 'GRT', 'HPED', 'HR', 'LS',
    'MCA', 'ME', 'MH', 'MHL', 'MS', 'MYA', 'ODC', 'ODE', 'ODP', 'ON', 'OPDM', 'PRH',
    'RD', 'RHL', 'RTR', 'RP', 'RPEC', 'RS', 'RT', 'SOFE', 'ST', 'TD', 'TSLN', 'TV', 
    'VS', 'HTN', 'IIH'
]

def analyze_rfmid_training_data(csv_path, output_dir):
    """
    Performs a complete label analysis on the RFMiD 2.0 training CSV.
    Computes N_pos, N_neg, positive rate, cardinality, Tiers, and writes the `pos_weight` matrix.
    """
    print(f"Loading training labels from: {csv_path}")
    df = pd.read_csv(csv_path, encoding="latin1")
    df.rename(columns=lambda x: str(x).strip(), inplace=True)

    # Ensure all 51 classes are actually present in the CSV
    missing_cols = [c for c in CLASS_NAMES if c not in df.columns]
    if missing_cols:
        print(f"CRITICAL ERROR: The following expected classes are missing from the CSV: {missing_cols}")
        return

    # Extract just the 51 label columns
    labels_df = df[CLASS_NAMES]
    
    # 1. Label Cardinality
    # The average number of diseases per image
    cardinality = labels_df.sum(axis=1).mean()
    print(f"\n--- LABEL CARDINALITY ---")
    print(f"Average labels per image: {cardinality:.2f}")

    # 2. Count Positive/Negative Samples per Class
    N_pos = labels_df.sum(axis=0)
    N_neg = len(df) - N_pos
    pos_rate = (N_pos / len(df)) * 100

    # 3. Categorize into Tiers
    tiers = {'Tier1_Frequent_over_30': [], 'Tier2_Moderate_10_to_30': [], 'Tier3_Rare_under_10': []}
    
    pos_weights = {}
    analysis_results = []

    for c in CLASS_NAMES:
        pos = N_pos[c]
        neg = N_neg[c]
        rate = pos_rate[c]
        
        # Compute pos_weight: w_c = min(N_neg / N_pos, 20)
        if pos == 0:
            weight = 20.0
        else:
            weight = min(neg / pos, 20.0)
            
        pos_weights[c] = weight
        
        # Categorize
        if pos >= 30:
            tiers['Tier1_Frequent_over_30'].append(c)
            tier_str = "Tier 1"
        elif 10 <= pos < 30:
            tiers['Tier2_Moderate_10_to_30'].append(c)
            tier_str = "Tier 2"
        else:
            tiers['Tier3_Rare_under_10'].append(c)
            tier_str = "Tier 3"

        analysis_results.append({
            'Class': c,
            'N_pos': pos,
            'N_neg': neg,
            'Pos_Rate_%': round(rate, 2),
            'Tier': tier_str,
            'pos_weight': round(weight, 4)
        })

    results_df = pd.DataFrame(analysis_results)
    
    print("\n--- CLASS BREADDOWN & TIERS ---")
    print(f"Tier 1 (Frequent >= 30 samples): {len(tiers['Tier1_Frequent_over_30'])} classes")
    print(f"Tier 2 (Moderate 10-29 samples): {len(tiers['Tier2_Moderate_10_to_30'])} classes")
    print(f"Tier 3 (Rare < 10 samples): {len(tiers['Tier3_Rare_under_10'])} classes")
    
    print("\nTop 10 Most Frequent Classes:")
    print(results_df.sort_values('N_pos', ascending=False).head(10).to_string(index=False))

    print("\nTier 3 Warning (These classes may not be learnable and require F1 documentation):")
    print(", ".join(tiers['Tier3_Rare_under_10']))

    # 4. Co-occurrence Matrix for top 10
    top_10_classes = results_df.sort_values('N_pos', ascending=False).head(10)['Class'].tolist()
    top_10_df = labels_df[top_10_classes]
    
    # Dot product of transpose computes co-occurrence
    cooc = top_10_df.T.dot(top_10_df)
    np.fill_diagonal(cooc.values, 0) # Remove self-co-occurrence
    
    print("\n--- CO-OCCURRENCE MATRIX (Top 10) ---")
    print(cooc)

    # 5. Save the pos_weight vector securely for training engines
    os.makedirs(output_dir, exist_ok=True)
    
    # We save an ordered list of weights that strictly maps to CLASS_NAMES
    weight_vector = [pos_weights[c] for c in CLASS_NAMES]
    
    weight_file = os.path.join(output_dir, 'rfmid_pos_weights.json')
    with open(weight_file, 'w') as f:
        json.dump({
            'CLASS_NAMES': CLASS_NAMES,
            'pos_weights_vector': weight_vector,
            'pos_weights_dict': pos_weights
        }, f, indent=4)
        
    print(f"\n[OK] Successfully saved pos_weight tensor geometry to {weight_file}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze RFMiD 2.0 Multilabel Dataset constraints.")
    parser.add_argument('--train-csv', required=True, help="Absolute path to RFMiD_2_Training_labels.CSV")
    parser.add_argument('--output-dir', default='./utils', help="Directory to save the pos_weight JSON map")
    
    args = parser.parse_args()
    analyze_rfmid_training_data(args.train_csv, args.output_dir)
