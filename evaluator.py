import numpy as np
import pandas as pd


class ClusterEvaluator:
    def calculate_forward_returns(self, data, labels, lookahead=5):
        n_samples = len(data)
        forward_returns = []
        valid_labels = []
        
        for i in range(n_samples):
            if i + lookahead < n_samples:
                ret = (data[i + lookahead] / data[i]) - 1
                forward_returns.append(ret)
                valid_labels.append(labels[i])
        
        forward_returns = np.array(forward_returns)
        valid_labels = np.array(valid_labels)
        
        results = []
        unique_clusters = np.unique(valid_labels)
        
        for cluster_id in unique_clusters:
            mask = valid_labels == cluster_id
            cluster_returns = forward_returns[mask]
            
            count = len(cluster_returns)
            avg_return = cluster_returns.mean()
            win_rate = (cluster_returns > 0).sum() / count
            
            results.append({
                'Cluster_ID': int(cluster_id),
                'Count': count,
                'Avg_Return': avg_return,
                'Win_Rate': win_rate
            })
        
        df = pd.DataFrame(results)
        return df
