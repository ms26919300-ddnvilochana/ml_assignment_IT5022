
#### ------------  Algorithm Comparison Summary ------------  ####

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Random Forest results
rf_metrics = {
    'Accuracy' : 0.9047,   
    'Precision': 0.7066,  
    'Recall'   : 0.8343,   
    'F1 Score' : 0.7652,   
    'ROC-AUC'  : 0.9599    
}

# Logistic Regression results
lr_metrics = {
    'Accuracy' : 0.8761,
    'Precision': 0.6296,
    'Recall'   : 0.8124,
    'F1 Score' : 0.7094,
    'ROC-AUC'  : 0.9245
}

comparison_df = pd.DataFrame({
    'Logistic Regression': lr_metrics,
    'Random Forest'      : rf_metrics
}).round(4)

print("=" * 55)
print("ALGORITHM COMPARISON – TEST SET PERFORMANCE")
print("=" * 55)
print(comparison_df.to_markdown())

fig, ax = plt.subplots(figsize=(10, 5))
metrics_names = list(lr_metrics.keys())
x = np.arange(len(metrics_names))
width = 0.35

bars1 = ax.bar(x - width/2, list(lr_metrics.values()),  width,
               label='Logistic Regression', color='#1976D2', alpha=0.87, edgecolor='white')
bars2 = ax.bar(x + width/2, list(rf_metrics.values()),  width,
               label='Random Forest',       color='#D32F2F', alpha=0.87, edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Algorithm Comparison: Logistic Regression vs Random Forest\n(Same Dataset – Motor Vehicle Insurance)',
             fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.axhline(0.5, color='grey', linestyle=':', lw=1, alpha=0.6)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8.5, color='#1976D2', fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8.5, color='#D32F2F', fontweight='bold')

plt.tight_layout()
plt.savefig('algorithm_comparison.png', dpi=110, bbox_inches='tight')
plt.show()

