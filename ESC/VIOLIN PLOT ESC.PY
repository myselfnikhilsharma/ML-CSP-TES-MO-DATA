import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 22
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['mathtext.fontset'] = 'stix' 

# residuals_df = pd.read_excel('residuals for LCOE.xlsx')
residuals_df = pd.read_excel('residuals for ESC.xlsx')
residuals_melted = residuals_df.melt(var_name='Algorithms', value_name='Residual')

# Create figure
fig, ax = plt.subplots(figsize=(10, 10))

sns.violinplot(
    x='Residual',
    y='Algorithms',
    data=residuals_melted,
    palette="Set2",
    ax=ax
)

ax.set_xlabel('Residual', fontweight='bold')
ax.set_ylabel('', fontweight='bold')

# ✅ FORCE truly square plot
ax.set_box_aspect(1)

plt.tight_layout()
plt.show()

kurtosis_coeff = residuals_df.apply(lambda x: kurtosis(x, fisher=False, bias=False))
print("Kurtosis coefficient for each model:")
print(kurtosis_coeff)
