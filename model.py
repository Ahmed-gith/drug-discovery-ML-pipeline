
import matplotlib
matplotlib.use('Agg') 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score

sns.set(style='white')

df = pd.read_csv('dataset_final.csv')

X = df.drop('pIC50', axis=1)
Y = df['pIC50']

valid_mask = ~np.isinf(Y) & ~np.isnan(Y)


X = X[valid_mask]
Y = Y[valid_mask]



selector = VarianceThreshold(threshold=(0.95 * (1 - 0.95)))

X_selected = selector.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(
    X_selected,    
    Y,             
    test_size=0.2, 
    random_state=42
)


model = RandomForestRegressor(
    n_estimators=500,    
    max_depth=None,      
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, Y_train)



Y_pred = model.predict(X_test)


r2 = r2_score(Y_test, Y_pred)

rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))




plt.figure(figsize=(6, 6))


ax = sns.regplot(
    x=Y_test,
    y=Y_pred,
    scatter_kws={'alpha': 0.4, 'color': 'steelblue'},
    line_kws={'color': 'red', 'linewidth': 2}
)


perfect_line = np.linspace(0, 12, 100)
plt.plot(perfect_line, perfect_line, 
         'k--', alpha=0.3, linewidth=1, label='Perfect prediction')

ax.set_xlabel('Experimental pIC50', fontsize=13, fontweight='bold')
ax.set_ylabel('Predicted pIC50', fontsize=13, fontweight='bold')
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.set_title(f'Random Forest: Experimental vs Predicted pIC50\nR² = {r2:.3f} | RMSE = {rmse:.3f}')
ax.figure.set_size_inches(6, 6)

plt.tight_layout()
plt.savefig('plot_experimental_vs_predicted.png', dpi=150)
plt.show()





importances = model.feature_importances_

selected_columns = X.columns[selector.get_support()]


feat_df = pd.DataFrame({
    'feature': selected_columns,
    'importance': importances
})

feat_df = feat_df.sort_values('importance', ascending=False)


plt.figure(figsize=(10, 6))
sns.barplot(
    x='importance',
    y='feature',
    data=feat_df.head(20),
    palette='viridis'

)
plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
plt.ylabel('Fingerprint Feature', fontsize=12, fontweight='bold')
plt.title('Top 20 Most Important Molecular Features\nfor Predicting Drug Potency (pIC50)')
plt.tight_layout()
plt.savefig('plot_feature_importance.png', dpi=150)
plt.show()



results_df = pd.DataFrame({
    'actual_pIC50': Y_test.values,
    'predicted_pIC50': Y_pred,
    'error': Y_test.values - Y_pred          

})

results_df.to_csv('model_predictions.csv', index=False)



import pickle
from sklearn.ensemble import HistGradientBoostingRegressor


hgb = HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
hgb.fit(X_train, Y_train)

hgb_r2 = r2_score(Y_test, hgb.predict(X_test))


with open('trained_model.pkl', 'wb') as f:
    pickle.dump(hgb, f)

with open('selector.pkl', 'wb') as f:
    pickle.dump(selector, f)

