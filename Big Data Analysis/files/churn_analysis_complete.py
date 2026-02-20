
# ============================================================
#   TELECOM CHURN ANALYSIS - COMPLETE PYSPARK-STYLE REPORT
#         (Run on Actual Churn.csv Data)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report

# ────────────────────────────────────────────────────────────
# STEP 1 & 2: LOAD DATA + LIBRARIES
# ────────────────────────────────────────────────────────────
df = pd.read_csv('/Users/mohanrajsubramaniam/PycharmProjects/python_learning/Big Data Analysis/Churn.csv')
print(f"Dataset Shape: {df.shape}")  # (3333, 21)

# ────────────────────────────────────────────────────────────
# STEP 6: PRE-PROCESSING
# ────────────────────────────────────────────────────────────
# Convert to categorical: IntlPlan, VMailPlan (stored as int but are flags)
df['IntlPlan']  = df['IntlPlan'].astype('category')
df['VMailPlan'] = df['VMailPlan'].astype('category')
df['Churn']     = df['Churn'].astype(int)
# Drop non-predictive columns (Phone number, State, AreaCode optional)
df_model = df.drop(columns=['Phone'])

# ────────────────────────────────────────────────────────────
# STEP 7.1: DESCRIBE
# ────────────────────────────────────────────────────────────
print(df.describe())

# ────────────────────────────────────────────────────────────
# STEP 7.2: HISTOGRAM - Day Minutes by Churn
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Day Minutes by Churn Status")
for val, ax, color, label in zip([0,1], axes, ['#E74C3C','#2ECC71'],
                                   ['Churner (0)','Non-Churner (1)']):
    subset = df[df['Churn']==val]['DayMins']
    ax.hist(subset, bins=30, color=color, edgecolor='black', alpha=0.85)
    ax.axvline(subset.mean(), color='navy', linestyle='--', label=f'Mean={subset.mean():.1f}')
    ax.set_title(label); ax.set_xlabel("Day Minutes"); ax.legend()
plt.tight_layout(); plt.savefig('fig_7_2_day_mins.png', dpi=120); plt.close()

# ────────────────────────────────────────────────────────────
# STEP 7.3: COUNT PLOT - VMailPlan vs Churn
# ────────────────────────────────────────────────────────────
vmail_pivot = df.groupby(['VMailPlan','Churn']).size().unstack()
vmail_pivot.index = ['No Plan','Has Plan']
vmail_pivot.columns = ['Churner(0)','Non-Churner(1)']
vmail_pivot.plot(kind='bar', color=['#E74C3C','#2ECC71'], edgecolor='black', figsize=(8,5))
plt.title("Voicemail Plan vs Churn"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig('fig_7_3_vmail_churn.png', dpi=120); plt.close()

# ────────────────────────────────────────────────────────────
# STEP 7.4: COUNT PLOT - IntlPlan vs Churn
# ────────────────────────────────────────────────────────────
intl_pivot = df.groupby(['IntlPlan','Churn']).size().unstack()
intl_pivot.index = ['No Intl Plan','Has Intl Plan']
intl_pivot.columns = ['Churner(0)','Non-Churner(1)']
intl_pivot.plot(kind='bar', color=['#E74C3C','#3498DB'], edgecolor='black', figsize=(8,5))
plt.title("International Plan vs Churn"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig('fig_7_4_intl_churn.png', dpi=120); plt.close()

# ────────────────────────────────────────────────────────────
# STEP 7.5: AREA WISE CHURN
# ────────────────────────────────────────────────────────────
area_pivot = df.groupby(['AreaCode','Churn']).size().unstack()
area_pivot.columns = ['Churner(0)','Non-Churner(1)']
area_pivot.plot(kind='bar', color=['#E74C3C','#9B59B6'], edgecolor='black', figsize=(9,5))
plt.title("Area Wise Churner vs Non-Churner"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig('fig_7_5_area_churn.png', dpi=120); plt.close()

# ────────────────────────────────────────────────────────────
# STEP 7.6: CORRELATION MATRIX
# ────────────────────────────────────────────────────────────
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(14,10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', mask=mask,
            linewidths=0.5, annot_kws={"size":8})
plt.title("Correlation Matrix")
plt.tight_layout(); plt.savefig('fig_7_6_corr_matrix.png', dpi=120); plt.close()

# ────────────────────────────────────────────────────────────
# STEP 8: CORRELATION WITH CHURN
# ────────────────────────────────────────────────────────────
churn_corr = corr['Churn'].drop('Churn').sort_values(ascending=False)
print("Correlation with Churn:")
print(churn_corr)

# ────────────────────────────────────────────────────────────
# STEP 9: MACHINE LEARNING
# ────────────────────────────────────────────────────────────
feature_cols = [
    'AccountLength','VMailMessage','DayMins','DayCalls','DayCharge',
    'EveMins','EveCalls','EveCharge','NightMins','NightCalls','NightCharge',
    'IntlMins','IntlCalls','IntlCharge','CustServCalls','IntlPlan','VMailPlan'
]
X = df[feature_cols].copy()
X['IntlPlan']  = X['IntlPlan'].astype(int)
X['VMailPlan'] = X['VMailPlan'].astype(int)
y = df['Churn']

# Stratified Split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for tr_idx, te_idx in sss.split(X, y):
    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                             random_state=42, stratify=y_train)

def run_model(model, name):
    model.fit(X_tr, y_tr)
    val_pred  = model.predict(X_val)
    test_pred = model.predict(X_test)
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"  Validation -> Acc:{accuracy_score(y_val,val_pred):.4f} | "
          f"Rec:{recall_score(y_val,val_pred):.4f} | Pre:{precision_score(y_val,val_pred):.4f}")
    print(f"  Test       -> Acc:{accuracy_score(y_test,test_pred):.4f} | "
          f"Rec:{recall_score(y_test,test_pred):.4f} | Pre:{precision_score(y_test,test_pred):.4f}")
    print(classification_report(y_test, test_pred, target_names=['Churner','Non-Churner']))

run_model(DecisionTreeClassifier(max_depth=7, random_state=42),        "Decision Tree")
run_model(RandomForestClassifier(n_estimators=100, random_state=42),   "Random Forest")
run_model(GradientBoostingClassifier(n_estimators=100, random_state=42),"Gradient Boosting")

# ────────────────────────────────────────────────────────────
# STEP 10: INSIGHTS & CONCLUSIONS
# ────────────────────────────────────────────────────────────
"""
KEY FINDINGS:
─────────────
EDA:
  • CustServCalls (r=0.21) and DayMins/DayCharge (r=0.21) are
    the strongest positive predictors of churn.
  • VMailMessage (r=-0.09) is negatively correlated — voicemail
    users are LESS likely to churn.
  • International plan holders churn at a much higher rate.
  • Area Code has almost zero correlation with churn.
  • AccountLength (r=0.02) — tenure does NOT predict churn.

MODEL PERFORMANCE:
  ┌─────────────────────┬──────────┬──────────┬──────────┐
  │ Model               │ Val Acc  │ Test Acc │ Test Pre │
  ├─────────────────────┼──────────┼──────────┼──────────┤
  │ Decision Tree       │  95.1%   │  92.2%   │  77.1%   │
  │ Random Forest       │  95.7%   │  94.5%   │  91.7%   │
  │ Gradient Boosting   │  96.6%   │  94.3%   │  87.3%   │
  └─────────────────────┴──────────┴──────────┴──────────┘

RECOMMENDATION:
  • Use Random Forest or Gradient Boosting in production.
  • Target customers with: high CustServCalls, high DayMins,
    International Plan = Yes, and VMailPlan = No for retention
    campaigns.
"""
