import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def apply_smote(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    # Convert to DataFrame if needed
    if not isinstance(X_res, pd.DataFrame):
        X_res = pd.DataFrame(X_res, columns=X.columns)
    if not isinstance(y_res, pd.DataFrame):
        y_res = pd.DataFrame(y_res, columns=['Class'])

    return pd.concat([X_res, y_res], axis=1)


df = pd.read_csv("creditcard.csv")

# ==============================
# Unscaled dataset
# ==============================
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pd.concat([X_test, y_test], axis=1).to_csv("creditcard_test_unscaled.csv", index=False)

resampled_unscaled = apply_smote(X_train, y_train)
resampled_unscaled.to_csv("creditcard_train_unscaled.csv", index=False)

# ==============================
# Scaled dataset
# ==============================
df_scaled = df.copy()
scaler = StandardScaler()
df_scaled[['Amount', 'Time']] = scaler.fit_transform(df_scaled[['Amount', 'Time']])

X_scaled = df_scaled.drop('Class', axis=1)
y_scaled = df_scaled['Class']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42, stratify=y_scaled
)

# Save test split
pd.concat([X_test_s, y_test_s], axis=1).to_csv("creditcard_test_scaled.csv", index=False)

# Apply SMOTE and save
resampled_scaled = apply_smote(X_train_s, y_train_s)
resampled_scaled.to_csv("creditcard_train_scaled.csv", index=False)
