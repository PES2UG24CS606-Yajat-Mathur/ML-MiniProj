import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(file_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Loads credit card dataset, scales Time & Amount features,
    splits into train/test sets, and applies SMOTE to training data.
    """
    df = pd.read_csv(file_path)
    
    # RobustScaler handles outliers in Time and Amount effectively
    scaler = RobustScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    # Drop raw Time and Amount columns
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Stratified train-test split to preserve fraud ratio in test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Apply SMOTE only on training set to prevent data leakage
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    return X_train_resampled, X_test, y_train_resampled, y_test