from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

def get_models(random_state: int = 42):
    """
    Returns a dictionary of models analyzed in CS229 Report 32.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, 
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=random_state, 
            n_jobs=-1
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            random_state=random_state
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=6, 
            learning_rate=0.05, 
            eval_metric='logloss', 
            random_state=random_state
        )
    }
    return models