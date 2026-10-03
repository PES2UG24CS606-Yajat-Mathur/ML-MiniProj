import pandas as pd
from src.data_processing import load_and_preprocess_data
from src.models import get_models
from src.evaluation import evaluate_model

DATA_PATH = "data/creditcard.csv"

def main():
    print("Loading data and applying SMOTE preprocessing...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)
    
    models = get_models()
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        
    summary_df = pd.DataFrame(results)
    print("\n================ Comparative Model Performance ================")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()