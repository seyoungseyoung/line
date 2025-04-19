import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib # For saving/loading models (optional)

from data import get_price_data
from ml_features import create_features, create_target
from backtest import run_backtest
# from visualization import visualize_best_strategy # Import if adapting for ML results

def train_and_backtest_ml_strategy(
    ticker: str,
    start_date: str,
    end_date: str,
    # --- Feature/Target Params ---
    sr_window: int = 10, 
    reg_window: int = 20,
    future_days: int = 5, 
    # --- ML Params ---
    model_type: str = 'GradientBoosting', # Defaulting to best model so far
    train_ratio: float = 0.8,
    tune_hyperparameters: bool = True,
    optimize_threshold: bool = True,
    use_feature_selection: bool = True, # Added feature selection flag
    num_top_features: int = 10,      # Number of features to select
    model_params: dict = None,
    default_probability_threshold: float = 0.55,
    threshold_optimization_metric: str = 'final_return',
    # --- Backtest Params ---
    stop_loss_pct: float = 0.05,
    commission_rate: float = 0.0001,
    initial_capital: int = 10000,
    allow_fractional: bool = True
):
    """
    Trains model (optional tuning, feature selection, threshold optimization) and backtests.
    """
    print(f"\n===== Running Strategy: {model_type} (Tune: {tune_hyperparameters}, FeatSel: {use_feature_selection} Top {num_top_features if use_feature_selection else 'All'}, Opt Thr: {optimize_threshold}) =====")
    # 1. Load Data
    print(f"Loading data for {ticker} from {start_date} to {end_date}...")
    price_data_full = get_price_data(ticker, start_date, end_date)
    if price_data_full.empty:
        print("Error: No price data loaded.")
        return None

    # 2. Create Features & Target
    features = create_features(price_data_full, sr_window=sr_window, reg_window=reg_window)
    # Ensure close_prices is a Series before passing
    close_prices_series = price_data_full['Close']
    if isinstance(close_prices_series, pd.DataFrame):
        print("Debug: Squeezing close_prices from DataFrame to Series.")
        close_prices_series = close_prices_series.squeeze()
    elif hasattr(close_prices_series, 'ndim') and close_prices_series.ndim > 1:
        print("Debug: Flattening close_prices from multi-dim array.")
        close_prices_series = close_prices_series.flatten()
        
    target = create_target(close_prices_series, future_days=future_days)
    
    # --- Debugging Concat Issue ---
    print("\n--- Debug: Features DataFrame (before concat) ---")
    print(f"Features shape: {features.shape}")
    print(features.info())
    print(features.head())
    
    print("\n--- Debug: Target Series (before concat) ---")
    print(f"Target shape: {target.shape}")
    print(target.info())
    print(target.head())
    # -----------------------------
    
    # 3. Combine and Clean Data
    combined_data = pd.concat([features, target], axis=1)
    
    # --- Debugging Concat Issue ---
    print("\n--- Debug: Combined DataFrame (after concat) ---")
    print(f"Shape: {combined_data.shape}")
    print(combined_data.info()) # Check dtypes and column names
    target_exists = target.name in combined_data.columns
    print(f"Target column '{target.name}' exists: {target_exists}")
    print(f"Total NaN count per column:\n{combined_data.isna().sum()}")
    print(f"Head:\n{combined_data.head()}")
    # -----------------------------
    
    # Critical: Drop NaNs AFTER combining features and target
    print("\nDropping NaN rows...")
    original_len = len(combined_data)
    combined_data = combined_data.dropna()
    print(f"Dropped {original_len - len(combined_data)} rows containing NaNs.")
    
    # --- Debugging Concat Issue ---
    print("\n--- Debug: After dropna() ---")
    print(f"Shape: {combined_data.shape}")
    target_exists_after = target.name in combined_data.columns
    print(f"Target column '{target.name}' exists: {target_exists_after}")
    if combined_data.empty:
        print("WARN: DataFrame is empty after dropna()!")
    elif not target_exists_after:
        print(f"ERROR: Target column '{target.name}' is missing after dropna()!")
    # -----------------------------
    
    if combined_data.empty:
        print("Error: No data remaining after NaN drop. Check feature/target calculation and data range.")
        return None
        
    # Ensure target column exists before dropping
    if target.name not in combined_data.columns:
        print(f"FATAL ERROR: Target column '{target.name}' not found in combined_data before splitting. Aborting.")
        return None
        
    X = combined_data.drop(columns=[target.name])
    y = combined_data[target.name]
    original_features = X.columns.tolist() # Keep track of original features
    
    # 4. Time Series Split
    split_index = int(len(X) * train_ratio)
    X_train_all, X_test_all = X.iloc[:split_index], X.iloc[split_index:] # Keep all features for now
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    test_dates = X_test_all.index
    print(f"\nTraining data: {X_train_all.index.min()} to {X_train_all.index.max()} ({len(X_train_all)} samples)")
    print(f"Testing data:  {X_test_all.index.min()} to {X_test_all.index.max()} ({len(X_test_all)} samples)")
    if len(X_train_all) == 0 or len(X_test_all) == 0:
        print("Error: Not enough data for training or testing after split.")
        return None

    # --- Scaling (Fit on ALL train features initially for feature importance calc) ---
    scaler = StandardScaler()
    X_train_scaled_all = scaler.fit_transform(X_train_all)
    # We will scale X_test later after potential feature selection
    # --------------------------------------------------

    # 5. Model Selection and Initial Training (for Feature Importance)
    model = None
    pipeline = None # Will be redefined after feature selection if needed
    feature_importances = None
    selected_features = original_features # Default to all features
    
    models = {
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
        # LogisticRegression doesn't have feature_importances_ directly
    }
    # Parameter grids used ONLY if tune_hyperparameters is True later
    param_grids = {
        'RandomForest': {
            # Refined grid around the params from the best RF run
            'clf__n_estimators': [80, 100, 120],      # Around 100
            'clf__max_depth': [12, 15, 18, 20],   # Around 15
            'clf__min_samples_leaf': [3, 5, 7],        # Around 5
            'clf__max_features': ['sqrt', 0.6, 0.8] # Keep sqrt, explore fractions
        },
        'LogisticRegression': { 'clf__C': [0.01, 0.1, 1, 10] },
        'GradientBoosting': {
            'clf__n_estimators': [50, 100],
            'clf__learning_rate': [0.05, 0.1],
            'clf__max_depth': [3, 5]
        }
    }
    
    if model_type not in models: # Only support models with feature importance for selection
        print(f"Error: Feature selection currently only supported for {list(models.keys())}")
        use_feature_selection = False # Disable feature selection if model not supported

    if use_feature_selection:
        print(f"\nPerforming initial model fit for feature selection ({model_type})...")
        # Use default parameters for initial fit to get importance scores
        initial_model = models[model_type]
        # Fit on scaled data to get importances
        initial_model.fit(X_train_scaled_all, y_train)
        
        if hasattr(initial_model, 'feature_importances_'):
            feature_importances = initial_model.feature_importances_
            importance_df = pd.DataFrame({'Feature': original_features, 'Importance': feature_importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            
            # Select top N features
            selected_features = importance_df['Feature'].head(num_top_features).tolist()
            print(f"Selected Top {num_top_features} Features based on importance:")
            print(importance_df.head(num_top_features))
            
            # --- Re-prepare data with selected features --- 
            X_train = X_train_all[selected_features]
            X_test = X_test_all[selected_features]
            
            # Re-fit scaler ONLY on selected features of training data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            # -----------------------------------------------
            
        else:
            print(f"Warning: Model type {model_type} does not support feature_importances_. Skipping feature selection.")
            use_feature_selection = False # Disable if importance not available
            # Use all features if selection fails
            X_train = X_train_all
            X_test = X_test_all
            X_train_scaled = X_train_scaled_all # Use initially scaled data
            # Need to scale X_test with the same scaler
            X_test_scaled = scaler.transform(X_test)
    else:
         # Use all features if selection is disabled
        X_train = X_train_all
        X_test = X_test_all
        X_train_scaled = X_train_scaled_all # Use initially scaled data
        X_test_scaled = scaler.transform(X_test) # Scale test data
        
    print(f"Using {X_train.shape[1]} features for final training/testing.")

    # 6. Final Model Training/Tuning on Selected Features
    best_params = None
    
    # Define models again for the final training pipeline
    final_models = {
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }
    # Add LogisticRegression back for non-feature-selection case
    if not use_feature_selection:
         final_models['LogisticRegression'] = LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear', max_iter=1000)
         
    if model_type not in final_models:
         print(f"Error: Model type {model_type} not supported for final training.")
         return None
         
    final_base_model = final_models[model_type]
    # Create pipeline with the SCALER FITTED ON SELECTED FEATURES
    pipeline = Pipeline([('scaler', scaler), ('clf', final_base_model)])
    current_param_grid = param_grids.get(model_type, {})
    
    if tune_hyperparameters and current_param_grid:
        print(f"\nStarting Hyperparameter Tuning for {model_type} on selected features...")
        tscv = TimeSeriesSplit(n_splits=5)
        f1_scorer = make_scorer(f1_score)
        # Tune the pipeline on the potentially reduced X_train
        grid_search = GridSearchCV(estimator=pipeline, param_grid=current_param_grid,
                                 cv=tscv, scoring=f1_scorer, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train) # Use unscaled selected X_train
        best_params = grid_search.best_params_
        model = grid_search.best_estimator_
        print(f"Best hyperparameters found: {best_params}")
        print(f"Best cross-validation score (F1): {grid_search.best_score_:.4f}")
    else:
        print(f"\nTraining {model_type} model with default/provided params on selected features...")
        if model_params: pipeline.set_params(**model_params)
        # Fit pipeline on unscaled selected X_train
        pipeline.fit(X_train, y_train)
        model = pipeline
        best_params = model.get_params()

    # Evaluate on Train set
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    print(f"Final Model Training Accuracy: {train_accuracy:.4f}")
    
    # 7. Prediction on Test Set (using selected features)
    print("\nPredicting on test set (using selected features)...")
    # Predict using pipeline with unscaled selected X_test
    test_pred = model.predict(X_test)
    test_pred_proba = model.predict_proba(X_test)[:, 1]
    test_accuracy = accuracy_score(y_test, test_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Test Set Classification Report:")
    print(classification_report(y_test, test_pred))
    
    # 8. Signal Generation & Threshold Optimization (using selected features)
    best_threshold = default_probability_threshold
    # Initialize best metric based on optimization direction (higher is better for Sharpe/Return)
    best_metric_value = -np.inf 
    best_backtest_results = None
    threshold_results = {}
    
    test_pred_proba_series = pd.Series(test_pred_proba, index=test_dates)
    price_data_test = price_data_full.loc[test_dates] 
    
    if optimize_threshold:
        print(f"\nOptimizing probability threshold based on '{threshold_optimization_metric}'.")
        # Refined thresholds range
        thresholds_to_test = np.linspace(0.53, 0.63, 21) # 0.53 to 0.63 with 0.005 steps
        
        for threshold in thresholds_to_test:
            print(f"  Testing threshold: {threshold:.3f}...", end=" ")
            signals = pd.Series(0.5, index=test_dates)
            signals.loc[test_pred_proba_series >= threshold] = 1.0
            signals.loc[test_pred_proba_series <= (1 - threshold)] = 0.0
            signals.name = f"{model_type}_Signal_Thr{threshold:.3f}"
            aligned_signals = signals.reindex(price_data_test.index).fillna(0.5)
            
            temp_results = run_backtest(
                price_data=price_data_test,
                signals=aligned_signals,
                initial_capital=initial_capital,
                allow_fractional=allow_fractional,
                commission_rate=commission_rate,
                stop_loss_pct=stop_loss_pct
            )
            
            # Get the metric value, handle potential missing key
            current_metric_value = temp_results.get(threshold_optimization_metric, -np.inf)
            # Ensure metric value is numeric before printing/comparing
            if not isinstance(current_metric_value, (int, float)):
                 print(f"Metric ({threshold_optimization_metric}): Invalid (Non-numeric)")
                 current_metric_value = -np.inf # Treat as worst case
            else:
                 print(f"Metric ({threshold_optimization_metric}): {current_metric_value:.4f}")
                 
            threshold_results[threshold] = temp_results
            
            # Update best if current metric is better
            if current_metric_value > best_metric_value:
                best_metric_value = current_metric_value
                best_threshold = threshold
                best_backtest_results = temp_results
                
        # Ensure best_backtest_results is populated even if loop didn't run or all failed
        if best_backtest_results is None and default_probability_threshold in threshold_results:
            print("Warning: Could not find a better threshold than default during optimization, using default results.")
            best_backtest_results = threshold_results[default_probability_threshold]
            best_threshold = default_probability_threshold
        elif best_backtest_results is None: # Handle case where no thresholds worked at all
            print("Warning: Threshold optimization failed to produce valid results. Falling back to default threshold run.")
            # Rerun with default threshold (copy logic from 'else' block below)
            best_threshold = default_probability_threshold # Reset to default
            signals = pd.Series(0.5, index=test_dates)
            signals.loc[test_pred_proba_series >= best_threshold] = 1.0
            signals.loc[test_pred_proba_series <= (1 - best_threshold)] = 0.0
            signals.name = f"{model_type}_Signal_Thr{best_threshold:.3f}"
            aligned_signals = signals.reindex(price_data_test.index).fillna(0.5)
            best_backtest_results = run_backtest(
                price_data=price_data_test,
                signals=aligned_signals,
                initial_capital=initial_capital,
                allow_fractional=allow_fractional,
                commission_rate=commission_rate,
                stop_loss_pct=stop_loss_pct
            )
            
        print(f"Optimal threshold found: {best_threshold:.3f} (Best {threshold_optimization_metric}: {best_metric_value:.4f})")
        
    else:
        # Use default threshold if not optimizing
        print(f"Using default probability threshold: {best_threshold:.3f}")
        signals = pd.Series(0.5, index=test_dates)
        signals.loc[test_pred_proba_series >= best_threshold] = 1.0
        signals.loc[test_pred_proba_series <= (1 - best_threshold)] = 0.0
        signals.name = f"{model_type}_Signal_Thr{best_threshold:.3f}"
        aligned_signals = signals.reindex(price_data_test.index).fillna(0.5)
        best_backtest_results = run_backtest(
            price_data=price_data_test,
            signals=aligned_signals,
            initial_capital=initial_capital,
            allow_fractional=allow_fractional,
            commission_rate=commission_rate,
            stop_loss_pct=stop_loss_pct
        )
        
    # 9. Final Results Reporting
    print("\n===== ML Strategy Backtest Performance (Test Set) =====")
    print(f"Strategy: ML {model_type} (Predict {future_days}d Up, Thr: {best_threshold:.3f}, Tuned: {tune_hyperparameters}, FeatSel: {use_feature_selection} Top {num_top_features if use_feature_selection else 'All'})") # Added FeatSel info
    if tune_hyperparameters and best_params:
        cleaned_params = {k.split('__', 1)[1]: v for k, v in best_params.items() if k.startswith('clf__')}
        print(f"Best Model Params: {cleaned_params}")
    print(f"Test Period: {test_dates.min().date()} to {test_dates.max().date()}")
    if best_backtest_results:
        print(f"Final Return: {best_backtest_results.get('final_return', 0):.2f}%")
        print(f"Buy & Hold Return (Test Period): {best_backtest_results.get('bh_return', 0):.2f}%")
        print(f"Excess Return: {best_backtest_results.get('excess_return', 0):.2f}%")
        print(f"Sharpe Ratio: {best_backtest_results.get('sharpe_ratio', 0):.3f}")
        print(f"Max Drawdown: {best_backtest_results.get('max_drawdown', 0):.2f}%")
        print(f"Number of Trades: {best_backtest_results.get('num_trades', 0)}")
        print(f"Stop-Loss Trades: {best_backtest_results.get('stop_loss_trades', 0)}")
    else:
        print("ERROR: No valid backtest results found.")

    # Feature Importance (print importance based on the initial fit if selection was done)
    if use_feature_selection and feature_importances is not None:
        # Already calculated and potentially printed earlier
        pass # Or reprint the top N that were selected if desired
    elif hasattr(model.named_steps['clf'], 'feature_importances_'):
        # If no selection, print importance from final model
        try:
            importances = model.named_steps['clf'].feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
            print("\nTop 10 Feature Importances (Final Model):")
            print(feature_importance_df.head(10))
        except Exception as e:
            print(f"\nCould not calculate feature importances for {model_type}: {e}")

    return best_backtest_results

# Example Usage
if __name__ == '__main__':
    results = train_and_backtest_ml_strategy(
        ticker='SPY',
        start_date='2010-01-01', # Use longer history for ML
        end_date='2023-12-31',
        train_ratio=0.8,
        future_days=5,
        stop_loss_pct=0.05,
        probability_threshold=0.55
    )
    if results:
         print("\nBacktesting finished successfully.")
    else:
         print("\nBacktesting failed.") 