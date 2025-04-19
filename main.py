import warnings
import pandas as pd # Import pandas for storing results
import numpy as np # Import numpy for isfinite check
import optuna # Optuna import 추가
from tqdm import tqdm # tqdm import 추가

# 불필요한 경고 무시
warnings.filterwarnings('ignore')

# from optimization import optimize_strategy
# from strategy import analyze_best_strategy
from ml_strategy import train_and_backtest_ml_strategy # Import ML strategy function

# --- Tqdm Callback for Optuna ---
class TqdmCallback:
    def __init__(self, total_trials):
        self.pbar = tqdm(total=total_trials, desc="Optimizing", unit="trial")
        self.completed_trials = 0

    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        # study.trials 데이터프레임 등을 사용하여 이미 완료된 trial 수를 계산할 수도 있음
        # 여기서는 간단하게 콜백이 호출될 때마다 1씩 증가
        self.completed_trials += 1
        self.pbar.update(1)
        # 진행률 표시줄에 최신 최고 점수 표시 (옵션)
        if study.best_trial:
             self.pbar.set_postfix({"best_value": f"{study.best_value:.2f}"})

    def close(self):
        self.pbar.close()
# ---------------------------------

# Optuna 목적 함수 정의
def objective(trial: optuna.Trial):
    # 최적화할 파라미터 제안
    sr_window = trial.suggest_int('sr_window', 5, 30, step=5) # 5, 10, ..., 30
    reg_window = trial.suggest_int('reg_window', 10, 40, step=5) # 10, 15, ..., 40
    future_days = trial.suggest_int('future_days', 3, 10) # 3 ~ 10
    stop_loss_pct = trial.suggest_float('stop_loss_pct', 0.01, 0.1, step=0.01) # 0.01 ~ 0.1 (1% ~ 10%)
    num_top_features = trial.suggest_int('num_top_features', 5, 20) # 5 ~ 20
    # 추가적으로 다른 파라미터 (예: threshold)도 여기서 제안 가능

    # 고정된 기본값들
    ticker = "SPY"
    start_date = "2019-01-01"
    end_date = "2025-04-17"
    model_type_to_run = 'RandomForest' # 현재는 RandomForest 고정

    # Trial 정보 출력 (간결하게)
    # print(f"\n--- Optuna Trial {trial.number} ---")
    # print(f"Parameters: sr={sr_window}, reg={reg_window}, fut={future_days}, stop={stop_loss_pct:.2f}, n_feat={num_top_features}")

    try:
        # 제안된 파라미터로 백테스트 실행
        ml_results = train_and_backtest_ml_strategy(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            sr_window=sr_window, # 제안된 값 사용
            reg_window=reg_window, # 제안된 값 사용
            future_days=future_days, # 제안된 값 사용
            model_type=model_type_to_run,
            tune_hyperparameters=True, # 하이퍼파라미터 튜닝은 계속 사용
            optimize_threshold=True, # 임계값 최적화도 계속 사용 (또는 이것도 Optuna로?)
            use_feature_selection=True,
            num_top_features=num_top_features, # 제안된 값 사용
            train_ratio=0.8,
            default_probability_threshold=0.55, # 기본값 또는 Optuna 제안값 사용 가능
            threshold_optimization_metric='final_return', # 최적화 기준 (final_return)
            stop_loss_pct=stop_loss_pct, # 제안된 값 사용
            commission_rate=0.0001 # 고정 수수료
        )

        if ml_results and 'final_return' in ml_results:
            final_return = ml_results['final_return']
            # Optuna는 NaN/inf 값을 처리하지 못하므로, 유효하지 않은 경우 낮은 값 반환
            if pd.isna(final_return) or not np.isfinite(final_return):
                 # print(f"Trial {trial.number} result invalid (NaN/inf), returning low value.")
                 # raise optuna.TrialPruned() # 또는 매우 낮은 값 반환
                 return -99999.0 # 혹은 다른 아주 작은 값
            # print(f"Trial {trial.number} result: Final Return = {final_return:.2f}%")
            return final_return
        else:
            # print(f"Trial {trial.number} failed or produced no results, returning low value.")
            # raise optuna.TrialPruned() # 백테스트 실패 시 시도 중단
            return -99999.0 # 혹은 다른 아주 작은 값

    except Exception as e:
        # print(f"Trial {trial.number} encountered an error: {e}, returning low value.")
        # 실패/오류 시 Optuna에 알림 (Pruning 대신 낮은 값 반환 유지)
        # trial.report(-99999.0, step=0) # 보고 후 Pruning 대신 낮은 값 반환
        return -99999.0 # 혹은 다른 아주 작은 값

if __name__ == "__main__":
    print("====== Optuna Parameter Optimization Start ======")

    # Optuna 스터디 생성 (SQLite DB 사용 및 기존 스터디 로드)
    study_name = 'ml_strategy_optimization_spy_rf' # 스터디 이름 지정 (DB 파일명과 연관)
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=storage_name, # SQLite DB 지정
        load_if_exists=True # DB 파일이 있으면 로드
    )

    # 최적화 실행 (n_trials 설정 및 tqdm 콜백 추가)
    n_trials = 100 # 예시: 100번 시도
    tqdm_callback = TqdmCallback(n_trials)

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[tqdm_callback], # tqdm 콜백 추가
            # n_jobs=1 # 필요 시 병렬 실행 (단, objective 함수가 thread-safe해야 함)
        )
    finally:
        tqdm_callback.close() # 완료 또는 중단 시 progress bar 닫기

    print("\n====== Optimization Finished ======")

    # 최적 결과 출력
    print(f"Number of finished trials (in DB): {len(study.trials)}")

    try:
        # DB에 저장된 모든 trial 중 가장 좋은 trial 찾기
        valid_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and t.value > -99999]
        if not valid_trials:
             raise ValueError("No successful trials were completed.")

        best_trial = max(valid_trials, key=lambda t: t.value)
        # best_trial = study.best_trial # 이 방식은 실패한 trial도 고려할 수 있음

        print(f"Best trial number: {best_trial.number}")
        print(f"Best trial final return: {best_trial.value:.2f}%")
        print("Best parameters found:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
    except ValueError as e:
        print(e)
        print("Could not find best parameters. Check the database or logs.")


    # 필요하다면 최적 파라미터로 최종 백테스트 다시 실행 가능
    # print("\nRunning final backtest with best parameters...")
    # try:
    #     best_params_dict = best_trial.params
    #     final_results = train_and_backtest_ml_strategy(
    #         ticker="SPY",
    #         start_date="2019-01-01",
    #         end_date="2025-04-17",
    #         model_type='RandomForest',
    #         tune_hyperparameters=True,
    #         optimize_threshold=True,
    #         use_feature_selection=True,
    #         train_ratio=0.8,
    #         default_probability_threshold=0.55,
    #         threshold_optimization_metric='final_return',
    #         commission_rate=0.0001,
    #         **best_params_dict # 최적 파라미터 적용
    #     )
    #     if final_results:
    #         print("\n--- Final Backtest Results with Best Params ---")
    #         for key, value in final_results.items():
    #             if key != 'portfolio_values' and key != 'trades':
    #                 print(f"{key}: {value}")
    #     else:
    #         print("Final backtest run failed.")
    # except ValueError:
    #     print("Could not run final backtest as no best parameters were found.")
    # except Exception as e:
    #      print(f"Error during final backtest run: {e}")


    # === Previous Optimization Strategy (Commented out) ===
    # print("====== 최적 전략 찾기 ======")
    # best_strategy = optimize_strategy(
    #     ticker=ticker,
    #     start_date=start_date, # Use original shorter start date? Or adjust?
    #     end_date=end_date,
    #     windows=[10, 20, 30],
    #     plot_results=True,
    #     stop_loss_pct=0.05
    # )
    # 
    # if best_strategy:
    #     print("\n====== 최적 전략 상세 분석 ======")
    #     window = best_strategy['window']
    #     weights = best_strategy['weights']
    #     
    #     analyze_best_strategy(
    #         ticker=ticker,
    #         start_date=start_date,
    #         end_date=end_date,
    #         window=window,
    #         weights=weights,
    #         stop_loss_pct=0.05
    #     ) 
    # else:
    #     print("최적화 과정에서 유효한 전략을 찾지 못했습니다.")
    # ==================================================== 