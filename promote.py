import logging
 
from register import load_model, register_model, get_current_champion_model
from train import evaluate
 
logger = logging.getLogger(__name__)
 
MIN_RMSE_IMPROVEMENT_PCT = 0.01
 
def get_current_champion(mr, horizon_hours):
    return get_current_champion_model(mr, horizon_hours)
 
 
def promote_if_better(mr, horizon_hours, challenger_result, min_improvement_pct=MIN_RMSE_IMPROVEMENT_PCT):
    challenger_rmse = challenger_result["test_metrics"]["rmse"]
    champion_hw_model = get_current_champion(mr, horizon_hours)
 
    if champion_hw_model is None:
        logger.info(
            "Horizon %dh: no champion exists — registering challenger "
            "(%s, rmse=%.4f) unconditionally",
            horizon_hours, challenger_result["winner_name"], challenger_rmse,
        )
        return register_model(
            mr,
            horizon_hours=horizon_hours,
            model=challenger_result["winner_model"],
            test_metrics=challenger_result["test_metrics"],
            winner_name=challenger_result["winner_name"],
            n_train=challenger_result["n_train"],
            n_val=challenger_result["n_val"],
            n_test=challenger_result["n_test"],
        )
 
    champion_model = load_model(champion_hw_model)
    champion_metrics = evaluate(
        champion_model, challenger_result["X_test"], challenger_result["y_test"]
    )
    champion_rmse = champion_metrics["rmse"]
 
    improvement = champion_rmse - challenger_rmse
    threshold = champion_rmse * min_improvement_pct
    promote = improvement > threshold
 
    logger.info(
        "Horizon %dh: champion v%s rmse=%.4f (re-scored on today's test "
        "split) vs challenger (%s) rmse=%.4f -> improvement=%.4f, "
        "threshold=%.4f -> %s",
        horizon_hours, champion_hw_model.version, champion_rmse,
        challenger_result["winner_name"], challenger_rmse,
        improvement, min_improvement_pct,
        "PROMOTE" if promote else "KEEP CHAMPION",
    )
 
    if not promote:
        return None
 
    return register_model(
        mr,
        horizon_hours=horizon_hours,
        model=challenger_result["winner_model"],
        test_metrics=challenger_result["test_metrics"],
        winner_name=challenger_result["winner_name"],
        n_train=challenger_result["n_train"],
        n_val=challenger_result["n_val"],
        n_test=challenger_result["n_test"],
    )
 
 
def promote_all(mr, results, min_improvement_pct=MIN_RMSE_IMPROVEMENT_PCT):
    outcomes = {}
    for horizon_hours, challenger_result in results.items():
        outcomes[horizon_hours] = promote_if_better(
            mr, horizon_hours, challenger_result, min_improvement_pct
        )
 
    logger.info("=== Promotion summary ===")
    for h, outcome in outcomes.items():
        status = f"promoted to v{outcome.version}" if outcome else "champion retained"
        logger.info("  %2dh: %s", h, status)
 
    return outcomes
