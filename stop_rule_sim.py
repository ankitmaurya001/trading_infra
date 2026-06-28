import random
import statistics as stats

# ------------------------------
# Config
# ------------------------------
N_SESSIONS = 100_000   # number of Monte Carlo runs
MAX_TRADES = 10       # max trades per session
RISK_PER_TRADE = 1.0   # loss size if trade loses

WIN_RATE_RANGE = (0.10, 0.30)  # uniform range for true win probability
RR_RANGE = (3.0, 6.0)          # uniform range for true risk-reward

random.seed(42)  # for reproducibility


# ------------------------------
# Strategy stopping rules
# ------------------------------
def strat_always_max(trade_results):
    """
    Always take all trades up to MAX_TRADES.
    trade_results: list of per-trade PnL (won't be fully used here)
    Returns index (exclusive) of how many trades to actually take.
    """
    return len(trade_results)


def strat_stop_after_first_win(trade_results):
    """
    Stop immediately after the first winning trade.
    """
    taken = 0
    for i, pnl in enumerate(trade_results):
        taken = i + 1
        if pnl > 0:
            break
    return taken


def strat_stop_after_k_consecutive_losses(trade_results, k=4):
    """
    Stop if you hit k consecutive losses, or reach MAX_TRADES.
    """
    taken = 0
    consec_losses = 0
    for i, pnl in enumerate(trade_results):
        taken = i + 1
        if pnl < 0:
            consec_losses += 1
        else:
            consec_losses = 0

        if consec_losses >= k:
            break

    return taken


def strat_stop_after_first_win_or_k_losses(trade_results, k=4):
    """
    Hybrid: stop after first win OR after k consecutive losses.
    """
    taken = 0
    consec_losses = 0
    for i, pnl in enumerate(trade_results):
        taken = i + 1

        if pnl > 0:
            break

        if pnl < 0:
            consec_losses += 1
            if consec_losses >= k:
                break

    return taken


# ------------------------------
# Simulation core
# ------------------------------
def simulate_one_session(win_prob, rr, stop_func):
    """
    Simulate one 10-trade session given true win_prob and rr.
    stop_func decides when to stop based on the PnL sequence.
    Returns total PnL of the session and number of trades taken.
    """
    # Pre-generate full sequence of potential trades
    trade_results = []
    for _ in range(MAX_TRADES):
        if random.random() < win_prob:
            trade_results.append(+rr * RISK_PER_TRADE)
        else:
            trade_results.append(-RISK_PER_TRADE)

    # Let strategy decide how many of those to actually take
    n_taken = stop_func(trade_results)
    n_taken = max(0, min(n_taken, MAX_TRADES))  # safety clip

    session_pnl = sum(trade_results[:n_taken])
    return session_pnl, n_taken


def run_experiment(n_sessions=N_SESSIONS):
    strategies = {
        "always_max": strat_always_max,
        "stop_after_first_win": strat_stop_after_first_win,
        "stop_after_2_losses": lambda tr: strat_stop_after_k_consecutive_losses(tr, k=2),
        "stop_after_3_losses": lambda tr: strat_stop_after_k_consecutive_losses(tr, k=3),
        "stop_after_4_losses": lambda tr: strat_stop_after_k_consecutive_losses(tr, k=4),
        "stop_after_5_losses": lambda tr: strat_stop_after_k_consecutive_losses(tr, k=5),
        "stop_after_win_or_2_losses": lambda tr: strat_stop_after_first_win_or_k_losses(tr, k=2),
        "stop_after_win_or_3_losses": lambda tr: strat_stop_after_first_win_or_k_losses(tr, k=3),
        "stop_after_win_or_4_losses": lambda tr: strat_stop_after_first_win_or_k_losses(tr, k=4),
        "stop_after_win_or_5_losses": lambda tr: strat_stop_after_first_win_or_k_losses(tr, k=5),
    }

    results = {name: {"pnls": [], "trades": []} for name in strategies}

    for _ in range(n_sessions):
        # Sample true (but unknown) win rate and RR for this run
        win_prob = random.uniform(*WIN_RATE_RANGE)
        rr = random.uniform(*RR_RANGE)

        for name, stop_func in strategies.items():
            pnl, n_trades = simulate_one_session(win_prob, rr, stop_func)
            results[name]["pnls"].append(pnl)
            results[name]["trades"].append(n_trades)

    # Summary stats
    summary = {}
    for name, data in results.items():
        pnls = data["pnls"]
        trades = data["trades"]
        avg_pnl = stats.mean(pnls)
        std_pnl = stats.pstdev(pnls)
        avg_trades = stats.mean(trades)
        prob_positive = sum(1 for x in pnls if x > 0) / len(pnls)

        summary[name] = {
            "avg_pnl_per_session": avg_pnl,
            "std_pnl_per_session": std_pnl,
            "avg_trades_per_session": avg_trades,
            "prob_session_profitable": prob_positive,
        }

    return summary


if __name__ == "__main__":
    summary = run_experiment()

    print(f"Simulated {N_SESSIONS} sessions.")
    print("Parameters:")
    print(f"  WIN_RATE_RANGE = {WIN_RATE_RANGE}")
    print(f"  RR_RANGE       = {RR_RANGE}")
    print(f"  MAX_TRADES     = {MAX_TRADES}")
    print()

    for name, stats_dict in summary.items():
        print(f"Strategy: {name}")
        print(f"  avg_pnl_per_session     = {stats_dict['avg_pnl_per_session']:.3f}")
        print(f"  std_pnl_per_session     = {stats_dict['std_pnl_per_session']:.3f}")
        print(f"  avg_trades_per_session  = {stats_dict['avg_trades_per_session']:.2f}")
        print(f"  prob_session_profitable = {stats_dict['prob_session_profitable']*100:.1f}%")
        print()