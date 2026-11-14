"""Synthetic private credit marks and a logistic regression that tries to beat them.

Everything here is made up on purpose. The point is to show the method, not to say
anything about a real lender. The real version will use loan-level marks from BDC
quarterly filings (10-Qs) on SEC EDGAR.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss

LENDERS = ["Lender A", "Lender B", "Lender C", "Lender D", "Lender E", "Lender F", "Lender G", "Lender H"]
# Hidden optimism in mark points. Positive means the lender marks too generously.
TRUE_BIAS = np.array([4.0, -2.0, 0.5, 6.0, -1.0, 2.0, -3.5, 1.0])
NOISE = np.array([2.5, 2.0, 3.0, 2.5, 1.5, 3.5, 2.0, 2.5])

FEATURES = ["mark", "chg", "disp", "pik", "sw21"]
FEATURE_NAMES = {
    "mark": "Lender's mark",
    "chg": "Change since last quarter",
    "disp": "Gap between lenders",
    "pik": "Paying interest with debt",
    "sw21": "Software loan from 2021",
}
TRAIN_QUARTERS = 8  # quarters 0-7 train, 8-11 test


def simulate(n_borrowers: int = 900, seed: int = 7) -> pd.DataFrame:
    """One row per lender per loan. Lenders mark on stale information on purpose."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_borrowers):
        q = int(rng.integers(0, 12))
        software = rng.random() < 0.35
        vint21 = rng.random() < 0.4
        sw21 = bool(software and vint21)
        pik = bool(rng.random() < (0.35 if software else 0.15))
        z1 = rng.normal()  # credit health lenders already see
        z2 = rng.normal()  # fresh bad news most marks have not caught up to
        p_true = 1 / (1 + np.exp(-(-3.2 + 1.0 * z1 + 0.9 * z2 + 0.8 * pik + 0.7 * sw21)))
        default = rng.random() < p_true
        # Lenders see z1 and only half-price the PIK and vintage risk.
        p_seen = 1 / (1 + np.exp(-(-3.2 + 1.0 * z1 + 0.4 * pik + 0.2 * sw21)))
        fair = 99 - 55 * p_seen
        drift = -2.5 * max(z2, 0) + rng.normal(0, 1.0)
        n = int(rng.integers(2, 6))
        holders = rng.choice(len(LENDERS), size=n, replace=False)
        reacts = rng.random(n) < 0.4  # some lenders react to the news, so marks spread out
        marks = np.clip(fair - 9 * max(z2, 0) * reacts + TRUE_BIAS[holders] + rng.normal(0, NOISE[holders]), 30, 101)
        disp = float(marks.max() - marks.min())
        for h, m in zip(holders, marks):
            rows.append(dict(borrower=b, quarter=q, lender=LENDERS[h], mark=float(m),
                             chg=float(drift + rng.normal(0, 1)), disp=disp, pik=int(pik),
                             sw21=int(sw21), stopped_paying=int(default)))
    return pd.DataFrame(rows)


@dataclass
class Results:
    data: pd.DataFrame
    mu: np.ndarray
    sd: np.ndarray
    model: LogisticRegression
    baseline: LogisticRegression
    logloss: dict
    calibration: pd.DataFrame
    lender_bias: pd.DataFrame
    test: pd.DataFrame

    def predict(self, mark, chg, disp, pik, sw21):
        x = (np.array([[mark, chg, disp, pik, sw21]]) - self.mu) / self.sd
        full = float(self.model.predict_proba(x)[0, 1])
        base = float(self.baseline.predict_proba(x[:, :1])[0, 1])
        return full, base

    @property
    def weights(self) -> pd.DataFrame:
        return pd.DataFrame({"input": [FEATURE_NAMES[f] for f in FEATURES], "weight": self.model.coef_[0]})


def fit(df: pd.DataFrame) -> Results:
    train = df[df.quarter < TRAIN_QUARTERS]
    test = df[df.quarter >= TRAIN_QUARTERS].copy()
    Xtr, ytr = train[FEATURES].to_numpy(float), train.stopped_paying.to_numpy()
    Xte, yte = test[FEATURES].to_numpy(float), test.stopped_paying.to_numpy()

    # Put every input on the same scale so the weights are comparable.
    mu, sd = Xtr.mean(0), Xtr.std(0)
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd

    model = LogisticRegression(C=10).fit(Ztr, ytr)
    baseline = LogisticRegression(C=10).fit(Ztr[:, :1], ytr)  # the lender's mark and nothing else

    test["p_model"] = model.predict_proba(Zte)[:, 1]
    test["p_mark"] = baseline.predict_proba(Zte[:, :1])[:, 1]
    ll = {
        "Base rate only": log_loss(yte, np.full(len(yte), ytr.mean())),
        "Lender's mark": log_loss(yte, test.p_mark),
        "Full model": log_loss(yte, test.p_model),
    }

    bins = pd.cut(test.p_model, np.linspace(0, 1, 11))
    cal = (test.groupby(bins, observed=True)
           .agg(predicted=("p_model", "mean"), actual=("stopped_paying", "mean"), loans=("p_model", "size"))
           .query("loans >= 15").reset_index(drop=True))

    # Lender bias: estimate each loan's risk from the market's average mark plus the warning signs,
    # turn that risk into the mark it "deserves", and average each lender's gap from it.
    d = df.copy()
    d["market_mark"] = d.groupby("borrower").mark.transform("mean")
    F = d[["market_mark", "chg", "disp", "pik", "sw21"]].to_numpy(float)
    risk = LogisticRegression(C=10).fit(F, d.stopped_paying).predict_proba(F)[:, 1].reshape(-1, 1)
    d["deserved_mark"] = LinearRegression().fit(risk, d.mark).predict(risk)
    d["gap"] = d.mark - d.deserved_mark
    bias = (d.groupby("lender").agg(estimated=("gap", "mean"), marks=("gap", "size")).reset_index())
    bias["actual"] = bias.lender.map(dict(zip(LENDERS, TRUE_BIAS)))

    return Results(df, mu, sd, model, baseline, ll, cal, bias.sort_values("estimated", ascending=False), test)
