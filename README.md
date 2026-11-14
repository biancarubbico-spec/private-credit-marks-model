# Can a Regression Beat the Lenders?

Private credit funds publish a price, or "mark," for every loan they hold, usually once a quarter. I started treating those marks as forecasts and asked two questions: can a simple model predict which loans stop paying better than the lenders' own marks, and can it tell which lenders mark too generously?

This repo is the public version of that project, with a Streamlit app on top. It runs the same pipeline on **synthetic data**, so every borrower and every lender here is made up. The write-up below is my full thought process, including the parts that didn't work.

It goes with my Substack post on Medallia and private credit valuations: [biancarubbico.substack.com](https://biancarubbico.substack.com)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Contents

1. [Where this started](#1-where-this-started)
2. [Framing the problem](#2-framing-the-problem)
3. [Why synthetic data, and how I built it](#3-why-synthetic-data-and-how-i-built-it)
4. [Features](#4-features)
5. [The model](#5-the-model)
6. [How I tested it](#6-how-i-tested-it)
7. [Results](#7-results)
8. [Finding generous lenders](#8-finding-generous-lenders)
9. [What didn't work: Elo](#9-what-didnt-work-elo)
10. [Limitations](#10-limitations)
11. [Moving to real filings](#11-moving-to-real-filings)
12. [What I'd try next](#12-what-id-try-next)
13. [Repo layout](#13-repo-layout)

---

## 1. Where this started

Medallia was a customer feedback software company that Thoma Bravo bought in 2021 for $6.4 billion, financed partly with a $1.8 billion private loan. In 2026, the lenders took the company over and about $5 billion of equity was wiped out.

What stuck with me was a smaller detail. At one point, three lenders holding the same Medallia loan valued it at 91, 82 and 77 cents on the dollar at the same time. Same loan, same company. One lender treated it as basically fine and another treated it as distressed.

In public markets that can't really happen, because a bond trades and gets one price. Private loans almost never trade, so each fund estimates a value with its own model and its own assumptions. That makes each mark an opinion, and opinions about the future can be scored once the future shows up.

## 2. Framing the problem

I set it up as a binary classification problem.

- **Unit of observation:** one lender's mark on one loan in one quarter. A loan held by four lenders produces four rows.
- **Target:** did the loan stop paying interest (go on nonaccrual) within roughly 15 months of the mark? `1` if yes, `0` if no.
- **Question 1:** does a model using the mark plus a few warning signs predict the target better than the mark alone?
- **Question 2:** after controlling for how risky a loan actually is, do some lenders mark consistently higher than others?

The 15-month window comes from an Octus study of business development company (BDC) portfolios, which found that about one in five loans marked below 90 went on nonaccrual within 15 months. That told me marks already carry signal, and that 15 months is a reasonable horizon to grade them on.

## 3. Why synthetic data, and how I built it

The version of this I run on real BDC filings isn't public. For this repo I wanted something anyone could run and inspect, with a known answer key so you can check whether the method actually recovers what's there. Synthetic data gives you that: I decide how biased each lender is, so I can see whether the model finds it.

The generator is in `simulate()` in `model.py`. Here is the logic, step by step.

**Borrowers.** 900 made-up borrowers, each observed in one of 12 quarters. Each gets:

- a software flag (35% chance) and a 2021 vintage flag (40%). `sw21` is on when both are true.
- a payment-in-kind (PIK) flag, meaning the borrower pays interest with more debt. It's more likely for software borrowers (35%) than others (15%), which matches what I saw in the Medallia-era deals.

**Two kinds of credit risk.** This is the core design choice. Each borrower has two hidden risk factors:

- `z1`: credit health that lenders can already see
- `z2`: fresh bad news that most marks haven't caught up to yet

The true probability of the loan going bad uses both:

```
logit(p_true) = -3.2 + 1.0*z1 + 0.9*z2 + 0.8*pik + 0.7*sw21
```

**Lenders mark on stale information.** A lender's "fair" mark only uses `z1`, and it only half-prices the PIK and vintage risk:

```
logit(p_seen) = -3.2 + 1.0*z1 + 0.4*pik + 0.2*sw21
fair_mark    = 99 - 55 * p_seen
```

I built it this way on purpose. It's my hypothesis about why marks lag: lenders anchor on what they already know, and they underweight structural risks like PIK and peak-market vintages. If the hypothesis is right, a model that adds those signals back in should beat the marks. If the marks already captured everything, the extra inputs would add nothing. You can test that by changing the coefficients in `simulate()`.

**Some lenders react, most don't.** Each lender on a loan has a 40% chance of seeing the bad news (`z2`) and marking down for it. That creates disagreement between lenders exactly when news is bad, which is what I think happened with Medallia.

**Lender bias.** Eight lenders, each with a hidden optimism bias in mark points, plus their own noise:

| Lender | True bias (points) | Noise (sd) |
| --- | ---: | ---: |
| Lender A | +4.0 | 2.5 |
| Lender B | -2.0 | 2.0 |
| Lender C | +0.5 | 3.0 |
| Lender D | +6.0 | 2.5 |
| Lender E | -1.0 | 1.5 |
| Lender F | +2.0 | 3.5 |
| Lender G | -3.5 | 2.0 |
| Lender H | +1.0 | 2.5 |

Each loan is held by 2 to 5 random lenders, about 3.5 on average. The final mark is the fair mark, minus a news hit if that lender reacted, plus that lender's bias and noise, clipped to between 30 and 101.

The result is 3,144 lender-loan rows, with about 12% of loans going bad. That's higher than real private credit default rates. I kept it elevated so the test set has enough positive cases to measure calibration.

## 4. Features

Five inputs, all things you could pull from a BDC's quarterly filing:

| Feature | Column | Why it's there |
| --- | --- | --- |
| Lender's mark | `mark` | The lender's own forecast, and the baseline everything has to beat |
| Change since last quarter | `chg` | A mark that's falling fast often says more than a low one |
| Gap between lenders | `disp` | Highest minus lowest mark on the same loan. Disagreement usually means someone knows something. |
| Paying interest with debt | `pik` | PIK lets a company defer the bill, which hides stress |
| Software loan from 2021 | `sw21` | Loans written at the top of the software market, at peak multiples |

`disp` is the feature I was most curious about. The Medallia spread (91 vs. 77) was the whole reason I started, so I wanted to know whether disagreement predicts trouble on its own, separate from the level of the mark.

One subtle point: `disp` is the same for every lender on a loan, so it carries information from the other lenders into each row. That's intentional. The question is whether the crowd's disagreement adds anything beyond one lender's own number.

## 5. The model

Logistic regression, from scikit-learn.

```
z = b + w1*mark + w2*chg + w3*disp + w4*pik + w5*sw21
p = 1 / (1 + exp(-z))
```

`z` is the log-odds of the loan going bad, and the sigmoid turns it into a probability between 0 and 1.

**Why logistic regression and not something bigger.** Three reasons:

1. Every weight is readable. In a project about not trusting other people's black boxes, I didn't want to build another one.
2. Defaults are rare. With a few hundred positive cases, a high-variance model like a gradient-boosted tree can memorize noise. A linear model is the honest starting point.
3. It gives a clean baseline. If logistic regression can't beat the marks, a fancier model probably shouldn't be trusted when it does.

**Standardization.** Each feature is scaled using the training set's mean and standard deviation:

```
x_scaled = (x - mean_train) / sd_train
```

Without it, the weight on `mark` (measured in points) and the weight on `pik` (0 or 1) would be in different units and impossible to compare. I use the training statistics on the test set too, so no test information leaks into the scaling.

**Regularization.** `LogisticRegression(C=10)` uses scikit-learn's default L2 penalty. `C` is the inverse of the penalty strength, so `C=10` is a light penalty. It keeps the weights from blowing up on a smallish dataset without forcing them much toward zero. I didn't tune `C`. On a dataset this size, tuning it would mostly be fitting noise.

**Baseline model.** A second logistic regression that uses only `mark`. This matters: the honest comparison is "mark plus warning signs" against "the mark, calibrated." The raw mark isn't a probability, so I let a one-feature model learn the best mapping from mark to risk first.

## 6. How I tested it

I used the same rules as my golf Elo project:

**Time-ordered split.** Quarters 0 to 7 train the model (2,051 rows), and quarters 8 to 11 test it (1,093 rows). A random split would let later quarters inform predictions about earlier ones, which is leakage. In real use, you only ever predict forward.

**Log loss as the main metric.**

```
log loss = -(1/N) * sum[ y*log(p) + (1-y)*log(1-p) ]
```

It's a proper scoring rule, so the best possible score comes from reporting your true belief. It also punishes confident mistakes much harder than timid ones. A lender at 91 on a loan that stops paying should get hit hard, and log loss does that.

I didn't use accuracy. With about 12% of loans going bad, a model that predicts "fine" every time is 88% accurate and completely useless.

**Three comparisons:**

- Base rate: every loan gets the training-set default rate
- Mark only: the calibrated one-feature model
- Full model: all five features

**Calibration.** I bucket test predictions into 10% bins and compare the average prediction with the actual default rate in each bucket. Bins with fewer than 15 loans are dropped because they're too noisy to read.

## 7. Results

**Log loss on the test quarters** (lower is better):

| Approach | Log loss |
| --- | ---: |
| Base rate only | 0.3845 |
| Lender's mark | 0.3693 |
| Full model | 0.3597 |

The marks beat the base rate, so they do carry real information. The full model beats the marks by about 2.6%. That's a modest gain, but it holds on quarters the model never saw, which is what I care about.

**Weights** (on standardized inputs), with odds multipliers:

| Input | Weight | Odds multiplier |
| --- | ---: | ---: |
| Lender's mark | -0.292 | 0.75 |
| Change since last quarter | -0.107 | 0.90 |
| Gap between lenders | +0.204 | 1.23 |
| Paying interest with debt | +0.302 | 1.35 |
| Software loan from 2021 | +0.264 | 1.30 |

How to read the multiplier: one standard deviation more of that input multiplies the odds of the loan going bad by that amount. So one standard deviation more disagreement between lenders makes the odds of trouble about 23% higher, even with the mark held fixed.

What I take from this:

- The mark is still the single biggest input, which makes sense. Lenders aren't clueless.
- PIK and the 2021 software vintage carry real weight on their own. That's the structural risk the lenders half-priced in the generator, and the model picked it up.
- Disagreement between lenders matters, which was the Medallia hypothesis.
- The quarter-over-quarter change is the weakest input. I expected more from it. In this data, the reacting lenders' markdowns show up in `disp` more strongly than in `chg`, so the two signals overlap.

**Calibration on the test set:**

| Predicted | Actual | Loans |
| ---: | ---: | ---: |
| 7.2% | 7.4% | 584 |
| 13.8% | 15.8% | 387 |
| 23.8% | 24.1% | 79 |
| 34.4% | 35.0% | 20 |
| 45.3% | 62.5% | 16 |

The model is well calibrated where most of the loans are. In the top bucket it underpredicts, but that bucket only has 16 loans, so I wouldn't read much into it. It does hint that the riskiest loans are riskier than a linear model can express, which is one reason to try nonlinear models later.

## 8. Finding generous lenders

This is the part I care about most, because it turns "I think some funds mark late" into something you can measure.

**The obvious approach doesn't work.** My first idea was to add a dummy variable for each lender to the outcome model. The problem: on a shared loan, every lender gets the same outcome, so lender dummies can't explain much about defaults. The bias lives in the marks, not in the outcomes.

**What I did instead,** in two steps:

1. **Estimate each loan's real risk without letting any lender grade itself.** I fit a logistic regression on all rows, using the *average* mark across all lenders on that loan (the "market mark") plus the four warning signs. Using the average keeps any one lender's optimism from defining the loan's risk.
2. **Turn risk into a "deserved" mark and measure each lender's gap.** I regress actual marks on that predicted risk with a simple linear regression, which gives the mark a loan "deserves" at its level of risk. Each row's residual is `actual mark - deserved mark`. Averaging residuals by lender gives that lender's estimated bias.

**Results against the answer key:**

| Lender | Estimated bias | True bias | Marks |
| --- | ---: | ---: | ---: |
| Lender D | +3.50 | +6.0 | 387 |
| Lender A | +2.48 | +4.0 | 383 |
| Lender F | +1.20 | +2.0 | 395 |
| Lender H | +0.61 | +1.0 | 393 |
| Lender C | -0.10 | +0.5 | 391 |
| Lender E | -1.34 | -1.0 | 404 |
| Lender B | -2.75 | -2.0 | 404 |
| Lender G | -3.43 | -3.5 | 387 |

The method gets the ranking exactly right: all eight lenders land in the correct order. The sizes are compressed for the most optimistic lenders. Lender D's true bias is +6, and the estimate is +3.5.

I think I know why. Step 1 uses the average mark across lenders, and that average includes the optimistic lender's own inflated mark. So part of Lender D's optimism gets absorbed into the "market" view of the loan, which makes the loan look safer and its deserved mark higher. That shrinks D's measured gap. A fix would be a leave-one-out market mark, computing the average without the lender being scored. That's the next thing I'd change.

## 9. What didn't work: Elo

My first plan for scoring lenders was Elo, the rating system I used for golf. Every time a shared loan resolves, compare each pair of lenders on it. Whoever's mark was closer to what actually happened "wins," and ratings update.

**Version 1 rewarded the worst lenders.** I scored each pairwise matchup as a win or loss based on whose mark was closer to the outcome. The two most optimistic lenders came out on top. It took me a minute to see why: about 88% of loans don't default, so on most matchups the lender closer to "fine" wins. Win/loss scoring is a hit rate, and a hit rate rewards always betting on the common outcome. It isn't a proper scoring rule.

**Version 2 fixed the incentive but lost the signal.** I switched to a margin-based result. I converted each mark to an implied probability of trouble, scored it with the Brier score (a proper rule), and set each matchup's result from the difference in scores rather than who won. That stopped rewarding optimism, but the ratings barely moved. Every lender ended between about 1,494 and 1,508, which is noise. Each lender only shows up on a few hundred loans, and each loan resolution moves ratings by a tiny amount.

**What I learned:** Elo is built for head-to-head games with lots of matches and clear winners. Lender marks are closer to a forecasting contest with rare events, and for that, a regression that pools all the evidence at once is much more efficient. Elo is out of this version. It might come back on real data with many more quarters, as a leaderboard on top of the regression rather than the main method.

## 10. Limitations

- **The data is made up.** I built the generator around my own hypothesis that lenders mark on stale information. The model finding that effect shows the method works, not that the effect exists in real markets.
- **A mark isn't a clean probability.** It mixes two things: the chance a loan goes bad, and how much the lender expects to recover if it does. A loan marked at 80 could be a 40% chance of a 50% loss, or something else. This model treats the mark as one risk signal and doesn't separate the two.
- **Rows aren't independent.** Several lenders hold each loan, so their rows share one outcome. That makes the effective sample size smaller than 3,144, and naive standard errors would be too optimistic. I don't report p-values for that reason.
- **Rare events.** About 110 borrowers go bad in this data. Real private credit default rates are lower, which makes it even harder.
- **Lenders may know more.** A lender can mark a loan down because of information that never shows up in a filing. When that happens, the "bias" is actually skill, and this method can't tell the difference.
- **Survivorship.** Loans that get refinanced or sold drop out of filings before an outcome is recorded. On real data, that can make marks look better than they were.

## 11. Moving to real filings

BDCs are the one corner of private credit that publishes holdings. Each quarterly 10-Q on SEC EDGAR includes a schedule of investments with every loan's borrower, cost, fair value and usually whether it's on nonaccrual or paying PIK. Getting from those filings to a modeling dataset takes a few steps:

1. **Pull the schedules.** They're HTML tables that vary a lot by fund. Some funds also tag them in XBRL, which is easier to parse.
2. **Entity matching.** The same borrower can show up as "Medallia, Inc.," "Medallia Inc" or under the name of the holding company that actually took the loan, depending on the fund. Name normalization plus manual review for the big names is unavoidable.
3. **Tranche matching.** One borrower can have a first-lien term loan, a revolver and a delayed-draw tranche, and they're marked differently. You have to match tranche to tranche, not just borrower to borrower.
4. **Label outcomes.** Nonaccrual flags in later filings are the cleanest label. Restructurings and exits need their own rules.
5. **Mark as a percent of cost or par**, not dollars, so loans of different sizes are comparable.

## 12. What I'd try next

- **Leave-one-out market mark** for the lender bias step, to fix the compression in section 8.
- **Mixed-effects model** with a random effect per lender and per borrower. That's a cleaner way to estimate lender bias while accounting for shared loans.
- **Separate default and recovery.** Model the chance of nonaccrual and the recovery given nonaccrual separately, so a mark can be compared with expected value and not only default risk.
- **Gradient-boosted trees**, but only after the linear model is solid, and compared with it on the same time-ordered split.
- **Time-varying lender bias.** Does a lender get more generous when its fund is facing redemptions? That's the incentive problem from the Substack post, and I think it's the most interesting question here.

## 13. Repo layout

| File | What it does |
| --- | --- |
| `model.py` | Builds the synthetic data (`simulate`) and fits everything (`fit`) |
| `app.py` | The Streamlit app, with four tabs: Try a loan, Lenders, Performance, How it works |
| `HOW_IT_WORKS.md` | A shorter, plain-language explainer, shown in the app's last tab |
| `requirements.txt` | Pinned versions the app was tested with |
| `.streamlit/config.toml` | The black and white theme |

Everything is deterministic (`seed=7`), so you should get the same numbers as the tables above.

---

All data in this repo is synthetic. Lender names are placeholders, and nothing here describes a real lender or loan.

By [Bianca Rubbico](https://biancarubbico.substack.com)
