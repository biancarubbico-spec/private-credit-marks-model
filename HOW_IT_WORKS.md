## The question

Private credit funds publish a price, or "mark," for every loan they hold, usually once a quarter. A mark is the lender's opinion of what the loan is worth, in cents on the dollar. That makes it a prediction: a mark of 98 says "this will get paid," a mark of 75 says "this is in trouble."

Predictions can be graded. This model asks two things:

1. Can a simple model predict which loans stop paying better than the lenders' own marks?
2. Which lenders mark consistently too high?

## Why logistic regression

The outcome is yes or no: did the loan stop paying interest within about 15 months? Logistic regression is built for exactly that. It takes a few inputs and returns a probability between 0 and 1.

It starts by adding up the inputs, each multiplied by a weight:

$$
z = b + w_1 x_1 + w_2 x_2 + \dots + w_5 x_5
$$

That sum, $z$, can be any number, so it gets squeezed into a probability with the sigmoid function:

$$
p = \frac{1}{1 + e^{-z}}
$$

A big negative $z$ gives a probability near 0. A big positive $z$ gives one near 1. Technically, $z$ is the log of the odds, so each weight says how much its input moves the odds of the loan going bad.

I picked it over fancier models for one reason: every weight can be read. In a piece about trusting other people's black boxes, I didn't want to build another one.

## The five inputs

| Input | What it captures |
| --- | --- |
| Lender's mark | The lender's own opinion, the starting point |
| Change since last quarter | Whether the mark is sliding. A falling mark can say more than a low one. |
| Gap between lenders | Highest mark minus lowest mark on the same loan. Disagreement often means someone knows something. |
| Paying interest with debt | Payment-in-kind (PIK): interest added to the loan instead of paid in cash |
| Software loan from 2021 | Loans written at the top of the software market |

Before fitting, every input is standardized: subtract its average and divide by its standard deviation.

$$
x_{\text{scaled}} = \frac{x - \mu}{\sigma}
$$

Without that step, a weight on the mark (measured in points) and a weight on a yes/no switch would be in different units and impossible to compare. After it, a bigger weight really does mean a bigger effect.

## How it learns the weights

The model tries weights and scores them with log loss, which measures how surprised the model was by what actually happened:

$$
\text{log loss} = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i \log p_i + (1 - y_i)\log(1 - p_i)\Big]
$$

Here $y_i$ is 1 if loan $i$ stopped paying and 0 if it didn't, and $p_i$ is the model's prediction. Saying 90% on a loan that ends up fine costs a lot more than saying 55%. That's the right standard for lenders too: a confident wrong mark should hurt.

Training finds the weights with the lowest log loss. I also add a small penalty for very large weights (L2 regularization, `C=10` in scikit-learn), which keeps the model from over-reacting to noise in a small dataset.

## Reading the results

- **Lender's mark** has a negative weight. Higher marks mean lower risk, as expected.
- **Gap between lenders** and **paying interest with debt** have positive weights. Both raise the risk even when the mark looks fine.
- **Change since last quarter** is negative: a mark that's rising means lower risk, and one that's falling means higher risk.

The table at the bottom of this tab turns each weight into an odds multiplier, $e^{w}$. A multiplier of 1.35 means one standard deviation more of that input makes the odds of trouble about 35% higher.

## How I tested it

I used the same rules as my golf Elo model:

1. **Move through time in order.** The model trains on the first 8 quarters and is tested on the last 4. A random split would let later quarters leak into earlier predictions, which is cheating.
2. **Compare against baselines.** The base rate (every loan gets the average default rate) and the lender's mark on its own. If the full model can't beat the mark, the extra inputs aren't adding anything.
3. **Check calibration.** Group loans by predicted probability and compare to what actually happened. If the model says 20%, about 20% should go bad.

## Finding generous lenders

This happens in two steps:

1. Estimate each loan's real risk using the average mark across all its lenders plus the warning signs. Using the average keeps any single lender from grading itself.
2. Turn that risk into the mark the loan "deserves," then take each lender's actual mark minus the deserved one. A lender whose gap is positive on average marks too generously.

Because the data is synthetic, I know each lender's real bias, so the Lenders tab shows the answer key next to the estimate. The method recovers the ranking and most of the size.

## Where Elo fits

I also tried an Elo-style leaderboard, where lenders gain rating when their mark was closer to what happened on a shared loan. On this data it was too noisy to separate the lenders. The regression found the bias much more clearly. I'll try Elo again on real data, with more quarters.

## Limits

- **The data is made up.** I built it so lenders mark on stale information, which is the effect I expect to find. The real test is real filings.
- **A mark isn't a clean probability.** It mixes the odds of a loan going bad with how much a lender expects to recover if it does.
- **Defaults are rare.** About 1 in 8 loans goes bad here, which leaves fewer examples to learn from. That's one reason to start with a simple model before trying gradient-boosted trees.
- **Rows aren't fully independent.** Several lenders hold the same loan, so their rows share one outcome.
- **Lenders may know more.** A lender can mark a loan down because of information that never shows up in a filing.

## The real version

BDCs list every loan they hold in their quarterly 10-Q filings on SEC EDGAR, with cost and fair value. The plan is to pull those schedules, match the same borrower across funds and label what happened next. After that, the same model gets fit to real marks.
