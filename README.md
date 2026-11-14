# Can a Regression Beat the Lenders?

A small machine learning project on private credit marks. Private credit funds publish a price, or "mark," for every loan they hold. This project treats each mark as a prediction and asks whether a logistic regression can do better, and which lenders mark too generously.

It goes with my Substack post on Medallia and private credit valuations: [biancarubbico.substack.com](https://biancarubbico.substack.com)

**All data here is synthetic.** It shows the method. It says nothing about any real lender.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `model.py`: builds the synthetic loans and fits the models
- `app.py`: the Streamlit app, with four tabs: Try a loan, Lenders, Performance, How it works
- `HOW_IT_WORKS.md`: the full walkthrough of the regression, also shown in the app

## Next

Swap the synthetic data for loan-level marks from BDC 10-Q filings on SEC EDGAR.
