import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import model

st.set_page_config(page_title="Can a Regression Beat the Lenders?", layout="centered")

st.markdown(
    """
    <style>
    .block-container {padding-top: 3rem; max-width: 880px;}
    h1 {font-weight: 600; letter-spacing: -0.02em; text-align: center;}
    .sub {text-align: center; color: #6e6e73; font-size: 1.15rem; margin-top: -0.6rem;}
    .eyebrow {text-align: center; color: #6e6e73; font-size: 0.85rem;}
    .big {font-size: 5rem; font-weight: 600; letter-spacing: -0.03em; line-height: 1; text-align: center;}
    .cap {color: #6e6e73; text-align: center; font-size: 0.95rem;}
    .verdict {text-align: center; font-weight: 500; font-size: 1.05rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

INK, SOFT = "#1d1d1f", "#c7c7cc"


@st.cache_resource
def load():
    return model.fit(model.simulate())


r = load()

st.markdown('<p class="eyebrow">Synthetic data</p>', unsafe_allow_html=True)
st.title("Can a regression beat the lenders?")
st.markdown('<p class="sub">Price a private loan. See what the model thinks.</p>', unsafe_allow_html=True)
st.write("")

tab_try, tab_lenders, tab_perf, tab_how = st.tabs(["Try a loan", "Lenders", "Performance", "How it works"])

PRESETS = {
    "Medallia-style": (82.0, -5.0, 14.0, True, True),
    "Healthy": (98.0, 0.0, 2.0, False, False),
    "Lenders split": (93.0, -1.0, 18.0, True, False),
}

with tab_try:
    preset = st.segmented_control("Example", list(PRESETS), default="Medallia-style", label_visibility="collapsed")
    m0, c0, d0, p0, s0 = PRESETS[preset or "Medallia-style"]
    left, right = st.columns(2, gap="large")
    with left:
        mark = st.slider("Lender's mark", 50.0, 100.0, m0, 0.5, key=f"m{preset}")
        chg = st.slider("Change since last quarter", -15.0, 2.0, c0, 0.5, key=f"c{preset}")
        disp = st.slider("Gap between lenders", 0.0, 25.0, d0, 0.5, key=f"d{preset}")
        pik = st.toggle("Paying interest with debt", p0, key=f"p{preset}")
        sw21 = st.toggle("Software loan from 2021", s0, key=f"s{preset}")
    full, base = r.predict(mark, chg, disp, int(pik), int(sw21))
    with right:
        st.markdown('<p class="cap">Chance it stops paying within a year</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="big">{full:.0%}</p>', unsafe_allow_html=True)
        cmp = pd.DataFrame({"": ["Model", "Mark alone"], "chance": [full, base]})
        st.altair_chart(
            alt.Chart(cmp).mark_bar(cornerRadius=3, height=10)
            .encode(x=alt.X("chance:Q", axis=alt.Axis(format="%", title=None, grid=False), scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y(":N", title=None, sort=None),
                    color=alt.Color(":N", scale=alt.Scale(range=[INK, SOFT]), legend=None),
                    tooltip=[alt.Tooltip("chance:Q", format=".1%")])
            .properties(height=90),
            width="stretch",
        )
        ratio = full / base
        verdict = ("Riskier than the mark suggests." if ratio > 1.4
                   else "Safer than the mark suggests." if ratio < 0.75 else "The model agrees with the mark.")
        st.markdown(f'<p class="verdict">{verdict}</p>', unsafe_allow_html=True)

    st.divider()
    st.subheader("What drives the risk")
    w = r.weights
    st.altair_chart(
        alt.Chart(w).mark_bar(cornerRadius=3)
        .encode(x=alt.X("weight:Q", title="Weight (inputs on the same scale)"),
                y=alt.Y("input:N", sort=None, title=None),
                color=alt.condition("datum.weight > 0", alt.value(INK), alt.value(SOFT)),
                tooltip=["input", alt.Tooltip("weight:Q", format="+.2f")])
        .properties(height=220),
        width="stretch",
    )

with tab_lenders:
    st.subheader("Who marks too generously")
    st.caption("Points each lender marks above what its loans deserve. Circles show the real bias in the synthetic data.")
    b = r.lender_bias
    bars = alt.Chart(b).mark_bar(cornerRadius=3).encode(
        x=alt.X("estimated:Q", title="Mark points above deserved"),
        y=alt.Y("lender:N", sort="-x", title=None),
        color=alt.condition("datum.estimated > 0", alt.value(INK), alt.value(SOFT)),
        tooltip=[alt.Tooltip("lender:N"), alt.Tooltip("estimated:Q", format="+.1f"), alt.Tooltip("actual:Q", format="+.1f")],
    )
    dots = alt.Chart(b).mark_point(filled=True, size=90, color="white", stroke=INK, strokeWidth=2).encode(
        x="actual:Q", y=alt.Y("lender:N", sort=alt.EncodingSortField("estimated", order="descending")))
    st.altair_chart((bars + dots).properties(height=320), width="stretch")

with tab_perf:
    a, c = st.columns(2, gap="large")
    with a:
        st.subheader("Model vs. marks")
        st.caption("Log loss on quarters the model never saw. Lower is better.")
        ll = pd.DataFrame({"approach": list(r.logloss), "log loss": list(r.logloss.values())})
        st.altair_chart(
            alt.Chart(ll).mark_bar(cornerRadius=3)
            .encode(x=alt.X("log loss:Q", scale=alt.Scale(domain=[0.3, 0.4]), title=None),
                    y=alt.Y("approach:N", sort=None, title=None),
                    color=alt.condition("datum.approach == 'Full model'", alt.value(INK), alt.value(SOFT)),
                    tooltip=["approach", alt.Tooltip("log loss:Q", format=".3f")])
            .properties(height=180),
            width="stretch",
        )
    with c:
        st.subheader("Calibration")
        st.caption("Dots on the line mean honest probabilities.")
        line = alt.Chart(pd.DataFrame({"x": [0, 0.7], "y": [0, 0.7]})).mark_line(strokeDash=[4, 4], color=SOFT).encode(x="x", y="y")
        pts = alt.Chart(r.calibration).mark_circle(color=INK).encode(
            x=alt.X("predicted:Q", axis=alt.Axis(format="%"), title="Predicted"),
            y=alt.Y("actual:Q", axis=alt.Axis(format="%"), title="Actual"),
            size=alt.Size("loans:Q", legend=None), tooltip=["loans", alt.Tooltip("predicted:Q", format=".0%"), alt.Tooltip("actual:Q", format=".0%")])
        st.altair_chart((line + pts).properties(height=260), width="stretch")
    st.caption(f"{r.data.borrower.nunique()} made-up borrowers and {len(r.data):,} lender marks. "
               f"Trained on 8 quarters, tested on the next 4.")

with tab_how:
    st.markdown(open("HOW_IT_WORKS.md").read())
    st.subheader("The weights, as odds")
    st.caption("How much each input multiplies the odds of a loan going bad, per one standard deviation of that input.")
    odds = r.weights.assign(**{"odds multiplier": np.exp(r.weights.weight).round(2)})
    st.dataframe(odds, hide_index=True, width="stretch")

st.write("")
st.caption("By [Bianca Rubbico](https://biancarubbico.substack.com). Lender names are made up.")
