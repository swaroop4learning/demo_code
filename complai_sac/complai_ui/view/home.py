import streamlit as st

def app(something):
    st.header('AI 360° Monitor for Model Explanation, Fairness and Performance Monitoring')
    st.markdown(
    """
    # Introduction
    ComplAI is an interactive 360° Machine Learning Model Scanning Tool. Supports Binary, Multi-class and Regression models.

    # Features
    1. **Model Performance Monitor** - Overall Model Performance, Custom Threshold Monitor, Custom Slice Based Model Performance
    2. **Model Explainability Monitor** - Instance Level Explanation, Global Level Explanation, Nearest Counterfactual Explanation, What-If Interactive Explanation
    3. **Model Fairness Monitor** - Correlation Matrix, Disparate Impact, Counterfactual Flip Test
    4. **Model Drift Monitor** - Overall Model Performance on Live Data, Drifted Data, Detection of Model's susceptibility to drift.
    5. **Overall Model Report** - Overall Model Production Readiness Scores
    """
    )