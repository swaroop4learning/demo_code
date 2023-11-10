import streamlit as st
from multi_app import MultiApp
from view import home, performance_view, explainability_view, drift_default_view, fairness_view, summary_view # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Performance", performance_view.app)
app.add_app("Model explainability", explainability_view.app)
app.add_app("Drift Analysis", drift_default_view.app)
app.add_app("Fairness Analysis", fairness_view.app)
app.add_app("Summary", summary_view.app)

# The main app
app.run()