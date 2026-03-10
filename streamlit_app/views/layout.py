import streamlit as st


def page_config():
    st.set_page_config(
        page_title="Student Dropout Prediction",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def page_header(title: str, subtitle: str = ""):
    st.title(title)
    if subtitle:
        st.caption(subtitle)