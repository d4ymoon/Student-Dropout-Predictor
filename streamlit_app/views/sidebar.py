import streamlit as st

PAGES = [
    "Dashboard",
    "Predict Dropout",
    "Predictions History",      
    "Model Performance",
]

def render_sidebar():

    with st.sidebar:

        st.title("Student Dropout Predictor")
        st.divider()

        if "active_page" not in st.session_state:
            st.session_state.active_page = "Dashboard"

        for page in PAGES:

            is_active = st.session_state.active_page == page
            btn_type = "primary" if is_active else "secondary"

            if st.button(
                page,
                key=f"nav_{page}",
                type=btn_type,
                use_container_width=True,
            ):
                st.session_state.active_page = page
                st.rerun()

        st.divider()

        st.caption("Student Dropout ML System")

    return st.session_state.active_page