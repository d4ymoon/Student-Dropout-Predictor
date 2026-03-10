import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


from db import init_db, insert_prediction, fetch_latest
from views.layout import page_header


ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "models" / "student_dropout_xgb.pkl"


@st.cache_resource
def load_model():
    model_bundle = joblib.load(MODEL_PATH)
    return model_bundle["model"], model_bundle["columns"]

def explain_prediction(input_encoded, model, columns):
    importance = model.feature_importances_
    contrib = input_encoded.iloc[0] * importance

    df = pd.DataFrame({
        "Feature": columns,
        "Contribution": contrib
    })

    df = df.sort_values("Contribution", ascending=False).head(10)

    st.subheader("Top Factors Influencing This Prediction")

    fig = px.bar(
        df,
        x="Contribution",
        y="Feature",
        orientation="h"
    )

    fig.update_layout(
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )

    st.plotly_chart(fig, use_container_width=True, key="feature_importance")

def render_gauge(prob):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Dropout Risk (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkred"},
                "steps": [
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 60], "color": "yellow"},
                    {"range": [60, 100], "color": "salmon"},
                ],
            },
        )
    )

    fig.update_layout(height=300)

    st.plotly_chart(fig, use_container_width=True, key="gauge_chart")

def render():
    page_header(
        "Predict Student Dropout",
        "Enter student information to estimate dropout risk"
    )

    init_db()
    model, model_cols = load_model()

    with st.form("prediction_form"):

        # ---------------------------
        # STUDENT INFORMATION
        # ---------------------------
        with st.expander("👤 Student Information", expanded=True):
            col1, col2 = st.columns(2)
            
            age = col1.number_input(
                "Age at enrollment",
                min_value=15,
                max_value=80,
                value=20
            )
            
            gender = col2.selectbox(
                "Gender",
                ["Male", "Female"]
            )
            
            col3, col4 = st.columns(2)
            
            marital_status = col3.selectbox(
                "Marital status",
                ["Single", "Married", "Widower", "Divorced", "Common-law marriage", "Legally separated"]
            )
            
            nationality = col4.selectbox(
                "Nationality",
                ["Portuguese", "German", "Spanish", "Italian", "Dutch", "English", "Lithuanian", 
                 "Angolan", "Cape Verdean", "Guinean", "Mozambican", "Santomean", "Turkish",
                 "Brazilian", "Romanian", "Moldova (Republic of)", "Mexican", "Ukrainian", 
                 "Russian", "Cuban", "Colombian"]
            )

        # ---------------------------
        # COURSE INFORMATION
        # ---------------------------
        with st.expander("🎓 Course Information", expanded=True):
            col1, col2 = st.columns(2)
            
            course = col1.selectbox(
                "Course",
                ["Biofuel Production Technologies", "Animation and Multimedia Design", 
                 "Social Service (evening attendance)", "Agronomy", "Communication Design",
                 "Veterinary Nursing", "Informatics Engineering", "Equiniculture", "Management",
                 "Social Service", "Tourism", "Nursing", "Oral Hygiene", 
                 "Advertising and Marketing Management", "Journalism and Communication",
                 "Basic Education", "Management (evening attendance)"]
            )
            
            attendance = col2.selectbox(
                "Daytime/evening attendance",
                ["Daytime", "Evening"]
            )
            
            col3, col4 = st.columns(2)
            
            application_mode = col3.selectbox(
                "Application mode",
                ["1st phase—general contingent", "Ordinance No. 612/93", 
                 "1st phase—special contingent (Azores Island)", "Holders of other higher courses",
                 "Ordinance No. 854-B/99", "International student (bachelor)",
                 "1st phase—special contingent (Madeira Island)", "2nd phase—general contingent",
                 "3rd phase—general contingent", "Ordinance No. 533-A/99, item b2) (Different Plan)",
                 "Ordinance No. 533-A/99, item b3 (Other Institution)", "Over 23 years old",
                 "Transfer", "Change in course", "Technological specialization diploma holders",
                 "Change in institution/course", "Short cycle diploma holders",
                 "Change in institution/course (International)"]
            )
            
            previous_qual = col4.selectbox(
                "Previous qualification",
                ["Secondary education", "Higher education—bachelor’s degree", 
                 "Higher education—degree", "Higher education—master’s degree",
                 "Higher education—doctorate", "Frequency of higher education",
                 "12th year of schooling—not completed", "11th year of schooling—not completed",
                 "Other—11th year of schooling", "10th year of schooling",
                 "10th year of schooling—not completed", "Basic education 3rd cycle (9th/10th/11th year) or equivalent",
                 "Basic education 2nd cycle (6th/7th/8th year) or equivalent", 
                 "Technological specialization course", "Higher education—degree (1st cycle)",
                 "Professional higher technical course", "Higher education—master’s degree (2nd cycle)"]
            )

        # ---------------------------
        # FAMILY BACKGROUND
        # ---------------------------
        with st.expander("👪 Family Background", expanded=False):
            col1, col2 = st.columns(2)
            
            mother_qual = col1.selectbox(
                "Mother's qualification",
                ["Secondary Education—12th Year or Equivalent", "Higher Education—bachelor’s degree",
                 "Higher Education—degree", "Higher Education—master’s degree", 
                 "Higher Education—doctorate", "Frequency of Higher Education",
                 "12th Year—not completed", "11th Year—not completed", "7th Year (Old)",
                 "Other—11th Year", "2nd year complementary high school course", "10th Year",
                 "General commerce course", "Basic Education 3rd Cycle", "Complementary High School Course",
                 "Technical-professional course", "Complementary High School—not concluded", "7th year",
                 "2nd cycle general high school", "9th Year—not completed", "8th year",
                 "General Course of Administration and Commerce", "Supplementary Accounting and Administration",
                 "Unknown", "Cannot read or write", "Can read without 4th year", "Basic education 1st cycle",
                 "Basic Education 2nd Cycle", "Technological specialization course", 
                 "Higher education—degree (1st cycle)", "Specialized higher studies course",
                 "Professional higher technical course", "Higher Education—master’s degree (2nd cycle)",
                 "Higher Education—doctorate (3rd cycle)"]
            )
            
            father_qual = col2.selectbox(
                "Father's qualification",
                ["Secondary Education—12th Year or Equivalent", "Higher Education—bachelor’s degree",
                 "Higher Education—degree", "Higher Education—master’s degree", 
                 "Higher Education—doctorate", "Frequency of Higher Education",
                 "12th Year—not completed", "11th Year—not completed", "7th Year (Old)",
                 "Other—11th Year", "2nd year complementary high school course", "10th Year",
                 "General commerce course", "Basic Education 3rd Cycle", "Complementary High School Course",
                 "Technical-professional course", "Complementary High School—not concluded", "7th year",
                 "2nd cycle general high school", "9th Year—not completed", "8th year",
                 "General Course of Administration and Commerce", "Supplementary Accounting and Administration",
                 "Unknown", "Cannot read or write", "Can read without 4th year", "Basic education 1st cycle",
                 "Basic Education 2nd Cycle", "Technological specialization course", 
                 "Higher education—degree (1st cycle)", "Specialized higher studies course",
                 "Professional higher technical course", "Higher Education—master’s degree (2nd cycle)",
                 "Higher Education—doctorate (3rd cycle)"]
            )
            
            col3, col4 = st.columns(2)
            
            mother_occ = col3.selectbox(
                "Mother's occupation",
                ["Student", "Legislative/Executive Directors", "Scientific Specialists",
                 "Intermediate Technicians", "Administrative staff", "Services & Sellers",
                 "Agriculture & Fisheries Workers", "Industry & Construction Workers", 
                 "Machine Operators", "Unskilled Workers", "Armed Forces Professions",
                 "Other Situation", "Armed Forces Officers", "Armed Forces Sergeants",
                 "Other Armed Forces", "Admin & Commercial Directors", "Hotel & Trade Directors",
                 "Engineering Specialists", "Health professionals", "Teachers",
                 "Finance & Accounting Specialists", "Science Technicians", "Health Technicians",
                 "Legal/Social Technicians", "ICT Technicians", "Office Workers",
                 "Accounting Operators", "Administrative Support", "Personal Service Workers",
                 "Sellers", "Personal Care Workers", "Security Personnel", "Market Farmers",
                 "Subsistence Farmers", "Construction Workers", "Metal Workers", 
                 "Electrical Workers", "Food & Clothing Industry Workers", "Plant Operators",
                 "Assembly Workers", "Vehicle Drivers", "Unskilled Agriculture Workers",
                 "Unskilled Industry Workers", "Meal Preparation Assistants", "Street Vendors"]
            )
            
            father_occ = col4.selectbox(
                "Father's occupation",
                ["Student", "Legislative/Executive Directors", "Scientific Specialists",
                 "Intermediate Technicians", "Administrative staff", "Services & Sellers",
                 "Agriculture & Fisheries Workers", "Industry & Construction Workers", 
                 "Machine Operators", "Unskilled Workers", "Armed Forces Professions",
                 "Other Situation", "Armed Forces Officers", "Armed Forces Sergeants",
                 "Other Armed Forces", "Admin & Commercial Directors", "Hotel & Trade Directors",
                 "Engineering Specialists", "Health professionals", "Teachers",
                 "Finance & Accounting Specialists", "Science Technicians", "Health Technicians",
                 "Legal/Social Technicians", "ICT Technicians", "Office Workers",
                 "Accounting Operators", "Administrative Support", "Personal Service Workers",
                 "Sellers", "Personal Care Workers", "Security Personnel", "Market Farmers",
                 "Subsistence Farmers", "Construction Workers", "Metal Workers", 
                 "Electrical Workers", "Food & Clothing Industry Workers", "Plant Operators",
                 "Assembly Workers", "Vehicle Drivers", "Unskilled Agriculture Workers",
                 "Unskilled Industry Workers", "Meal Preparation Assistants", "Street Vendors"]
            )

        # ---------------------------
        # ACADEMIC PERFORMANCE
        # ---------------------------
        with st.expander("📚 Academic Performance", expanded=True):
            c1, c2 = st.columns(2)

            grade1 = c1.number_input(
                "1st Sem Grade",
                min_value=0.0,
                max_value=20.0,
                value=10.0
            )

            grade2 = c2.number_input(
                "2nd Sem Grade",
                min_value=0.0,
                max_value=20.0,
                value=10.0
            )

            approved1 = c1.number_input(
                "1st Sem Units Approved",
                min_value=0,
                value=5
            )

            enrolled1 = c2.number_input(
                "1st Sem Units Enrolled",
                min_value=0,
                value=6
            )

            approved2 = c1.number_input(
                "2nd Sem Units Approved",
                min_value=0,
                value=5
            )

            enrolled2 = c2.number_input(
                "2nd Sem Units Enrolled",
                min_value=0,
                value=6
            )

        # ---------------------------
        # ECONOMIC FACTORS
        # ---------------------------
        with st.expander("💰 Economic Indicators", expanded=False):
            c1, c2, c3 = st.columns(3)

            unemployment = c1.number_input(
                "Unemployment Rate",
                value=7.0
            )

            inflation = c2.number_input(
                "Inflation Rate",
                value=2.0
            )

            gdp = c3.number_input(
                "GDP",
                value=1.0
            )
            
            col1, col2 = st.columns(2)
            
            scholarship = col1.selectbox(
                "Scholarship holder",
                ["No", "Yes"]
            )
            
            debtor = col2.selectbox(
                "Debtor",
                ["No", "Yes"]
            )
            
            col3, col4 = st.columns(2)
            
            tuition = col3.selectbox(
                "Tuition Fees Up To Date",
                ["Yes", "No"]
            )

        # ---------------------------
        # OTHER INFORMATION
        # ---------------------------
        with st.expander("📋 Other Information", expanded=False):
            c1, c2 = st.columns(2)

            international = c1.selectbox(
                "International Student",
                ["No", "Yes"]
            )
            
            displaced = c2.selectbox(
                "Displaced",
                ["No", "Yes"]
            )

            special = c1.selectbox(
                "Educational Special Needs",
                ["No", "Yes"]
            )

        submitted = st.form_submit_button(
            "Predict Dropout Risk",
            type="primary",
            use_container_width=True
        )

    # ---------------------------
    # PREDICTION
    # ---------------------------
    if submitted:
        # Create reverse mappings for categorical fields
        gender_map = {"Male": 1, "Female": 0}
        
        marital_map = {
            "Single": 1, "Married": 2, "Widower": 3, "Divorced": 4, 
            "Common-law marriage": 5, "Legally separated": 6
        }
        
        nationality_map = {
            "Portuguese": 1, "German": 2, "Spanish": 3, "Italian": 4, "Dutch": 5,
            "English": 6, "Lithuanian": 7, "Angolan": 8, "Cape Verdean": 9,
            "Guinean": 10, "Mozambican": 11, "Santomean": 12, "Turkish": 13,
            "Brazilian": 14, "Romanian": 15, "Moldova (Republic of)": 16,
            "Mexican": 17, "Ukrainian": 18, "Russian": 19,
            "Cuban": 20, "Colombian": 21
        }
        
        course_map = {
            "Biofuel Production Technologies": 1,
            "Animation and Multimedia Design": 2,
            "Social Service (evening attendance)": 3,
            "Agronomy": 4,
            "Communication Design": 5,
            "Veterinary Nursing": 6,
            "Informatics Engineering": 7,
            "Equiniculture": 8,
            "Management": 9,
            "Social Service": 10,
            "Tourism": 11,
            "Nursing": 12,
            "Oral Hygiene": 13,
            "Advertising and Marketing Management": 14,
            "Journalism and Communication": 15,
            "Basic Education": 16,
            "Management (evening attendance)": 17
        }
        
        attendance_map = {"Daytime": 1, "Evening": 0}
        
        application_map = {
            "1st phase—general contingent": 1,
            "Ordinance No. 612/93": 2,
            "1st phase—special contingent (Azores Island)": 3,
            "Holders of other higher courses": 4,
            "Ordinance No. 854-B/99": 5,
            "International student (bachelor)": 6,
            "1st phase—special contingent (Madeira Island)": 7,
            "2nd phase—general contingent": 8,
            "3rd phase—general contingent": 9,
            "Ordinance No. 533-A/99, item b2) (Different Plan)": 10,
            "Ordinance No. 533-A/99, item b3 (Other Institution)": 11,
            "Over 23 years old": 12,
            "Transfer": 13,
            "Change in course": 14,
            "Technological specialization diploma holders": 15,
            "Change in institution/course": 16,
            "Short cycle diploma holders": 17,
            "Change in institution/course (International)": 18
        }
        
        previous_qual_map = {
            "Secondary education": 1,
            "Higher education—bachelor’s degree": 2,
            "Higher education—degree": 3,
            "Higher education—master’s degree": 4,
            "Higher education—doctorate": 5,
            "Frequency of higher education": 6,
            "12th year of schooling—not completed": 7,
            "11th year of schooling—not completed": 8,
            "Other—11th year of schooling": 9,
            "10th year of schooling": 10,
            "10th year of schooling—not completed": 11,
            "Basic education 3rd cycle (9th/10th/11th year) or equivalent": 12,
            "Basic education 2nd cycle (6th/7th/8th year) or equivalent": 13,
            "Technological specialization course": 14,
            "Higher education—degree (1st cycle)": 15,
            "Professional higher technical course": 16,
            "Higher education—master’s degree (2nd cycle)": 17
        }
        
        # Create a generic mapping function for long lists
        def create_qual_map(qual_list):
            return {name: i+1 for i, name in enumerate(qual_list)}
        
        # Get all qualification and occupation options from above
        mother_qual_options = [
            "Secondary Education—12th Year or Equivalent", "Higher Education—bachelor’s degree",
            "Higher Education—degree", "Higher Education—master’s degree", 
            "Higher Education—doctorate", "Frequency of Higher Education",
            "12th Year—not completed", "11th Year—not completed", "7th Year (Old)",
            "Other—11th Year", "2nd year complementary high school course", "10th Year",
            "General commerce course", "Basic Education 3rd Cycle", "Complementary High School Course",
            "Technical-professional course", "Complementary High School—not concluded", "7th year",
            "2nd cycle general high school", "9th Year—not completed", "8th year",
            "General Course of Administration and Commerce", "Supplementary Accounting and Administration",
            "Unknown", "Cannot read or write", "Can read without 4th year", "Basic education 1st cycle",
            "Basic Education 2nd Cycle", "Technological specialization course", 
            "Higher education—degree (1st cycle)", "Specialized higher studies course",
            "Professional higher technical course", "Higher Education—master’s degree (2nd cycle)",
            "Higher Education—doctorate (3rd cycle)"
        ]
        
        occupation_options = [
            "Student", "Legislative/Executive Directors", "Scientific Specialists",
            "Intermediate Technicians", "Administrative staff", "Services & Sellers",
            "Agriculture & Fisheries Workers", "Industry & Construction Workers", 
            "Machine Operators", "Unskilled Workers", "Armed Forces Professions",
            "Other Situation", "Armed Forces Officers", "Armed Forces Sergeants",
            "Other Armed Forces", "Admin & Commercial Directors", "Hotel & Trade Directors",
            "Engineering Specialists", "Health professionals", "Teachers",
            "Finance & Accounting Specialists", "Science Technicians", "Health Technicians",
            "Legal/Social Technicians", "ICT Technicians", "Office Workers",
            "Accounting Operators", "Administrative Support", "Personal Service Workers",
            "Sellers", "Personal Care Workers", "Security Personnel", "Market Farmers",
            "Subsistence Farmers", "Construction Workers", "Metal Workers", 
            "Electrical Workers", "Food & Clothing Industry Workers", "Plant Operators",
            "Assembly Workers", "Vehicle Drivers", "Unskilled Agriculture Workers",
            "Unskilled Industry Workers", "Meal Preparation Assistants", "Street Vendors"
        ]
        
        mother_qual_map = create_qual_map(mother_qual_options)
        father_qual_map = create_qual_map(mother_qual_options)  # Same options
        mother_occ_map = create_qual_map(occupation_options)
        father_occ_map = create_qual_map(occupation_options)

        inputs = {
            "Age at enrollment": age,
            "Gender": gender_map[gender],
            "Marital status": marital_map[marital_status],
            "Nationality": nationality_map[nationality],
            "Course": course_map[course],
            "Daytime/evening attendance": attendance_map[attendance],
            "Application mode": application_map[application_mode],
            "Previous qualification": previous_qual_map[previous_qual],
            "Mother's qualification": mother_qual_map[mother_qual],
            "Father's qualification": father_qual_map[father_qual],
            "Mother's occupation": mother_occ_map[mother_occ],
            "Father's occupation": father_occ_map[father_occ],
            "Curricular units 1st sem (grade)": grade1,
            "Curricular units 2nd sem (grade)": grade2,
            "Curricular units 1st sem (approved)": approved1,
            "Curricular units 1st sem (enrolled)": enrolled1,
            "Curricular units 2nd sem (approved)": approved2,
            "Curricular units 2nd sem (enrolled)": enrolled2,
            "Unemployment rate": unemployment,
            "Inflation rate": inflation,
            "GDP": gdp,
            "Scholarship holder": 1 if scholarship == "Yes" else 0,
            "Debtor": 1 if debtor == "Yes" else 0,
            "Tuition fees up to date": 1 if tuition == "Yes" else 0,
            "International": 1 if international == "Yes" else 0,
            "Displaced": 1 if displaced == "Yes" else 0,
            "Educational special needs": 1 if special == "Yes" else 0,
        }

        input_df = pd.DataFrame([inputs])
        input_encoded = pd.get_dummies(input_df)

        for col in model_cols:
            if col not in input_encoded:
                input_encoded[col] = 0

        input_encoded = input_encoded[model_cols]

        pred = model.predict(input_encoded)[0]
        prob = model.predict_proba(input_encoded)[0][1]

        insert_prediction(
            datetime.now().isoformat(timespec="seconds"),
            json.dumps(inputs),
            float(prob)
        )

        st.divider()

        col1, col2 = st.columns([1,1])

        with col1:
            render_gauge(prob)

        with col2:
            st.metric(
                "Dropout Probability",
                f"{prob*100:.1f}%"
            )

            if pred == 1:
                st.error("⚠️ High Dropout Risk")
            else:
                st.success("✅ Low Dropout Risk")

            st.progress(float(prob))

        explain_prediction(input_encoded, model, model_cols)

    # ---------------------------
    # HISTORY
    # ---------------------------
    st.divider()
    st.subheader("Recent Predictions")

    rows = fetch_latest(20)

    if rows:
        df_hist = pd.DataFrame(
            rows,
            columns=["id", "created_at", "features_json", "prediction"]
        )

        st.dataframe(
            df_hist,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No predictions yet.")