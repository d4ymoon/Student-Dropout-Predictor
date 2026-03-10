import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime
import joblib
from pathlib import Path
from views.layout import page_header
from db import fetch_latest


# Define paths
ROOT = Path(__file__).resolve().parents[2]  # Adjust based on your structure
MODEL_PATH = ROOT / "models" / "student_dropout_xgb.pkl"
DATA_PATH = ROOT / "data" / "student_dataset_processed.csv"


# Course name mapping
COURSE_MAP = {
    1: "Biofuel Production Technologies",
    2: "Animation and Multimedia Design",
    3: "Social Service (evening attendance)",
    4: "Agronomy",
    5: "Communication Design",
    6: "Veterinary Nursing",
    7: "Informatics Engineering",
    8: "Equiniculture",
    9: "Management",
    10: "Social Service",
    11: "Tourism",
    12: "Nursing",
    13: "Oral Hygiene",
    14: "Advertising and Marketing Management",
    15: "Journalism and Communication",
    16: "Basic Education",
    17: "Management (evening attendance)"
}


@st.cache_resource
def load_model_and_data():
    """Load model and preprocessed dataset with caching"""
    try:
        # Load model
        model_bundle = joblib.load(MODEL_PATH)
        model = model_bundle["model"]
        model_cols = model_bundle["columns"]
        
        # Load student dataset
        df_students = pd.read_csv(DATA_PATH)
        
        # Map course numbers to names if Course column exists
        if "Course" in df_students.columns:
            df_students["Course_Name"] = df_students["Course"].map(COURSE_MAP)
            # Fill any unmapped values with the original number
            df_students["Course_Name"] = df_students["Course_Name"].fillna(
                df_students["Course"].astype(str) + " (Unknown)"
            )
        
        # Generate predictions for the entire dataset
        available_cols = [col for col in model_cols if col in df_students.columns]
        X = df_students[available_cols].copy()
        
        # Add any missing columns with default values (0)
        for col in model_cols:
            if col not in X.columns:
                X[col] = 0
        
        # Ensure correct column order
        X = X[model_cols]
        
        # Generate predictions
        predictions = model.predict_proba(X)[:, 1]
        df_students["prediction"] = predictions
        
        # Add risk labels with 3 categories
        def get_risk_label(prob):
            if prob <= 0.3:
                return "Low Risk"
            elif prob <= 0.6:
                return "Medium Risk"
            else:
                return "High Risk"
        
        df_students["risk_label"] = df_students["prediction"].apply(get_risk_label)
        
        return df_students
    except FileNotFoundError as e:
        st.error(f"Error loading model or data: {e}")
        return None
    except Exception as e:
        st.error(f"Error generating predictions: {e}")
        return None


def render():
    page_header(
        "Dashboard",
        "Monitor predictions and system activity"
    )

    # ---------------------------
    # LOAD STUDENT DATA WITH PREDICTIONS
    # ---------------------------
    df_students = load_model_and_data()
    
    if df_students is None:
        st.warning("Unable to load student dataset. Please check file paths.")
        return

    # ---------------------------
    # FETCH LATEST PREDICTIONS FROM DB (for recent predictions table only)
    # ---------------------------
    rows = fetch_latest(20)  # Fetch latest 20 predictions from DB
    
    # ---------------------------
    # SUMMARY METRICS (using student dataset)
    # ---------------------------
    total_students = len(df_students)
    
    # Count by risk category
    risk_counts = df_students["risk_label"].value_counts()
    high_risk_count = risk_counts.get("High Risk", 0)
    medium_risk_count = risk_counts.get("Medium Risk", 0)
    low_risk_count = risk_counts.get("Low Risk", 0)
    
    high_risk_pct = (high_risk_count / total_students) * 100
    medium_risk_pct = (medium_risk_count / total_students) * 100
    low_risk_pct = (low_risk_count / total_students) * 100
    
    avg_prob = df_students["prediction"].mean() * 100

    # Display metrics in 4 columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Students",
            f"{total_students:,}"
        )

    with col2:
        st.metric(
            "Low Risk",
            f"{low_risk_count:,} ({low_risk_pct:.1f}%)",
            help="Probability ≤ 30%"
        )

    with col3:
        st.metric(
            "Medium Risk",
            f"{medium_risk_count:,} ({medium_risk_pct:.1f}%)",
            help="Probability 31-60%"
        )

    with col4:
        st.metric(
            "High Risk",
            f"{high_risk_count:,} ({high_risk_pct:.1f}%)",
            delta=f"Avg: {avg_prob:.1f}%",
            delta_color="inverse",
            help="Probability > 60%"
        )

    st.divider()

    # ---------------------------
    # RISK DISTRIBUTION CHARTS (using student dataset)
    # ---------------------------
    # Create two columns for the charts
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # DONUT CHART with 3 categories
        risk_counts_df = df_students["risk_label"].value_counts().reset_index()
        risk_counts_df.columns = ["Risk Category", "Count"]
        
        # Define colors for each risk category
        color_map = {
            "Low Risk": "lightgreen",
            "Medium Risk": "gold",
            "High Risk": "salmon"
        }
        colors = [color_map.get(cat, "gray") for cat in risk_counts_df["Risk Category"]]
        
        # Create donut chart
        fig_donut = go.Figure(data=[go.Pie(
            labels=risk_counts_df["Risk Category"],
            values=risk_counts_df["Count"],
            hole=0.6,  # Creates the donut hole
            marker=dict(colors=colors),
            textinfo="label+percent",
            textposition="inside",
            insidetextorientation="radial",
            sort=False  # Keep original order
        )])
        
        fig_donut.update_layout(
            title="Risk Distribution (All Students)",
            height=350,
            showlegend=False,
            margin=dict(t=50, b=0, l=0, r=0)
        )
        
        st.plotly_chart(fig_donut, use_container_width=True, key="dashboard_donut_chart")
        
        # Add summary text below donut with emojis
        st.caption(
            f"🟢 {low_risk_count:,} Low Risk  |  🟡 {medium_risk_count:,} Medium Risk  |  🔴 {high_risk_count:,} High Risk"
        )
    
    with col2:
        # BAR CHART with 3 categories
        fig_bar = px.histogram(
            df_students,
            x="risk_label",
            color="risk_label",
            title="Risk Distribution by Count (All Students)",
            text_auto=True,
            color_discrete_map=color_map,
            category_orders={"risk_label": ["Low Risk", "Medium Risk", "High Risk"]}
        )
        fig_bar.update_layout(
            xaxis_title="Risk Category",
            yaxis_title="Number of Students",
            showlegend=False,
            height=350,
            margin=dict(t=50, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="dashboard_bar_chart")

    st.divider()

    # ---------------------------
    # ADDITIONAL INSIGHTS FROM STUDENT DATA
    # ---------------------------
    st.subheader("Additional Insights")
    
    # Create tabs for different insights
    tab1, tab2, tab3, tab4 = st.tabs(["Risk Distribution", "Risk by Course", "Risk by Age", "Risk by Gender"])
    
    with tab1:
        # Probability distribution histogram
        fig_dist = px.histogram(
            df_students,
            x="prediction",
            nbins=50,
            title="Distribution of Dropout Probabilities",
            labels={"prediction": "Dropout Probability", "count": "Number of Students"},
            color_discrete_sequence=["#1f77b4"]
        )
        
        # Add vertical lines for risk thresholds
        fig_dist.add_vline(x=0.3, line_dash="dash", line_color="green", 
                          annotation_text="Low/Medium Threshold", annotation_position="top")
        fig_dist.add_vline(x=0.6, line_dash="dash", line_color="red", 
                          annotation_text="Medium/High Threshold", annotation_position="top")
        
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab2:
        # Risk by Course with actual course names
        if "Course" in df_students.columns:
            # Use Course_Name if available, otherwise use Course
            course_col = "Course_Name" if "Course_Name" in df_students.columns else "Course"
            
            # Calculate risk distribution by course
            course_risk_summary = df_students.groupby(course_col)["risk_label"].value_counts().unstack().fillna(0)
            
            # Convert to percentages
            course_risk_pct = course_risk_summary.div(course_risk_summary.sum(axis=1), axis=0) * 100
            
            # Sort by high risk percentage
            if "High Risk" in course_risk_pct.columns:
                course_risk_pct = course_risk_pct.sort_values("High Risk", ascending=False)
            
            # Create stacked bar chart
            fig_course = go.Figure()
            
            risk_categories = ["Low Risk", "Medium Risk", "High Risk"]
            colors = ["lightgreen", "gold", "salmon"]
            
            for i, (category, color) in enumerate(zip(risk_categories, colors)):
                if category in course_risk_pct.columns:
                    fig_course.add_trace(go.Bar(
                        name=category,
                        y=course_risk_pct.index,  # Course names on y-axis for better readability
                        x=course_risk_pct[category],
                        orientation='h',  # Horizontal bars for better course name display
                        marker_color=color,
                        text=course_risk_pct[category].round(1).astype(str) + "%",
                        textposition="inside",
                        insidetextanchor="middle"
                    ))
            
            fig_course.update_layout(
                title="Risk Distribution by Course (%)",
                xaxis_title="Percentage (%)",
                yaxis_title="Course",
                barmode="stack",
                height=max(500, len(course_risk_pct) * 25),  # Dynamic height based on number of courses
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=300)  # Add left margin for course names
            )
            
            # Update x-axis to show percentage
            fig_course.update_xaxes(range=[0, 100])
            
            st.plotly_chart(fig_course, use_container_width=True)
            
            # Add a data table for more detailed view
            with st.expander("View Course Risk Data Table"):
                # Create a summary table
                course_table = course_risk_pct.copy()
                course_table = course_table.round(1)
                course_table.columns = [f"{col} (%)" for col in course_table.columns]
                course_table["Total Students"] = course_risk_summary.sum(axis=1)
                course_table = course_table.sort_values("High Risk (%)", ascending=False)
                
                st.dataframe(
                    course_table,
                    use_container_width=True,
                    column_config={
                        "Total Students": st.column_config.NumberColumn(format="%d")
                    }
                )
        else:
            st.info("Course information not available in dataset")
    
    with tab3:
        # Risk by Age Group
        if "Age at enrollment" in df_students.columns:
            df_students["age_group"] = pd.cut(
                df_students["Age at enrollment"], 
                bins=[0, 20, 25, 30, 35, 40, 100],
                labels=["<20", "20-25", "26-30", "31-35", "36-40", "40+"]
            )
            
            # Calculate risk distribution by age group
            age_risk_summary = df_students.groupby("age_group", observed=True)["risk_label"].value_counts().unstack().fillna(0)
            
            # Convert to percentages
            age_risk_pct = age_risk_summary.div(age_risk_summary.sum(axis=1), axis=0) * 100
            
            # Create stacked bar chart
            fig_age = go.Figure()
            
            for i, (category, color) in enumerate(zip(risk_categories, colors)):
                if category in age_risk_pct.columns:
                    fig_age.add_trace(go.Bar(
                        name=category,
                        x=age_risk_pct.index,
                        y=age_risk_pct[category],
                        marker_color=color,
                        text=age_risk_pct[category].round(1).astype(str) + "%",
                        textposition="inside"
                    ))
            
            fig_age.update_layout(
                title="Risk Distribution by Age Group (%)",
                xaxis_title="Age Group",
                yaxis_title="Percentage (%)",
                barmode="stack",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.info("Age information not available in dataset")
    
    with tab4:
        # Risk by Gender
        if "Gender" in df_students.columns:
            # Map gender if needed
            if df_students["Gender"].dtype in ['int64', 'float64']:
                df_students["Gender_label"] = df_students["Gender"].map({0: "Female", 1: "Male"})
            else:
                df_students["Gender_label"] = df_students["Gender"]
            
            # Calculate risk distribution by gender
            gender_risk_summary = df_students.groupby("Gender_label")["risk_label"].value_counts().unstack().fillna(0)
            
            # Convert to percentages
            gender_risk_pct = gender_risk_summary.div(gender_risk_summary.sum(axis=1), axis=0) * 100
            
            # Create grouped bar chart
            fig_gender = go.Figure()
            
            for i, (category, color) in enumerate(zip(risk_categories, colors)):
                if category in gender_risk_pct.columns:
                    fig_gender.add_trace(go.Bar(
                        name=category,
                        x=gender_risk_pct.index,
                        y=gender_risk_pct[category],
                        marker_color=color,
                        text=gender_risk_pct[category].round(1).astype(str) + "%",
                        textposition="inside"
                    ))
            
            fig_gender.update_layout(
                title="Risk Distribution by Gender (%)",
                xaxis_title="Gender",
                yaxis_title="Percentage (%)",
                barmode="group",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_gender, use_container_width=True)
        else:
            st.info("Gender information not available in dataset")

    st.divider()

    # ---------------------------
    # RECENT PREDICTIONS (from database)
    # ---------------------------
    st.subheader("Recent Predictions from Database")

    if rows:
        df_recent = pd.DataFrame(
            rows,
            columns=["id", "created_at", "features_json", "prediction"]
        )
        
        # Format the dataframe for display
        df_display = df_recent.copy()
        df_display["prediction"] = df_display["prediction"].astype(float)
        df_display["probability_pct"] = df_display["prediction"].apply(lambda x: f"{x*100:.1f}%")
        
        # Add a risk indicator column with 3 categories
        def get_risk_emoji(prob):
            if prob <= 0.3:
                return "🟢 Low Risk"
            elif prob <= 0.6:
                return "🟡 Medium Risk"
            else:
                return "🔴 High Risk"
        
        df_display["risk_level"] = df_display["prediction"].apply(get_risk_emoji)
        
        # Extract a few key features from the JSON to show in a preview column
        def extract_key_features(features_json):
            try:
                features = json.loads(features_json)
                # Show a subset of features
                preview = []
                if "Course" in features:
                    course_val = features["Course"]
                    # Map course number to name if possible
                    if isinstance(course_val, (int, float)) and int(course_val) in COURSE_MAP:
                        course_name = COURSE_MAP[int(course_val)]
                        # Shorten course name for display
                        if len(course_name) > 30:
                            course_name = course_name[:27] + "..."
                        preview.append(f"Course: {course_name}")
                    else:
                        preview.append(f"Course: {course_val}")
                if "Age at enrollment" in features:
                    preview.append(f"Age: {features['Age at enrollment']}")
                if "Curricular units 1st sem (grade)" in features:
                    preview.append(f"Grade: {features['Curricular units 1st sem (grade)']:.1f}")
                return " | ".join(preview[:2])
            except:
                return "N/A"
        
        df_display["features_preview"] = df_recent["features_json"].apply(extract_key_features)
        
        # Select and rename columns for display
        df_display = df_display[["created_at", "risk_level", "probability_pct", "features_preview"]]
        df_display.columns = ["Timestamp", "Risk Level", "Probability", "Key Features"]
        
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Timestamp": st.column_config.DatetimeColumn(
                    "Timestamp",
                    format="YYYY-MM-DD HH:mm:ss"
                ),
                "Probability": st.column_config.TextColumn(
                    "Probability",
                    width="small"
                ),
                "Key Features": st.column_config.TextColumn(
                    "Key Features",
                    width="large"
                )
            }
        )
        
        # Add a download button for the data
        csv = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Download Recent Predictions (CSV)",
            data=csv,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.info("No predictions in database yet. Go to the Predict page to make your first prediction!")

    st.divider()

    # ---------------------------
    # ABOUT SYSTEM
    # ---------------------------
    with st.container():
        st.subheader("About This System")
        st.markdown(
            """
            This system predicts **student dropout risk** using machine learning.

            The model analyzes demographic, academic, and socioeconomic variables to
            estimate the probability that a student may drop out.

            **Risk Categories:**
            - 🟢 **Low Risk**: Probability ≤ 30%
            - 🟡 **Medium Risk**: Probability 31-60%
            - 🔴 **High Risk**: Probability > 60%

            The prediction model was trained using several algorithms including:

            - Logistic Regression
            - Random Forest
            - XGBoost (Final Model)

            XGBoost achieved the best overall performance.

            **Dashboard Data**: Charts and metrics are generated from the student dataset 
            (`student_dataset_processed.csv`) with predictions from the trained model.
            """
        )