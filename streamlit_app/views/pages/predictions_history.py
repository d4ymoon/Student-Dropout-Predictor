# views/pages/predictions_history.py

import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go  
import plotly.express as px  
from datetime import datetime
from views.layout import page_header
from db import fetch_all_predictions


# Course name mapping for better display
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


def render():
    page_header(
        "Predictions History",
        "View all prediction records from the database"
    )

    # ---------------------------
    # FETCH ALL PREDICTIONS
    # ---------------------------
    rows = fetch_all_predictions()  # Fetch all predictions from DB
    
    if not rows:
        st.info("No predictions found in the database. Go to the Predict page to make your first prediction!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=["id", "created_at", "features_json", "prediction"])
    
    # ---------------------------
    # SUMMARY STATS
    # ---------------------------
    total_predictions = len(df)
    
    # Calculate risk categories
    df["prediction"] = df["prediction"].astype(float)
    
    def get_risk_category(prob):
        if prob <= 0.3:
            return "Low Risk"
        elif prob <= 0.6:
            return "Medium Risk"
        else:
            return "High Risk"
    
    df["risk_category"] = df["prediction"].apply(get_risk_category)
    df["risk_emoji"] = df["prediction"].apply(
        lambda x: "🟢" if x <= 0.3 else ("🟡" if x <= 0.6 else "🔴")
    )
    
    # Count by category
    risk_counts = df["risk_category"].value_counts()
    low_count = risk_counts.get("Low Risk", 0)
    medium_count = risk_counts.get("Medium Risk", 0)
    high_count = risk_counts.get("High Risk", 0)
    
    # Date range
    df["created_at"] = pd.to_datetime(df["created_at"])
    date_range = f"{df['created_at'].min().strftime('%Y-%m-%d')} to {df['created_at'].max().strftime('%Y-%m-%d')}"
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", f"{total_predictions:,}")
    
    with col2:
        st.metric("Date Range", date_range)
    
    with col3:
        latest_date = df["created_at"].max().strftime("%Y-%m-%d %H:%M")
        st.metric("Latest Prediction", latest_date)
    
    with col4:
        avg_prob = df["prediction"].mean() * 100
        st.metric("Average Probability", f"{avg_prob:.1f}%")
    
    # Risk counts summary
    st.caption(f"🟢 {low_count} Low Risk  |  🟡 {medium_count} Medium Risk  |  🔴 {high_count} High Risk")
    
    st.divider()
    
    # ---------------------------
    # FILTERS
    # ---------------------------
    st.subheader("Filter Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filter by risk category
        risk_filter = st.multiselect(
            "Risk Category",
            options=["Low Risk", "Medium Risk", "High Risk"],
            default=["Low Risk", "Medium Risk", "High Risk"]
        )
    
    with col2:
        # Filter by date range
        min_date = df["created_at"].min().date()
        max_date = df["created_at"].max().date()
        
        date_range_filter = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Handle date range input
        if len(date_range_filter) == 2:
            start_date, end_date = date_range_filter
        else:
            start_date, end_date = min_date, max_date
    
    with col3:
        # Search by ID or features
        search_term = st.text_input("Search (ID or Keywords)", placeholder="Enter search term...")
    
    # Apply filters
    filtered_df = df.copy()
    
    # Filter by risk category
    if risk_filter:
        filtered_df = filtered_df[filtered_df["risk_category"].isin(risk_filter)]
    
    # Filter by date range
    filtered_df = filtered_df[
        (filtered_df["created_at"].dt.date >= start_date) & 
        (filtered_df["created_at"].dt.date <= end_date)
    ]
    
    # Filter by search term
    if search_term:
        # Search in ID (convert to string) and in features_json
        filtered_df = filtered_df[
            filtered_df["id"].astype(str).str.contains(search_term, case=False) |
            filtered_df["features_json"].str.contains(search_term, case=False, na=False)
        ]
    
    st.caption(f"Showing {len(filtered_df)} of {total_predictions} predictions")
    
    st.divider()
    
    # ---------------------------
    # PREDICTIONS TABLE
    # ---------------------------
    st.subheader("All Predictions")
    
    # Prepare display dataframe
    df_display = filtered_df.copy()
    
    # Format probability
    df_display["probability"] = df_display["prediction"].apply(lambda x: f"{x*100:.1f}%")
    
    # Extract key features for preview
    def extract_features_for_display(features_json):
        try:
            features = json.loads(features_json)
            
            # Create a formatted string of key features
            feature_parts = []
            
            # Course (with name mapping if available)
            if "Course" in features:
                course_val = features["Course"]
                # Try to map course number to name
                if isinstance(course_val, (int, float)) and int(course_val) in COURSE_MAP:
                    course_name = COURSE_MAP[int(course_val)]
                    # Shorten if too long
                    if len(course_name) > 25:
                        course_name = course_name[:22] + "..."
                    feature_parts.append(f"{course_name}")
                else:
                    feature_parts.append(f"Course: {course_val}")
            
            # Age
            if "Age at enrollment" in features:
                feature_parts.append(f"Age: {features['Age at enrollment']}")
            
            # Gender
            if "Gender" in features:
                gender = "Male" if features["Gender"] == 1 else "Female"
                feature_parts.append(f"⚥ {gender}")
            
            # Grades
            if "Curricular units 1st sem (grade)" in features:
                grade = features["Curricular units 1st sem (grade)"]
                feature_parts.append(f"Grade: {grade:.1f}")
            
            # Scholarship
            if "Scholarship holder" in features:
                scholarship = "Yes" if features["Scholarship holder"] == 1 else "No"
                feature_parts.append(f"Scholar: {scholarship}")
            
            # Marital status
            if "Marital status" in features:
                marital_map = {1: "Single", 2: "Married", 3: "Widower", 4: "Divorced", 
                              5: "Common-law", 6: "Separated"}
                marital = features["Marital status"]
                if isinstance(marital, (int, float)) and int(marital) in marital_map:
                    feature_parts.append(f"{marital_map[int(marital)]}")
            
            return " | ".join(feature_parts[:4])  # Limit to 4 items
        except Exception as e:
            return "N/A"
    
    df_display["features"] = df_display["features_json"].apply(extract_features_for_display)
    
    # Add risk indicator with emoji
    df_display["risk"] = df_display["risk_emoji"] + " " + df_display["risk_category"]
    
    # Select and rename columns for display
    df_display = df_display[["id", "created_at", "risk", "probability", "features"]]
    df_display.columns = ["ID", "Timestamp", "Risk Level", "Probability", "Features"]
    
    # Sort by timestamp (newest first) - already sorted from DB but just to be sure
    df_display = df_display.sort_values("Timestamp", ascending=False)
    
    # Display the dataframe with custom formatting
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ID": st.column_config.NumberColumn(
                "ID",
                width="small",
                format="%d"
            ),
            "Timestamp": st.column_config.DatetimeColumn(
                "Timestamp",
                format="YYYY-MM-DD HH:mm:ss",
                width="medium"
            ),
            "Risk Level": st.column_config.TextColumn(
                "Risk Level",
                width="small"
            ),
            "Probability": st.column_config.TextColumn(
                "Probability",
                width="small"
            ),
            "Features": st.column_config.TextColumn(
                "Features",
                width="large"
            )
        }
    )
    
    st.divider()
    
    # ---------------------------
    # EXPORT OPTIONS
    # ---------------------------
    st.subheader("Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export current filtered view
        csv_filtered = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered View (CSV)",
            data=csv_filtered,
            file_name=f"predictions_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Export all predictions
        df_all = df.copy()
        df_all["probability"] = df_all["prediction"].apply(lambda x: f"{x*100:.1f}%")
        df_all["risk"] = df_all["risk_emoji"] + " " + df_all["risk_category"]
        df_all["features"] = df_all["features_json"].apply(extract_features_for_display)
        df_all_display = df_all[["id", "created_at", "risk", "probability", "features"]]
        df_all_display.columns = ["ID", "Timestamp", "Risk Level", "Probability", "Features"]
        
        csv_all = df_all_display.to_csv(index=False)
        st.download_button(
            label="📥 Download All Predictions (CSV)",
            data=csv_all,
            file_name=f"predictions_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        # Export raw data as JSON
        json_data = df[["id", "created_at", "features_json", "prediction"]].to_json(orient="records", indent=2)
        st.download_button(
            label="📥 Download Raw JSON",
            data=json_data,
            file_name=f"predictions_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    st.divider()
    
    # ---------------------------
    # STATISTICS
    # ---------------------------
    with st.expander("📊 Detailed Statistics"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk distribution pie chart
            risk_dist = df["risk_category"].value_counts().reset_index()
            risk_dist.columns = ["Risk Category", "Count"]
            
            color_map = {
                "Low Risk": "lightgreen",
                "Medium Risk": "gold",
                "High Risk": "salmon"
            }
            colors = [color_map.get(cat, "gray") for cat in risk_dist["Risk Category"]]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=risk_dist["Risk Category"],
                values=risk_dist["Count"],
                marker=dict(colors=colors),
                textinfo="label+percent",
                hole=0.3
            )])
            fig_pie.update_layout(title="Risk Distribution", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Predictions over time
            df_time = df.copy()
            df_time["date"] = df_time["created_at"].dt.date
            time_series = df_time.groupby("date").size().reset_index()
            time_series.columns = ["Date", "Count"]
            
            fig_line = px.line(
                time_series,
                x="Date",
                y="Count",
                title="Predictions Over Time",
                markers=True
            )
            fig_line.update_layout(height=400)
            st.plotly_chart(fig_line, use_container_width=True)
        
        # Probability statistics
        st.subheader("Probability Statistics")
        stats_df = pd.DataFrame({
            "Metric": ["Minimum", "Maximum", "Mean", "Median", "Standard Deviation"],
            "Value": [
                f"{df['prediction'].min()*100:.1f}%",
                f"{df['prediction'].max()*100:.1f}%",
                f"{df['prediction'].mean()*100:.1f}%",
                f"{df['prediction'].median()*100:.1f}%",
                f"{df['prediction'].std()*100:.2f}%"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)