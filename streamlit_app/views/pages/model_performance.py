# views/pages/about_model.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import pickle
import os
from pathlib import Path
from views.layout import page_header


def render():
    page_header(
        "📊 About the Model",
        "Understanding how the student dropout prediction model works"
    )

    # ---------------------------
    # MODEL OVERVIEW
    # ---------------------------
    st.subheader("🎯 Model Overview")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **Final Model:** XGBoost Classifier
        
        **Why XGBoost?**
        - Achieved the best overall performance across all metrics
        - Best ROC-AUC score (0.934) indicating excellent discrimination ability
        - Handles mixed data types well (numeric + categorical)
        - Provides feature importance for interpretability
        
        **Training Data:**
        - **4,424** student records
        - **35** original features
        - **236** features after one-hot encoding
        - Binary classification: Dropout (1) vs Graduate/Enrolled (0)
        - Train/Test split: 80/20 (3,539 train / 885 test)
        """)
    
    with col2:
        st.metric("ROC-AUC Score", "0.934", "Best among all models")
        st.metric("Accuracy", "89.3%", "Correct predictions")
        st.metric("Features (encoded)", "236", "After one-hot encoding")
        st.metric("Training Records", "3,539", "80% of dataset")
    
    # Class distribution
    st.caption("**Class Distribution:** 68% Graduate/Enrolled | 32% Dropout")
    
    st.divider()

    # ---------------------------
    # MODEL PERFORMANCE COMPARISON
    # ---------------------------
    st.subheader("🤖 Model Performance Comparison")
    
    st.markdown("""
    The research compared a **Human-Selected** baseline (Logistic Regression) against an **AI-Recommended** ensemble (XGBoost). 
    Both models were evaluated in their initial (default) state and then refined through hyperparameter optimization.
    """)
    
    comparison_data = pd.DataFrame({
        "Model": [
            "LR (Initial)", 
            "LR (Refined)", 
            "XGBoost (Initial)",
            "XGBoost (Refined)"
        ],
        "Accuracy": [0.878, 0.879, 0.883, 0.893],
        "ROC-AUC": [0.917, 0.918, 0.930, 0.935],
        "Recall": [0.757, 0.757, 0.757, 0.778]
    })
    
    fig_comparison = go.Figure()
    metrics = ["Accuracy", "ROC-AUC", "Recall"]
    # Distinct colors for Initial vs Refined states
    colors = ['#27ae60', '#2980b9', '#e6a122', '#ae2727'] 
    
    for i, model_name in enumerate(comparison_data["Model"]):
        model_row = comparison_data[comparison_data["Model"] == model_name]
        fig_comparison.add_trace(go.Bar(
            name=model_name,
            x=metrics,
            y=[model_row["Accuracy"].values[0], model_row["ROC-AUC"].values[0], model_row["Recall"].values[0]],
            text=[f"{model_row['Accuracy'].values[0]:.3f}", f"{model_row['ROC-AUC'].values[0]:.3f}", f"{model_row['Recall'].values[0]:.3f}"],
            textposition='outside',
            marker_color=colors[i]
        ))
    
    fig_comparison.update_layout(
        barmode='group',
        height=500,
        yaxis_range=[0.7, 1.0],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    st.divider()

    # ---------------------------
    # ROC CURVE COMPARISON
    # ---------------------------
    st.subheader("📈 ROC Curve Comparison") 
    
    st.markdown("""
    The ROC curve shows the trade-off between True Positive Rate (Recall) and False Positive Rate.
    **XGBoost achieves the highest AUC (0.934)**, indicating the best ability to distinguish between 
    students who will drop out and those who will persist.
    """)
    
    # Define path to your ROC data file
    ROOT = Path(__file__).resolve().parents[2] 
    ROC_DATA_PATH = ROOT / "data" / "roc_data.pkl"
    
    # Check if file exists
    if os.path.exists(ROC_DATA_PATH):
        with open(ROC_DATA_PATH, "rb") as f:
            roc_data = pickle.load(f)
        
        col_left, col_right = st.columns([2, 1])

        with col_left:
            # Color map matching your ROC script
            color_map = {
                "Logistic Regression (Initial)": "#27ae60",
                "Logistic Regression (Refined)": "#2980b9",
                "XGBoost (Initial)": "#e6a122",
                "XGBoost (Refined)": "#ae2727"
            }
            
            fig_roc = go.Figure()
            for model_name, data in roc_data.items():
                fig_roc.add_trace(go.Scatter(
                    x=data["fpr"], y=data["tpr"],
                    mode='lines',
                    name=f"{model_name}",
                    line=dict(color=color_map.get(model_name, "#808080"), width=3 if "Refined" in model_name else 2)
                ))
            
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='#888', dash='dash')))
            
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=450,
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with col_right:
            st.markdown("**AUC Rankings**")
            
            # 1. Create the DataFrame
            auc_data = [{"Model": m, "AUC": d['auc']} for m, d in roc_data.items()]
            auc_df = pd.DataFrame(auc_data).sort_values("AUC", ascending=False)
            
            color_map = {
                "Logistic Regression (Initial)": "#27ae60", 
                "Logistic Regression (Refined)": "#2980b9", 
                "XGBoost (Initial)": "#e6a122",            
                "XGBoost (Refined)": "#ae2727"             
            }
            
            def style_rows(row):
                color = color_map.get(row['Model'], '#808080')
                return [f'background-color: {color}; color: white; font-weight: bold'] * len(row)
            
            styled_df = auc_df.style.apply(style_rows, axis=1).format({"AUC": "{:.4f}"})
            
            st.dataframe(
                styled_df, 
                use_container_width=True, 
                hide_index=True
            )
        
    else:
        st.error("❌ ROC data file not found. Please ensure roc_data.pkl exists in the data directory.")
    
    st.divider()

    # ---------------------------
    # CONFUSION MATRICES
    # ---------------------------
    st.subheader("📊 Confusion Matrices")
    
    cm_data = {
        "Logisitc Regression (Initial)": [[562, 39], [69, 215]],
        "Logistic Regression (Refined)": [[563, 38], [69, 215]],
        "XGBoost (Initial)": [[566, 35], [69, 215]],
        "XGBoost (Refined)": [[569, 32], [63, 221]]
    }
    
    cols = st.columns(4)
    for i, (name, matrix) in enumerate(cm_data.items()):
        with cols[i]:
            st.caption(f"**{name}**")
            
            # matrix[0] = [TN, FP] -> Actual Grad
            # matrix[1] = [FN, TP] -> Actual Drop
            cm_df = pd.DataFrame(
                matrix,
                index=["Actual Grad", "Actual Drop"], 
                columns=["Pred Grad", "Pred Drop"]
            )
            
            # Pass the dataframe to the helper function
            fig_cm = create_confusion_matrix_plot(cm_df, "")
            st.plotly_chart(fig_cm, use_container_width=True, config={'displayModeBar': False})

    # ---------------------------
    # FEATURE IMPORTANCE - WITH YOUR ACTUAL TOP FEATURES
    # ---------------------------
    st.subheader("🔑 Top Features Influencing Dropout")
    
    st.markdown("""
    Based on XGBoost's feature importance analysis, these are the most influential factors 
    in predicting student dropout. **Academic performance metrics dominate the top spots.**
    """)
    
    # Your actual top features from the preprocessing (from the XGBoost importance output)
    feature_importance_data = {
        "Feature": [
            "Curricular units 2nd sem (approved)",
            "Course - Animation and Multimedia Design",
            "Tuiton fees up to date",
            "Curricular units 1st sem (approved)",
            "Curricular units 2nd sem (grade)",
            "Curricular units 1st sem (enrolled)",
            "Debtor",
            "Father's qualification - Supplementary Accounting and Administration",
            "Curricular units 2nd sem (credited)",
            "Curricular units 2nd sem (enrolled)",
            "Mother's qualification - 7th year of schooling",
            "Scholarship holder",
            "Course - Basic Education",
            "Age at enrollment",
            "Curricular units 1st sem (credited)"
        ],
        "Importance": [
            0.113, 0.067, 0.055, 0.039, 0.030,
            0.019, 0.018, 0.016, 0.015, 0.015,
            0.014, 0.013, 0.013, 0.012, 0.012
        ]
    }
    
    df_importance = pd.DataFrame(feature_importance_data)
    df_importance = df_importance.sort_values("Importance", ascending=True)
    
    fig_importance = px.bar(
        df_importance.tail(15),
        x="Importance",
        y="Feature",
        orientation='h',
        title="Top 15 Feature Importances (XGBoost)",
        text_auto='.3f',
        color="Importance",
        color_continuous_scale=['lightblue', 'darkblue']
    )
    
    fig_importance.update_layout(
        height=600,
        xaxis_title="Importance Score",
        yaxis_title="",
        showlegend=False
    )
    
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Feature explanations
    with st.expander("📝 Detailed Feature Explanations"):
        st.markdown("""
        ### Top Academic Features
        - **Curricular units 2nd sem (approved)**: Number of courses passed in second semester
        - **Curricular units 1st sem (approved)**: Number of courses passed in first semester
        - **Curricular units 1st sem (grade)**: Average grade in first semester courses
        - **Curricular units 2nd sem (enrolled)**: Number of courses enrolled in second semester
        - **Curricular units 1st sem (enrolled)**: Number of courses enrolled in first semester
        - **Curricular units 1st sem (credited)**: The number of curricular units credited by the student in the first semester
        - **Curricular units 2nd sem (credited)**: The number of curricular units credited by the student in the second semester
        
        ### Demographic Features
        - **Age at enrollment**: Student's age when enrolling (older students may have different risk patterns)
        
        ### Socioeconomic Features
        - **Tuition fees up to date**: Whether tuition payments are current
        - **Debtor**: Whether student has outstanding debts

        """)
    
    st.divider()

    # ---------------------------
    # DATA PREPROCESSING SUMMARY
    # ---------------------------
    st.subheader("🔄 Data Preprocessing Pipeline")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Original Dataset Features:**
        - 4,424 student records
        - 35 original features
        - No missing values
        - No duplicates
        
        **Categorical Features Encoded (11 total):**
        - Marital status (6 categories)
        - Course (17 categories)
        - Application mode (18 categories)
        - Previous qualification (17 categories)
        - Nationality (21 categories)
        - Mother's qualification (29 categories)
        - Father's qualification (34 categories)
        - Mother's occupation (32 categories)
        - Father's occupation (46 categories)
        - Daytime/evening attendance (2)
        """)
    
    with col2:
        st.markdown("""
        **After One-Hot Encoding:**
        - **236 total features**
        - Train-test split: 80/20
        - Standard scaling applied for Logistic Regression only
        - No scaling needed for tree-based models
        """)
    
    # Show some of the dropped feature candidates
    with st.expander("🔍 Low-Importance Feature Candidates"):
        st.markdown("""
        The following features had very low importance (<1%) in both Random Forest and XGBoost models,
        making them candidates for potential removal in a simplified model:
        
        - Various nationality indicators (e.g., Nationality_15, Nationality_18)
        - Specific occupation categories
        - Some parental qualification levels
        - Application mode variations
        
        *Note: The final production model retains all features for maximum accuracy.*
        """)
    
    st.divider()

    # ---------------------------
    # KEY INSIGHTS
    # ---------------------------
    st.subheader("💡 Key Insights from the Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **📚 Academic Performance is Critical**
        
        The top 4 features are all grade-related:
        - First and second semester grades
        - Number of approved courses
        - Students with lower grades are at much higher risk
        """)
    
    with col2:
        st.info("""
        **💰 Financial Factors Matter**
        
        - Tuition payment status
        - Scholarship holder status
        - Debtor status
        - Financial stability is a strong predictor of persistence
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.warning("""
        **🌍 Economic Context**
        
        - GDP, unemployment, and inflation rates
        - Broader economic conditions affect dropout risk
        - Students may be more vulnerable during economic downturns
        """)
    
    with col2:
        st.info("""
        **👤 Demographics Play a Role**
        
        - Age at enrollment is a top-10 feature
        - Older students may have different risk factors
        - Different courses have varying dropout rates
        """)
    
    st.divider()

    # ---------------------------
    # CONCLUSION
    
def create_confusion_matrix_plot(cm_df, title):
    """Helper function to create a confusion matrix plot with fixed row order"""
    fig = go.Figure(data=go.Heatmap(
        z=cm_df.values,
        x=cm_df.columns,
        y=cm_df.index,
        text=cm_df.values,
        texttemplate="%{text}",
        textfont={"size": 12},
        colorscale='Blues',
        showscale=False,
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        height=200, 
        width=200,
        xaxis_title="",
        yaxis_title="",
        xaxis={'side': 'bottom', 'tickfont': {'size': 10}},
        # CRITICAL FIX: forces index[0] (Actual Grad) to the top
        yaxis={'tickfont': {'size': 10}, 'autorange': 'reversed'}, 
        margin=dict(l=5, r=5, t=5, b=5)
    )
    
    return fig

