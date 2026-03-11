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
    # MODEL COMPARISON - WITH YOUR ACTUAL RESULTS
    # ---------------------------
    st.subheader("🤖 Model Performance Comparison")
    
    st.markdown("""
    Four different models were trained and evaluated to find the best performer.
    **XGBoost achieved the highest ROC-AUC score (0.934)** and best accuracy (89.3%).
    """)
    
    # Your actual results from the preprocessing
    comparison_data = pd.DataFrame({
        "Model": [
            "Logistic Regression (Refined)", 
            "Random Forest (Refined)", 
            "Balanced RF (Recall Optimized)",
            "XGBoost (Final)"
        ],
        "Accuracy": [0.879, 0.886, 0.861, 0.893],
        "Precision": [0.850, 0.889, 0.775, 0.874],
        "Recall": [0.757, 0.736, 0.799, 0.778],
        "F1-Score": [0.801, 0.805, 0.787, 0.823],
        "ROC-AUC": [0.918, 0.931, 0.918, 0.934]
    })
    
    # Create comparison chart
    fig_comparison = go.Figure()
    
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, metric in enumerate(metrics):
        fig_comparison.add_trace(go.Bar(
            name=metric,
            x=comparison_data["Model"],
            y=comparison_data[metric],
            text=comparison_data[metric].apply(lambda x: f"{x:.3f}"),
            textposition='outside',
            marker_color=colors[i % len(colors)]
        ))
    
    fig_comparison.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
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
    
    # Load your saved ROC data

    
    # Define path to your ROC data file
    ROOT = Path(__file__).resolve().parents[2]  # Adjust if needed
    ROC_DATA_PATH = ROOT / "data" / "roc_data.pkl"
    
    # Check if file exists
    if os.path.exists(ROC_DATA_PATH):
        with open(ROC_DATA_PATH, "rb") as f:
            roc_data = pickle.load(f)
        
        # Define color mapping for your exact model names
        col_left, col_right = st.columns([2, 1])

        with col_left:
            # --- ROC CURVE LOGIC ---
            color_map = {
                "Logistic Regression": "#1f77b4",
                "Random Forest": "#ff7f0e",
                "Balanced RF": "#2ca02c",
                "XGBoost": "#d62728"
            }
            
            fig_roc = go.Figure()
            sorted_models = sorted(roc_data.items(), key=lambda x: x[1]["auc"], reverse=True)
            
            for model_name, data in sorted_models:
                line_color = color_map.get(model_name, "#808080")
                line_width = 3 if model_name == "XGBoost" else 2
                
                fig_roc.add_trace(go.Scatter(
                    x=data["fpr"],
                    y=data["tpr"],
                    mode='lines',
                    name=f"{model_name}",
                    line=dict(color=line_color, width=line_width)
                ))
            
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='#888', width=1, dash='dash')
            ))
            
            fig_roc.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=450,
                legend=dict(orientation="h", y=-0.2),
                xaxis=dict(range=[0, 1], gridcolor='lightgray'),
                yaxis=dict(range=[0, 1.05], gridcolor='lightgray'),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with col_right:
            # --- AUC SCORE TABLE LOGIC ---
            st.markdown("**AUC Rankings**")
            auc_data = []
            for model_name, data in roc_data.items():
                auc_data.append({"Model": model_name, "AUC": data['auc']})
            
            auc_df = pd.DataFrame(auc_data).sort_values("AUC", ascending=False)
            
            def highlight_top_models(row):
                color = color_map.get(row['Model'], '#808080')
                return [f'background-color: {color}; color: white'] * len(row)
            
            st.dataframe(
                auc_df.style.apply(highlight_top_models, axis=1).format({"AUC": "{:.3f}"}),
                use_container_width=True,
                hide_index=True,
                height=175 
            )
        
    else:
        st.error("❌ ROC data file not found. Please ensure roc_data.pkl exists in the data directory.")
    
    st.divider()

    # ---------------------------
    # CONFUSION MATRICES - WITH YOUR ACTUAL NUMBERS
    # ---------------------------
    st.subheader("📊 Confusion Matrices")
    
    st.markdown("""
    Confusion matrices for all four models, showing true positives, false positives,
    true negatives, and false negatives on the test set.
    """)
    
    # Your actual confusion matrix data from the preprocessing
    cm_data = {
        "Logistic Regression": [[563, 38], [69, 215]],
        "Random Forest": [[575, 26], [75, 209]],
        "Balanced RF": [[535, 66], [57, 227]],
        "XGBoost": [[569, 32], [63, 221]]
    }
    
    # Create 4 columns for the matrices
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.caption("**Logistic Regression**")
        # Reorder the matrix values to match the new label order [Actual Dropout, Actual Graduate]
        cm_reordered = [cm_data["Logistic Regression"][1], cm_data["Logistic Regression"][0]]
        cm_df1 = pd.DataFrame(
            cm_reordered,
            index=["Actual Dropout", "Actual Graduate"],
            columns=["Pred Grad", "Pred Drop"]
        )
        fig_cm1 = create_confusion_matrix_plot(cm_df1, "")
        st.plotly_chart(fig_cm1, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        st.caption("**Random Forest**")
        cm_reordered = [cm_data["Random Forest"][1], cm_data["Random Forest"][0]]
        cm_df2 = pd.DataFrame(
            cm_reordered,
            index=["Actual Dropout", "Actual Graduate"],
            columns=["Pred Grad", "Pred Drop"]
        )
        fig_cm2 = create_confusion_matrix_plot(cm_df2, "")
        st.plotly_chart(fig_cm2, use_container_width=True, config={'displayModeBar': False})
    
    with col3:
        st.caption("**Balanced RF**")
        cm_reordered = [cm_data["Balanced RF"][1], cm_data["Balanced RF"][0]]
        cm_df3 = pd.DataFrame(
            cm_reordered,
            index=["Actual Dropout", "Actual Graduate"],
            columns=["Pred Grad", "Pred Drop"]
        )
        fig_cm3 = create_confusion_matrix_plot(cm_df3, "")
        st.plotly_chart(fig_cm3, use_container_width=True, config={'displayModeBar': False})
    
    with col4:
        st.caption("**XGBoost**")
        cm_reordered = [cm_data["XGBoost"][1], cm_data["XGBoost"][0]]
        cm_df4 = pd.DataFrame(
            cm_reordered,
            index=["Actual Dropout", "Actual Graduate"],
            columns=["Pred Grad", "Pred Drop"]
        )
        fig_cm4 = create_confusion_matrix_plot(cm_df4, "")
        st.plotly_chart(fig_cm4, use_container_width=True, config={'displayModeBar': False})
    
    st.divider()
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
        - **Curricular units 1st sem (credited): The number of curricular units credited by the student in the first semester
        - **Curricular units 2nd sem (credited): The number of curricular units credited by the student in the second semester
        
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
    # MODEL HYPERPARAMETERS
    # ---------------------------
    st.subheader("⚙️ Model Hyperparameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **XGBoost (Final Model)**
        ```python
        {
            'n_estimators': 200,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        ```
                    
        **Balanced Random Forest**
        ```python
        {
            'n_estimators': 400,
            'max_depth': 40,
            'min_samples_split': 15,
            'min_samples_leaf': 4,
            'max_features': 'log2',
            'class_weight': 'balanced',
            'random_state': 42
        }
        ```
        """)
    
    with col2:
        st.markdown("""
        **Random Forest (Refined)**
        ```python
        {
            'n_estimators': 200,
            'max_depth': None,
            'min_samples_split': 5,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42
        }
        ```
        
        **Logistic Regression (Refined)**
        ```python
        {
            'penalty': 'l2',
            'solver': 'liblinear',
            'max_iter': 1000
        }
        ```
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
    """Helper function to create a confusion matrix plot"""
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
        height=180,
        width=180,
        xaxis_title="",
        yaxis_title="",
        xaxis={'side': 'bottom', 'tickfont': {'size': 10}},
        yaxis={'tickfont': {'size': 10}},
        margin=dict(l=5, r=5, t=5, b=5)
    )
    
    return fig   

