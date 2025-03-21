import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load data
@st.cache_data
def load_data():
    return pd.read_excel("ml_foundations_augmented_from_sample_846.xlsx")

df = load_data()
st.title("ML Foundations Dashboard")

st.sidebar.header("Filter Options")

# Filters (if columns exist)
if 'completion_status' in df.columns:
    completion_filter = st.sidebar.multiselect(
        "Completion Status", options=df['completion_status'].dropna().unique(), default=df['completion_status'].dropna().unique())
    df = df[df['completion_status'].isin(completion_filter)]

st.header("Summary Statistics")
st.dataframe(df.describe())

st.header("Distributions")
col1, col2 = st.columns(2)

with col1:
    if 'final_score' in df.columns:
        df['final_score'] = pd.to_numeric(df['final_score'], errors='coerce')
        fig1, ax1 = plt.subplots()
        sns.histplot(df['final_score'].dropna(), kde=True, bins=20, ax=ax1, color='skyblue')
        ax1.set_title("Final Score Distribution")
        st.pyplot(fig1)

with col2:
    if 'attendance_final_score' in df.columns:
        df['attendance_final_score'] = pd.to_numeric(df['attendance_final_score'], errors='coerce')
        fig2, ax2 = plt.subplots()
        sns.histplot(df['attendance_final_score'].dropna(), kde=True, bins=20, ax=ax2, color='lightgreen')
        ax2.set_title("Attendance Score Distribution")
        st.pyplot(fig2)

st.header("Correlation Heatmap")
numeric_df = df.select_dtypes(include=['float64', 'int64'])
fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, fmt=".2f", ax=ax_corr)
ax_corr.set_title("Correlation Between Numeric Metrics")
st.pyplot(fig_corr)

st.header("Assignment Score Comparison")
numeric_cols = numeric_df.columns.tolist()
if len(numeric_cols) >= 6:
    fig_box, ax_box = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df[numeric_cols[:6]].apply(pd.to_numeric, errors='coerce'), ax=ax_box)
    ax_box.set_title("Boxplot of First 6 Assignment/Quiz Scores")
    ax_box.set_xticklabels(ax_box.get_xticklabels(), rotation=45)
    st.pyplot(fig_box)

st.header("Per Student Grade and Attendance Distribution")
if 'id' in df.columns:
    student_ids = df['id'].dropna().unique()
    selected_id = st.selectbox("Select Student ID", student_ids)
    student_data = df[df['id'] == selected_id].select_dtypes(include=['float64', 'int64'])

    if not student_data.empty:
        fig_student, ax_student = plt.subplots(figsize=(10, 4))
        student_data = student_data.T
        student_data.columns = ['Value']
        sns.barplot(x=student_data.index, y='Value', data=student_data.reset_index(), ax=ax_student, palette="viridis")
        ax_student.set_title(f"Grades and Attendance for Student {selected_id}")
        ax_student.set_xticklabels(ax_student.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig_student)

st.success("Dashboard rendered successfully!")
