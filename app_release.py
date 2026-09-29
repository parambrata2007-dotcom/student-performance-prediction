import streamlit as st
import pandas as pd
import joblib
import os
import urllib.request

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD MODELS, PREPROCESSORS AND DATASET
# ---------------------------------------------------------

# Large model files are stored as GitHub Release assets because
# GitHub's normal repository upload has a per-file size limit.
RELEASE_BASE_URL = (
    "https://github.com/parambrata2007-dotcom/"
    "student-performance-prediction/releases/download/v1.0"
)

MODEL_FILES = {
    "regression_model.pkl": f"{RELEASE_BASE_URL}/regression_model.pkl",
    "classification_model.pkl": f"{RELEASE_BASE_URL}/classification_model.pkl",
    "preprocessor.pkl": f"{RELEASE_BASE_URL}/preprocessor.pkl",
    "preprocessor_class.pkl": f"{RELEASE_BASE_URL}/preprocessor_class.pkl",
}


def download_if_missing(filename):
    if not os.path.exists(filename):
        st.info(f"Downloading {filename}...")
        urllib.request.urlretrieve(MODEL_FILES[filename], filename)


@st.cache_resource
def load_models():
    for filename in MODEL_FILES:
        download_if_missing(filename)

    regression_model = joblib.load("regression_model.pkl")
    classification_model = joblib.load("classification_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    preprocessor_class = joblib.load("preprocessor_class.pkl")

    return (
        regression_model,
        classification_model,
        preprocessor,
        preprocessor_class
    )


@st.cache_data
def load_dataset():
    return pd.read_csv("Student_Performance.csv")


regression_model, classification_model, preprocessor, preprocessor_class = load_models()
df = load_dataset()

# ---------------------------------------------------------
# RECOMMENDATION SYSTEM
# ---------------------------------------------------------

def generate_recommendations(student, predicted_category):

    recommendations = []

    # 1. Study hours
    if student["study_hours"] < 3:
        recommendations.append(
            "Increase your daily study hours and maintain a consistent study routine."
        )

    # 2. Attendance
    if student["attendance_percentage"] < 75:
        recommendations.append(
            "Improve your attendance and try to maintain it above 75%."
        )

    # 3. Find weakest subject
    subjects = {
        "Mathematics": student["math_score"],
        "Science": student["science_score"],
        "English": student["english_score"]
    }

    weakest_subject = min(subjects, key=subjects.get)
    weakest_score = subjects[weakest_subject]

    if weakest_score < 60:
        recommendations.append(
            f"Focus more on {weakest_subject}; your score is {weakest_score:.1f}."
        )

    # 4. Performance category
    if predicted_category == "Low":
        recommendations.append(
            "Your predicted performance is Low. Attend additional practice "
            "sessions and increase your study consistency."
        )

    elif predicted_category == "Average":
        recommendations.append(
            "Your predicted performance is Average. Increase your study effort "
            "and practice regularly to improve."
        )

    else:
        recommendations.append(
            "Your predicted performance is High. Maintain your current study "
            "routine and continue practicing."
        )

    return recommendations


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🎓 Student Performance Prediction System")

st.write(
    "Enter student information to predict academic performance "
    "and receive personalized recommendations."
)

# ---------------------------------------------------------
# STUDENT INFORMATION
# ---------------------------------------------------------

st.divider()
st.header("👤 Student Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=14,
        max_value=19,
        value=18
    )

    gender = st.selectbox(
        "Gender",
        ["male", "female"]
    )

    school_type = st.selectbox(
        "School Type",
        ["public", "private"]
    )

    parent_education = st.selectbox(
        "Parent Education",
        ["high school", "graduate", "postgraduate"]
    )

with col2:

    study_hours = st.number_input(
        "Study Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=3.0,
        step=0.5
    )

    attendance_percentage = st.number_input(
        "Attendance Percentage",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0
    )

    internet_access = st.selectbox(
        "Internet Access",
        ["yes", "no"]
    )

    travel_time = st.selectbox(
        "Travel Time",
        ["<15 min", "15-30 min", "30-60 min", ">60 min"]
    )

    extra_activities = st.selectbox(
        "Extra Activities",
        ["yes", "no"]
    )

    study_method = st.selectbox(
        "Study Method",
        ["textbook", "online videos", "group study", "self-study"]
    )


# ---------------------------------------------------------
# SUBJECT-WISE MARKS
# ---------------------------------------------------------

st.divider()
st.header("📚 Subject-wise Marks")

col1, col2, col3 = st.columns(3)

with col1:
    math_score = st.number_input(
        "Mathematics Score",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0
    )

with col2:
    science_score = st.number_input(
        "Science Score",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0
    )

with col3:
    english_score = st.number_input(
        "English Score",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0
    )


# ---------------------------------------------------------
# RUN PREDICTION
# ---------------------------------------------------------

st.divider()

run_prediction = st.button(
    "🚀 Run Prediction",
    use_container_width=True
)

if run_prediction:

    # Features used by the trained ML models.
    # Subject scores are intentionally not included because
    # the trained models were built without them.
    prediction_data = {
        "age": age,
        "gender": gender,
        "school_type": school_type,
        "parent_education": parent_education,
        "study_hours": study_hours,
        "attendance_percentage": attendance_percentage,
        "internet_access": internet_access,
        "travel_time": travel_time,
        "extra_activities": extra_activities,
        "study_method": study_method
    }

    student_df = pd.DataFrame([prediction_data])

    # -----------------------------------------------------
    # REGRESSION PREDICTION
    # -----------------------------------------------------

    student_reg_transformed = preprocessor.transform(student_df)

    predicted_score = regression_model.predict(
        student_reg_transformed
    )[0]

    # Keep the score within the natural 0-100 range.
    predicted_score = max(0, min(100, predicted_score))

    # -----------------------------------------------------
    # CLASSIFICATION PREDICTION
    # -----------------------------------------------------

    student_class_transformed = preprocessor_class.transform(student_df)

    predicted_category = classification_model.predict(
        student_class_transformed
    )[0]

    # -----------------------------------------------------
    # STUDENT DICTIONARY FOR RECOMMENDATIONS
    # -----------------------------------------------------

    student = {
        "age": age,
        "gender": gender,
        "school_type": school_type,
        "parent_education": parent_education,
        "study_hours": study_hours,
        "attendance_percentage": attendance_percentage,
        "internet_access": internet_access,
        "travel_time": travel_time,
        "extra_activities": extra_activities,
        "study_method": study_method,
        "math_score": math_score,
        "science_score": science_score,
        "english_score": english_score
    }

    recommendations = generate_recommendations(
        student,
        predicted_category
    )

    # -----------------------------------------------------
    # PREDICTION RESULTS
    # -----------------------------------------------------

    st.divider()
    st.header("📊 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Overall Score",
            f"{predicted_score:.2f}"
        )

    with col2:
        st.metric(
            "Predicted Performance",
            predicted_category
        )

    # -----------------------------------------------------
    # RECOMMENDATIONS
    # -----------------------------------------------------

    st.divider()
    st.header("💡 Personalized Recommendations")

    for recommendation in recommendations:
        st.info(recommendation)

    # -----------------------------------------------------
    # STUDENT SUMMARY
    # -----------------------------------------------------

    st.divider()
    st.header("📋 Student Summary")

    summary = pd.DataFrame({
        "Metric": [
            "Age",
            "Study Hours per Day",
            "Attendance Percentage",
            "Mathematics Score",
            "Science Score",
            "English Score",
            "Predicted Overall Score",
            "Predicted Performance"
        ],
        "Value": [
            age,
            study_hours,
            attendance_percentage,
            math_score,
            science_score,
            english_score,
            round(predicted_score, 2),
            predicted_category
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------
# DATASET VISUALIZATIONS
# ---------------------------------------------------------

st.divider()
st.header("📈 Student Performance Analytics")

# ---------------------------------------------------------
# VISUALIZATION 1: SUBJECT-WISE PERFORMANCE
# ---------------------------------------------------------

st.subheader("📚 Subject-wise Performance")

score_comparison = pd.DataFrame({
    "Subject": [
        "Mathematics",
        "Science",
        "English"
    ],
    "Score": [
        math_score,
        science_score,
        english_score
    ]
})

st.bar_chart(
    score_comparison.set_index("Subject")
)


# ---------------------------------------------------------
# VISUALIZATION 2: STUDY HOURS VS OVERALL SCORE
# ---------------------------------------------------------

st.subheader("📖 Study Hours vs Overall Score")

study_plot = df[
    ["study_hours", "overall_score"]
].copy()

st.scatter_chart(
    study_plot,
    x="study_hours",
    y="overall_score"
)


# ---------------------------------------------------------
# VISUALIZATION 3: ATTENDANCE VS OVERALL SCORE
# ---------------------------------------------------------

st.subheader("📝 Attendance vs Overall Score")

attendance_plot = df[
    ["attendance_percentage", "overall_score"]
].copy()

st.scatter_chart(
    attendance_plot,
    x="attendance_percentage",
    y="overall_score"
)
