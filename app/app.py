import json
import sys
from pathlib import Path

import streamlit as st


# ----------------------------------
# PATHS
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = BASE_DIR / "src"
FEATURES_PATH = BASE_DIR / "data" / "processed" / "feature_info.json"

# Allow importing predict.py from src/
sys.path.append(str(SRC_DIR))

from predict import predict_student


# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide"
)

# ----------------------------------
# CUSTOM STYLING
# ----------------------------------

st.markdown(
    """
<style>

.block-container {
    max-width: 1100px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}

.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

.main-subtitle {
    text-align: center;
    color: #9ca3af;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

.section-title {
    font-size: 1.45rem;
    font-weight: 650;
    margin-top: 1rem;
    margin-bottom: 1.2rem;
}

.prediction-card {
    padding: 1.8rem;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.10);
    background: rgba(255, 255, 255, 0.035);
    margin-top: 1rem;
    margin-bottom: 1.5rem;
    text-align: center;
}

.prediction-label {
    font-size: 1rem;
    color: #9ca3af;
    margin-bottom: 0.5rem;
}

.prediction-value {
    font-size: 2.3rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.prediction-description {
    color: #9ca3af;
    font-size: 0.9rem;
}

.probability-card {
    padding: 1.2rem;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.025);
    text-align: center;
}

.probability-title {
    font-size: 0.9rem;
    color: #9ca3af;
    margin-bottom: 0.4rem;
}

.probability-value {
    font-size: 1.8rem;
    font-weight: 650;
}

.disclaimer {
    text-align: center;
    color: #6b7280;
    font-size: 0.8rem;
    margin-top: 1.5rem;
}

</style>
""",
    unsafe_allow_html=True
)

# ----------------------------------
# LOAD FEATURE INFORMATION
# ----------------------------------

with open(FEATURES_PATH, "r", encoding="utf-8") as file:
    feature_info = json.load(file)


# ----------------------------------
# HELP TEXT
# ----------------------------------

HELP_TEXT = {
    "Marital status":
        "The student's marital status.",

    "Application mode":
        "How the student applied for admission.",

    "Application order":
        "The student's preference order for this course, from first choice to last choice.",

    "Course":
        "The academic course in which the student is enrolled.",

    "Daytime/evening attendance":
        "Whether the student attends classes during the daytime or evening.",

    "Previous qualification":
        "The student's qualification before enrolling in the institution.",

    "Nacionality":
        "The student's nationality.",

    "Mother's qualification":
        "The highest educational qualification of the student's mother.",

    "Father's qualification":
        "The highest educational qualification of the student's father.",

    "Mother's occupation":
        "The occupation of the student's mother.",

    "Father's occupation":
        "The occupation of the student's father.",

    "Displaced":
        "Whether the student moved from their usual residence to attend the institution.",

    "Educational special needs":
        "Whether the student has declared educational special needs.",

    "Debtor":
        "Whether the student is currently a debtor.",

    "Tuition fees up to date":
        "Whether the student's tuition fees are up to date.",

    "Gender":
        "The student's gender.",

    "Scholarship holder":
        "Whether the student is a scholarship holder.",

    "International":
        "Whether the student is an international student.",

    "Age at enrollment":
        "The student's age when they enrolled.",

    # First semester
    "Curricular units 1st sem (credited)":
        "Number of curricular units credited during the 1st semester.",

    "Curricular units 1st sem (enrolled)":
        "Number of curricular units the student enrolled in during the 1st semester.",

    "Curricular units 1st sem (evaluations)":
        "Number of evaluations for curricular units during the 1st semester.",

    "Curricular units 1st sem (approved)":
        "Number of curricular units approved during the 1st semester.",

    "Curricular units 1st sem (grade)":
        "Average grade during the 1st semester, from 0 to 20.",

    "Curricular units 1st sem (without evaluations)":
        "Number of curricular units without evaluations during the 1st semester.",

    # Second semester
    "Curricular units 2nd sem (credited)":
        "Number of curricular units credited during the 2nd semester.",

    "Curricular units 2nd sem (enrolled)":
        "Number of curricular units the student enrolled in during the 2nd semester.",

    "Curricular units 2nd sem (evaluations)":
        "Number of evaluations for curricular units during the 2nd semester.",

    "Curricular units 2nd sem (approved)":
        "Number of curricular units approved during the 2nd semester.",

    "Curricular units 2nd sem (grade)":
        "Average grade during the 2nd semester, from 0 to 20.",

    "Curricular units 2nd sem (without evaluations)":
        "Number of curricular units without evaluations during the 2nd semester.",
}


# ----------------------------------
# HEADER
# ----------------------------------

st.markdown(
    '<div class="main-title">🎓 Student Dropout Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-subtitle">
        Predict a student's academic outcome using Machine Learning.
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ----------------------------------
# FORM
# ----------------------------------

with st.form("student_form"):

    # ----------------------------------
    # STUDENT INFORMATION
    # ----------------------------------

    st.markdown(
    '<div class="section-title">👤 Student Information</div>',
    unsafe_allow_html=True
)

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age at enrollment",
            min_value=1,
            max_value=99,
            value=20,
            step=1,
            help=HELP_TEXT["Age at enrollment"]
        )

        gender_options = feature_info["Gender"]["values"]

        gender = st.selectbox(
            "Gender",
            options=list(gender_options.keys()),
            format_func=lambda x: gender_options[x],
            help=HELP_TEXT["Gender"]
        )

        nationality_options = feature_info["Nacionality"]["values"]

        nationality = st.selectbox(
            "Nationality",
            options=list(nationality_options.keys()),
            format_func=lambda x: nationality_options[x],
            help=HELP_TEXT["Nacionality"]
        )

        international_options = feature_info["International"]["values"]

        international = st.selectbox(
            "International student",
            options=list(international_options.keys()),
            format_func=lambda x: international_options[x],
            help=HELP_TEXT["International"]
        )

    with col2:

        marital_options = feature_info["Marital status"]["values"]

        marital_status = st.selectbox(
            "Marital status",
            options=list(marital_options.keys()),
            format_func=lambda x: marital_options[x],
            help=HELP_TEXT["Marital status"]
        )

        displaced_options = feature_info["Displaced"]["values"]

        displaced = st.selectbox(
            "Displaced",
            options=list(displaced_options.keys()),
            format_func=lambda x: displaced_options[x],
            help=HELP_TEXT["Displaced"]
        )

        special_needs_options = feature_info[
            "Educational special needs"
        ]["values"]

        special_needs = st.selectbox(
            "Educational special needs",
            options=list(special_needs_options.keys()),
            format_func=lambda x: special_needs_options[x],
            help=HELP_TEXT["Educational special needs"]
        )

    st.divider()


    # ----------------------------------
    # ACADEMIC INFORMATION
    # ----------------------------------

    st.markdown(
    '<div class="section-title">📚 Academic Information</div>',
    unsafe_allow_html=True
)

    col1, col2 = st.columns(2)

    with col1:

        application_mode_options = feature_info[
            "Application mode"
        ]["values"]

        application_mode = st.selectbox(
            "Application mode",
            options=list(application_mode_options.keys()),
            format_func=lambda x: application_mode_options[x],
            help=HELP_TEXT["Application mode"]
        )

        application_order_options = feature_info[
            "Application order"
        ]["values"]

        application_order = st.selectbox(
            "Application order",
            options=list(application_order_options.keys()),
            format_func=lambda x: application_order_options[x],
            help=HELP_TEXT["Application order"]
        )

        course_options = feature_info["Course"]["values"]

        course = st.selectbox(
            "Course",
            options=list(course_options.keys()),
            format_func=lambda x: course_options[x],
            help=HELP_TEXT["Course"]
        )

        attendance_options = feature_info[
            "Daytime/evening attendance"
        ]["values"]

        attendance = st.selectbox(
            "Attendance",
            options=list(attendance_options.keys()),
            format_func=lambda x: attendance_options[x],
            help=HELP_TEXT["Daytime/evening attendance"]
        )

    with col2:

        previous_qualification_options = feature_info[
            "Previous qualification"
        ]["values"]

        previous_qualification = st.selectbox(
            "Previous qualification",
            options=list(previous_qualification_options.keys()),
            format_func=lambda x: previous_qualification_options[x],
            help=HELP_TEXT["Previous qualification"]
        )

        mother_qualification_options = feature_info[
            "Mother's qualification"
        ]["values"]

        mother_qualification = st.selectbox(
            "Mother's qualification",
            options=list(mother_qualification_options.keys()),
            format_func=lambda x: mother_qualification_options[x],
            help=HELP_TEXT["Mother's qualification"]
        )

        father_qualification_options = feature_info[
            "Father's qualification"
        ]["values"]

        father_qualification = st.selectbox(
            "Father's qualification",
            options=list(father_qualification_options.keys()),
            format_func=lambda x: father_qualification_options[x],
            help=HELP_TEXT["Father's qualification"]
        )

        mother_occupation_options = feature_info[
            "Mother's occupation"
        ]["values"]

        mother_occupation = st.selectbox(
            "Mother's occupation",
            options=list(mother_occupation_options.keys()),
            format_func=lambda x: mother_occupation_options[x],
            help=HELP_TEXT["Mother's occupation"]
        )

        father_occupation_options = feature_info[
            "Father's occupation"
        ]["values"]

        father_occupation = st.selectbox(
            "Father's occupation",
            options=list(father_occupation_options.keys()),
            format_func=lambda x: father_occupation_options[x],
            help=HELP_TEXT["Father's occupation"]
        )

    st.divider()


    # ----------------------------------
    # FINANCIAL INFORMATION
    # ----------------------------------

    st.markdown(
    '<div class="section-title">💰 Financial Information</div>',
    unsafe_allow_html=True
)

    col1, col2, col3 = st.columns(3)

    with col1:

        debtor_options = feature_info["Debtor"]["values"]

        debtor = st.selectbox(
            "Debtor",
            options=list(debtor_options.keys()),
            format_func=lambda x: debtor_options[x],
            help=HELP_TEXT["Debtor"]
        )

    with col2:

        tuition_options = feature_info[
            "Tuition fees up to date"
        ]["values"]

        tuition = st.selectbox(
            "Tuition fees up to date",
            options=list(tuition_options.keys()),
            format_func=lambda x: tuition_options[x],
            help=HELP_TEXT["Tuition fees up to date"]
        )

    with col3:

        scholarship_options = feature_info[
            "Scholarship holder"
        ]["values"]

        scholarship = st.selectbox(
            "Scholarship holder",
            options=list(scholarship_options.keys()),
            format_func=lambda x: scholarship_options[x],
            help=HELP_TEXT["Scholarship holder"]
        )

    st.divider()


    # ----------------------------------
    # FIRST SEMESTER
    # ----------------------------------

    st.markdown(
    '<div class="section-title">📖 First Semester</div>',
    unsafe_allow_html=True
)

    col1, col2, col3 = st.columns(3)

    with col1:

        first_credited = st.number_input(
            "Credited",
            min_value=0,
            value=0,
            step=1,
            key="first_credited",
            help=HELP_TEXT[
                "Curricular units 1st sem (credited)"
            ]
        )

        first_enrolled = st.number_input(
            "Enrolled",
            min_value=0,
            value=0,
            step=1,
            key="first_enrolled",
            help=HELP_TEXT[
                "Curricular units 1st sem (enrolled)"
            ]
        )

    with col2:

        first_evaluations = st.number_input(
            "Evaluations",
            min_value=0,
            value=0,
            step=1,
            key="first_evaluations",
            help=HELP_TEXT[
                "Curricular units 1st sem (evaluations)"
            ]
        )

        first_approved = st.number_input(
            "Approved",
            min_value=0,
            value=0,
            step=1,
            key="first_approved",
            help=HELP_TEXT[
                "Curricular units 1st sem (approved)"
            ]
        )

    with col3:

        first_grade = st.number_input(
            "Grade",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=0.1,
            key="first_grade",
            help=HELP_TEXT[
                "Curricular units 1st sem (grade)"
            ]
        )

        first_without_evaluations = st.number_input(
            "Without evaluations",
            min_value=0,
            value=0,
            step=1,
            key="first_without_evaluations",
            help=HELP_TEXT[
                "Curricular units 1st sem (without evaluations)"
            ]
        )

    st.divider()


    # ----------------------------------
    # SECOND SEMESTER
    # ----------------------------------

    st.markdown(
    '<div class="section-title">📖 Second Semester</div>',
    unsafe_allow_html=True
)

    col1, col2, col3 = st.columns(3)

    with col1:

        second_credited = st.number_input(
            "Credited",
            min_value=0,
            value=0,
            step=1,
            key="second_credited",
            help=HELP_TEXT[
                "Curricular units 2nd sem (credited)"
            ]
        )

        second_enrolled = st.number_input(
            "Enrolled",
            min_value=0,
            value=0,
            step=1,
            key="second_enrolled",
            help=HELP_TEXT[
                "Curricular units 2nd sem (enrolled)"
            ]
        )

    with col2:

        second_evaluations = st.number_input(
            "Evaluations",
            min_value=0,
            value=0,
            step=1,
            key="second_evaluations",
            help=HELP_TEXT[
                "Curricular units 2nd sem (evaluations)"
            ]
        )

        second_approved = st.number_input(
            "Approved",
            min_value=0,
            value=0,
            step=1,
            key="second_approved",
            help=HELP_TEXT[
                "Curricular units 2nd sem (approved)"
            ]
        )

    with col3:

        second_grade = st.number_input(
            "Grade",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=0.1,
            key="second_grade",
            help=HELP_TEXT[
                "Curricular units 2nd sem (grade)"
            ]
        )

        second_without_evaluations = st.number_input(
            "Without evaluations",
            min_value=0,
            value=0,
            step=1,
            key="second_without_evaluations",
            help=HELP_TEXT[
                "Curricular units 2nd sem (without evaluations)"
            ]
        )

    st.divider()


    # ----------------------------------
    # PREDICTION BUTTON
    # ----------------------------------

    submitted = st.form_submit_button(
        "🔮 Predict",
        use_container_width=True
    )


# ----------------------------------
# PREDICTION
# ----------------------------------

if submitted:

    student = {

        # Student information
        "Marital status": int(marital_status),
        "Application mode": int(application_mode),
        "Application order": int(application_order),
        "Course": int(course),
        "Daytime/evening attendance": int(attendance),
        "Previous qualification": int(previous_qualification),
        "Nacionality": int(nationality),
        "Mother's qualification": int(mother_qualification),
        "Father's qualification": int(father_qualification),
        "Mother's occupation": int(mother_occupation),
        "Father's occupation": int(father_occupation),
        "Displaced": int(displaced),
        "Educational special needs": int(special_needs),
        "Debtor": int(debtor),
        "Tuition fees up to date": int(tuition),
        "Gender": int(gender),
        "Scholarship holder": int(scholarship),
        "Age at enrollment": age,
        "International": int(international),

        # First semester
        "Curricular units 1st sem (credited)": first_credited,
        "Curricular units 1st sem (enrolled)": first_enrolled,
        "Curricular units 1st sem (evaluations)": first_evaluations,
        "Curricular units 1st sem (approved)": first_approved,
        "Curricular units 1st sem (grade)": first_grade,
        "Curricular units 1st sem (without evaluations)":
            first_without_evaluations,

        # Second semester
        "Curricular units 2nd sem (credited)": second_credited,
        "Curricular units 2nd sem (enrolled)": second_enrolled,
        "Curricular units 2nd sem (evaluations)": second_evaluations,
        "Curricular units 2nd sem (approved)": second_approved,
        "Curricular units 2nd sem (grade)": second_grade,
        "Curricular units 2nd sem (without evaluations)":
            second_without_evaluations,

        # Economic variables
        "Unemployment rate": 10.8,
        "Inflation rate": 1.4,
        "GDP": 1.74,
    }

    try:

        result = predict_student(student)

        # ----------------------------------
        # PREDICTION RESULT
        # ----------------------------------

        st.divider()

        st.markdown(
            "## 🎯 Prediction Result"
        )

        prediction_label = result["label"]

        dropout_probability = result["probability_dropout"]
        non_dropout_probability = result["probability_non_dropout"]


        # ----------------------------------
        # MAIN PREDICTION
        # ----------------------------------

        if prediction_label == "Dropout":

            st.error(
                f"⚠️ Prediction: {prediction_label}"
            )

        else:

            st.success(
                f"✅ Prediction: {prediction_label}"
            )


        # ----------------------------------
        # PROBABILITY CARDS
        # ----------------------------------

        col1, col2 = st.columns(2)

        with col1:

            with st.container(border=True):

                st.markdown(
                    "### Dropout probability"
                )

                st.metric(
                    label="Estimated probability",
                    value=f"{dropout_probability * 100:.2f}%"
                )

                st.progress(
                    dropout_probability
                )


        with col2:

            with st.container(border=True):

                st.markdown(
                    "### Graduate / Enrolled probability"
                )

                st.metric(
                    label="Estimated probability",
                    value=f"{non_dropout_probability * 100:.2f}%"
                )

                st.progress(
                    non_dropout_probability
                )


        # ----------------------------------
        # INTERPRETATION
        # ----------------------------------

        st.info(
            "These probabilities are estimates generated by the "
            "machine learning model. They should not be interpreted "
            "as certainty."
        )


    except Exception as error:

        st.error(
            f"Prediction error: {error}"
        )