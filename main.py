import hashlib
import hmac
import json
import os
import re
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from features import ZERO_AS_MISSING_COLS, engineer_features

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "diabetes.csv"
DB_PATH = BASE_DIR / "diabetes_app.db"
MODEL_PATH = BASE_DIR / "model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"

PBKDF2_ITERATIONS = 600_000
MIN_PASSWORD_LENGTH = 8
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

st.set_page_config(
    page_title="Diabetes Risk & Patient Portal",
    page_icon="🩺",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Database and security helpers
# ---------------------------------------------------------------------------
def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=15.0)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Return 'iterations$salt$hash' so the iteration count can change later."""
    if salt is None:
        salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return f"{PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        parts = stored_hash.split("$")
        if len(parts) == 3:
            iterations = int(parts[0])
            salt_hex, dk_hex = parts[1], parts[2]
        elif len(parts) == 2:  # old format: 'salt$hash' (100,000 iterations)
            iterations = 100_000
            salt_hex, dk_hex = parts
        else:
            return False
        salt = bytes.fromhex(salt_hex)
        expected_dk = bytes.fromhex(dk_hex)
        actual_dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iterations
        )
        return hmac.compare_digest(actual_dk, expected_dk)
    except Exception:
        return False


def validate_password(password: str) -> str | None:
    """Return an error message, or None if the password is acceptable."""
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters long."
    if not any(ch.isalpha() for ch in password) or not any(ch.isdigit() for ch in password):
        return "Password must contain both letters and numbers."
    return None


def init_db() -> None:
    conn = get_db_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'patient',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                user_label TEXT NOT NULL,
                role TEXT NOT NULL,
                pregnancies REAL,
                glucose REAL,
                blood_pressure REAL,
                skin_thickness REAL,
                insulin REAL,
                bmi REAL,
                dpf REAL,
                age REAL,
                probability REAL NOT NULL,
                risk_tier TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def register_patient(full_name: str, email: str, password: str) -> tuple[bool, str]:
    clean_email = email.strip().lower()
    clean_name = full_name.strip()

    if not clean_name or not clean_email:
        return False, "Enter your name and email."
    if not EMAIL_PATTERN.match(clean_email):
        return False, "Enter a valid email address."
    password_error = validate_password(password)
    if password_error:
        return False, password_error

    conn = get_db_connection()
    try:
        existing = conn.execute(
            "SELECT id FROM users WHERE email = ?", (clean_email,)
        ).fetchone()
        if existing:
            return False, "An account with this email already exists. Please log in."
        conn.execute(
            "INSERT INTO users (full_name, email, password_hash, role) VALUES (?, ?, ?, 'patient')",
            (clean_name, clean_email, hash_password(password)),
        )
        conn.commit()
        return True, "Account created! You can log in now."
    finally:
        conn.close()


def authenticate_patient(email: str, password: str) -> dict | None:
    clean_email = email.strip().lower()
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT id, full_name, email, password_hash, role FROM users WHERE email = ?",
            (clean_email,),
        ).fetchone()
        if row and verify_password(password, row["password_hash"]):
            return {
                "id": row["id"],
                "full_name": row["full_name"],
                "email": row["email"],
                "role": row["role"],
            }
        return None
    finally:
        conn.close()


def save_assessment_record(
    session_user: dict,
    inputs: dict,
    probability: float,
    risk_tier: str,
    prediction: int,
) -> None:
    conn = get_db_connection()
    try:
        user_id = session_user.get("id")  # None for guests
        user_label = (
            f"{session_user.get('full_name')} ({session_user.get('email')})"
            if user_id is not None
            else "Guest User"
        )
        conn.execute(
            """
            INSERT INTO assessments (
                user_id, user_label, role,
                pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, dpf, age,
                probability, risk_tier, prediction
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                user_label,
                session_user.get("role", "guest"),
                float(inputs["Pregnancies"]),
                float(inputs["Glucose"]),
                float(inputs["BloodPressure"]),
                float(inputs["SkinThickness"]),
                float(inputs["Insulin"]),
                float(inputs["BMI"]),
                float(inputs["DiabetesPedigreeFunction"]),
                float(inputs["Age"]),
                float(probability),
                risk_tier,
                int(prediction),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def fetch_user_assessments(session_user: dict, limit: int = 25) -> pd.DataFrame:
    conn = get_db_connection()
    try:
        if session_user.get("id") is not None:
            rows = conn.execute(
                """
                SELECT created_at, glucose, bmi, age, probability, risk_tier, prediction
                FROM assessments
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_user["id"], limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT created_at, user_label, glucose, bmi, age, probability, risk_tier, prediction
                FROM assessments
                WHERE role = 'guest'
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])
    finally:
        conn.close()


def fetch_all_assessments() -> pd.DataFrame:
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, user_id, role, pregnancies, glucose, blood_pressure,
                   skin_thickness, insulin, bmi, dpf, age,
                   probability, risk_tier, prediction, created_at
            FROM assessments
            ORDER BY id
            """
        ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])
    finally:
        conn.close()


def count_registered_users() -> int:
    conn = get_db_connection()
    try:
        return int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@st.cache_data
def load_dataset(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


@st.cache_resource
def load_model():
    """Load the pipeline saved by train.py.

    If the file cannot be loaded (for example a different scikit-learn version
    on another computer), train a fresh model instead of crashing.
    """
    try:
        bundle = joblib.load(MODEL_PATH)
        return bundle["pipeline"], bundle["base_cols"]
    except Exception:
        from train import fit_final_model

        return fit_final_model()


def reference_medians(df: pd.DataFrame) -> dict:
    """Typical value of each input. Zeros in medical columns mean 'missing'."""
    medians = {}
    for col in df.columns:
        if col == "Outcome":
            continue
        series = df[col]
        if col in ZERO_AS_MISSING_COLS:
            series = series[series > 0]
        medians[col] = float(series.median())
    return medians


def explain_prediction(pipeline, row_df: pd.DataFrame, medians: dict) -> pd.Series:
    """What-if analysis: how many percentage points does each input add to the risk?

    For every input we replace it with the typical (median) value and see how
    the predicted probability changes. Positive = this value pushes risk UP
    compared with a typical patient. Negative = pushes risk DOWN.
    """
    base_prob = float(pipeline.predict_proba(engineer_features(row_df))[0][1])
    impacts = {}
    for col in row_df.columns:
        alt = row_df.copy()
        alt[col] = medians[col]
        alt_prob = float(pipeline.predict_proba(engineer_features(alt))[0][1])
        impacts[col] = (base_prob - alt_prob) * 100.0
    return pd.Series(impacts).sort_values()


@st.cache_data
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def classify_risk_tier(prob: float) -> tuple[str, str, str]:
    if prob < 0.35:
        return (
            "Low Chance of Diabetes",
            "success",
            f"Estimated diabetes likelihood is **{prob:.1%}** (below 35%).",
        )
    if prob < 0.65:
        return (
            "Moderate Chance of Diabetes",
            "warning",
            f"Estimated diabetes likelihood is **{prob:.1%}** (35%–64%).",
        )
    return (
        "High Chance of Diabetes",
        "error",
        f"Estimated diabetes likelihood is **{prob:.1%}** (65% or higher).",
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
PRESETS = {
    "Low Risk": {
        "Pregnancies": 1,
        "Glucose": 92.0,
        "BloodPressure": 68.0,
        "SkinThickness": 20.0,
        "Insulin": 70.0,
        "BMI": 22.6,
        "DiabetesPedigreeFunction": 0.24,
        "Age": 25,
    },
    "Moderate Risk": {
        "Pregnancies": 3,
        "Glucose": 134.0,
        "BloodPressure": 76.0,
        "SkinThickness": 28.0,
        "Insulin": 125.0,
        "BMI": 29.8,
        "DiabetesPedigreeFunction": 0.52,
        "Age": 38,
    },
    "High Risk": {
        "Pregnancies": 7,
        "Glucose": 186.0,
        "BloodPressure": 88.0,
        "SkinThickness": 38.0,
        "Insulin": 220.0,
        "BMI": 38.5,
        "DiabetesPedigreeFunction": 1.15,
        "Age": 54,
    },
}


def _apply_preset(preset_name: str, defaults: dict) -> None:
    chosen = PRESETS.get(preset_name, defaults)
    st.session_state["preg_input"] = int(chosen["Pregnancies"])
    st.session_state["gluc_input"] = float(chosen["Glucose"])
    st.session_state["bp_input"] = float(chosen["BloodPressure"])
    st.session_state["skin_input"] = float(chosen["SkinThickness"])
    st.session_state["ins_input"] = float(chosen["Insulin"])
    st.session_state["bmi_input"] = float(chosen["BMI"])
    st.session_state["dpf_input"] = float(chosen["DiabetesPedigreeFunction"])
    st.session_state["age_input"] = int(chosen["Age"])


def render_auth_portal() -> None:
    st.title("🩺 Diabetes Risk Portal — Patient or Guest")
    st.caption("Create an account, log in, or continue as Guest.")

    left, right = st.columns([1.2, 1])

    with left:
        login_tab, signup_tab = st.tabs(["🔑 Patient Login", "📝 Register Patient"])
        with login_tab:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Log In as Patient", type="primary"):
                    user = authenticate_patient(email, password)
                    if user:
                        st.session_state["auth_user"] = user
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")

        with signup_tab:
            with st.form("signup_form"):
                full_name = st.text_input("Full Name")
                new_email = st.text_input("Email")
                new_password = st.text_input(
                    f"Password (min {MIN_PASSWORD_LENGTH} chars, letters + numbers)",
                    type="password",
                )
                if st.form_submit_button("Create Account"):
                    ok, msg = register_patient(full_name, new_email, new_password)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

    with right:
        st.subheader("👤 Guest Access")
        st.write("Run instant diabetes risk predictions without creating an account.")
        if st.button("Continue as Guest", width="stretch"):
            st.session_state["auth_user"] = {
                "id": None,
                "full_name": "Guest",
                "email": None,
                "role": "guest",
            }
            st.rerun()

    st.caption(
        "⚠️ Educational project. Trained on the Pima Indians dataset (adult women, "
        "21+), so results may not apply to everyone. Not medical advice."
    )


def main() -> None:
    init_db()

    if "auth_user" not in st.session_state:
        render_auth_portal()
        return

    if not CSV_PATH.exists():
        st.error(f"Missing `{CSV_PATH.name}`.")
        return
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        st.error("Model not found. Run `python train.py` first, then restart the app.")
        return

    current_user = st.session_state["auth_user"]

    df = load_dataset(CSV_PATH)
    pipeline, base_cols = load_model()
    metrics = load_metrics()

    medians = reference_medians(df)
    defaults = {
        "Pregnancies": int(medians["Pregnancies"]),
        "Glucose": round(medians["Glucose"], 1),
        "BloodPressure": round(medians["BloodPressure"], 1),
        "SkinThickness": round(medians["SkinThickness"], 1),
        "Insulin": round(medians["Insulin"], 1),
        "BMI": round(medians["BMI"], 1),
        "DiabetesPedigreeFunction": round(medians["DiabetesPedigreeFunction"], 3),
        "Age": int(medians["Age"]),
    }

    if "preg_input" not in st.session_state:
        _apply_preset("Medians", defaults)

    h1, h2 = st.columns([3, 1])
    with h1:
        st.title("🩺 Diabetes Risk Assessment")
        st.caption(
            f"User: **{current_user['full_name']}** ({current_user['role']}) • "
            "Educational screening aid only, not medical advice."
        )
    with h2:
        if st.button("Log Out / Switch", width="stretch"):
            st.session_state.pop("auth_user", None)
            st.session_state.pop("last_prediction", None)
            st.rerun()

    st.sidebar.header("Presets")
    p1, p2 = st.sidebar.columns(2)
    with p1:
        if st.button("Low Risk", width="stretch"):
            _apply_preset("Low Risk", defaults)
            st.rerun()
    with p2:
        if st.button("Moderate Risk", width="stretch"):
            _apply_preset("Moderate Risk", defaults)
            st.rerun()
    p3, p4 = st.sidebar.columns(2)
    with p3:
        if st.button("High Risk", width="stretch"):
            _apply_preset("High Risk", defaults)
            st.rerun()
    with p4:
        if st.button("Medians", width="stretch"):
            _apply_preset("Medians", defaults)
            st.rerun()

    st.sidebar.header("Patient Details")
    pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, key="preg_input", step=1)
    glucose = st.sidebar.number_input("Glucose (mg/dL)", 0.0, 300.0, key="gluc_input", step=1.0)
    blood_pressure = st.sidebar.number_input("Blood Pressure (mm Hg)", 0.0, 200.0, key="bp_input", step=1.0)
    skin_thickness = st.sidebar.number_input("Skin Thickness (mm)", 0.0, 120.0, key="skin_input", step=1.0)
    insulin = st.sidebar.number_input("Insulin (mu U/ml)", 0.0, 900.0, key="ins_input", step=1.0)
    bmi = st.sidebar.number_input("BMI", 0.0, 80.0, key="bmi_input", step=0.1)
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, key="dpf_input", step=0.01)
    age = st.sidebar.number_input("Age (years)", 1, 120, key="age_input", step=1)

    input_map = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    is_admin = current_user.get("role") == "admin"
    tab_names = ["🔮 Prediction", "🗂️ History", "📈 Model"]
    if is_admin:
        tab_names.append("📊 Admin Dashboard")
    tabs = st.tabs(tab_names)
    tab_pred, tab_hist, tab_stats = tabs[0], tabs[1], tabs[2]

    with tab_pred:
        missing_inputs = [
            name for name in ZERO_AS_MISSING_COLS if float(input_map[name]) == 0
        ]
        if missing_inputs:
            st.warning(
                "These values are 0, which is not realistic, so the model treats "
                f"them as **unknown**: {', '.join(missing_inputs)}. "
                "Enter real measurements for a more reliable result."
            )

        if st.button("Calculate Chances of Diabetes", type="primary"):
            row_df = pd.DataFrame([{c: float(input_map[c]) for c in base_cols}], columns=base_cols)
            prob = float(pipeline.predict_proba(engineer_features(row_df))[0][1])
            pred = int(prob >= 0.50)
            tier_label, badge_type, summary_text = classify_risk_tier(prob)
            save_assessment_record(current_user, input_map, prob, tier_label, pred)
            st.session_state["last_prediction"] = {
                "prob": prob,
                "pred": pred,
                "tier_label": tier_label,
                "badge_type": badge_type,
                "summary_text": summary_text,
                "row_df": row_df,
                "impacts": explain_prediction(pipeline, row_df, medians),
            }

        if "last_prediction" in st.session_state:
            res = st.session_state["last_prediction"]
            if res["badge_type"] == "success":
                st.success(f"### {res['tier_label']}")
            elif res["badge_type"] == "warning":
                st.warning(f"### {res['tier_label']}")
            else:
                st.error(f"### {res['tier_label']}")
            st.markdown(res["summary_text"])
            st.progress(min(max(res["prob"], 0.0), 1.0), text=f"Chance of Diabetes: {res['prob']:.1%}")
            st.dataframe(res["row_df"], width="stretch")

            st.subheader("Why this result?")
            impacts = res["impacts"]
            raising = impacts[impacts > 0.5].sort_values(ascending=False)
            lowering = impacts[impacts < -0.5]
            if not raising.empty:
                st.markdown(
                    "**Raising the risk:** "
                    + ", ".join(f"{k} (+{v:.1f} pts)" for k, v in raising.items())
                )
            if not lowering.empty:
                st.markdown(
                    "**Lowering the risk:** "
                    + ", ".join(f"{k} ({v:.1f} pts)" for k, v in lowering.items())
                )
            if raising.empty and lowering.empty:
                st.markdown("Your values are close to a typical patient, so no single input stands out.")
            st.bar_chart(impacts.rename("Change in risk (percentage points)"), horizontal=True)
            st.caption(
                "Method: each input is replaced with the typical value from the dataset "
                "and we measure how the risk changes. A positive bar means your value "
                "pushes the risk up compared with a typical patient."
            )

    with tab_hist:
        hist_df = fetch_user_assessments(current_user)
        if hist_df.empty:
            st.info("No saved records yet.")
        else:
            st.dataframe(hist_df, width="stretch")

    with tab_stats:
        st.subheader("Model performance (5-fold cross-validation)")
        ens = metrics["ensemble_cv"]
        base = metrics["baseline_logreg_cv"]
        rows = []
        for key, label in [
            ("accuracy", "Accuracy"),
            ("recall", "Recall (finds diabetics)"),
            ("precision", "Precision"),
            ("f1", "F1 score"),
            ("roc_auc", "ROC-AUC"),
        ]:
            rows.append(
                {
                    "Metric": label,
                    "Ensemble (LightGBM+RF+KNN)": f"{ens[key]['mean']:.3f} ± {ens[key]['std']:.3f}",
                    "Logistic Regression baseline": f"{base[key]['mean']:.3f} ± {base[key]['std']:.3f}",
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            f"Trained on {metrics['n_rows']} patients. Cross-validation is more reliable "
            "than a single train/test split on a small dataset."
        )
        st.subheader("Dataset preview")
        st.dataframe(df.head(10), width="stretch")

    if is_admin:
        with tabs[3]:
            render_admin_dashboard()


def render_admin_dashboard() -> None:
    st.subheader("Admin dashboard (aggregate data only)")
    data = fetch_all_assessments()
    if data.empty:
        st.info("No assessments saved yet. Make a few predictions first.")
        return

    data["created_at"] = pd.to_datetime(data["created_at"])
    data["probability_pct"] = data["probability"] * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total assessments", len(data))
    c2.metric("Registered patients", count_registered_users())
    c3.metric("Guest assessments", int((data["role"] == "guest").sum()))
    c4.metric("Average risk", f"{data['probability_pct'].mean():.1f}%")

    left, right = st.columns(2)
    with left:
        st.markdown("**Risk tier distribution**")
        st.bar_chart(data["risk_tier"].value_counts())
    with right:
        st.markdown("**Average risk by age group (%)**")
        age_groups = pd.cut(
            data["age"], bins=[0, 29, 39, 49, 59, 120],
            labels=["<30", "30-39", "40-49", "50-59", "60+"],
        )
        st.bar_chart(data.groupby(age_groups, observed=True)["probability_pct"].mean())

    st.markdown("**Glucose vs predicted risk**")
    st.scatter_chart(data, x="glucose", y="probability_pct")

    st.markdown("**Assessments per day**")
    per_day = data.groupby(data["created_at"].dt.date).size().rename("assessments")
    st.line_chart(per_day)

    st.markdown("**Average inputs by risk tier**")
    st.dataframe(
        data.groupby("risk_tier")[["glucose", "bmi", "age", "blood_pressure"]].mean().round(1),
        width="stretch",
    )

    # Privacy: export WITHOUT names, emails or user ids
    anonymous = data.drop(columns=["user_id", "probability_pct"])
    st.download_button(
        "Download anonymous data (CSV)",
        anonymous.to_csv(index=False).encode("utf-8"),
        file_name="assessments_anonymous.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()