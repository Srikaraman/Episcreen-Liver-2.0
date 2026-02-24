# =============================================================
# HCC UNIFIED PIPELINE — STREAMLIT APP
# Run with: streamlit run hcc_app.py
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os
import io

from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# =========================================================
# CONFIG (must match training script)
# =========================================================
MODEL_DIR = "hcc_unified_models"
FEATURES = ['DQ', 'DQN', 'AFP', 'Age', 'Sex_encoded', 'cfDNA']

AFP_BINS   = [-np.inf, 20, np.inf]
AFP_LABELS = ['afp_0_20', 'afp_gt_20']

DQ_BINS   = [-np.inf, 10, 20, 30, 40, 50, np.inf]
DQ_LABELS = ['dq_0_10', 'dq_10_20', 'dq_20_30', 'dq_30_40', 'dq_40_50', 'dq_gt_50']

THRESHOLD_LE          = 0.10
THRESHOLD_GE          = 0.90
DQ_THRESHOLD_SPLIT    = 15
ELS_BOUNDARIES        = [0.10, 0.35, 0.65, 0.90]
LOW_RULE_OUT          = 0.35
FNW                   = 0.3
FPW                   = 0.7
DQ_RELIABLE_THRESHOLD = 14
MODEL_DIR             = "hcc_unified_models"

# =========================================================
# REQUIRED: FeatureSelector must be defined for joblib
# =========================================================

class FeatureSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.cols = X.columns.tolist() if hasattr(X, 'columns') else list(range(X.shape[1]))
        return self
    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.cols)
        return X[FEATURES].values

# =========================================================
# HELPERS
# =========================================================

def assign_afp_segment(afp):
    for i in range(len(AFP_BINS) - 1):
        if AFP_BINS[i] < afp <= AFP_BINS[i + 1]:
            return AFP_LABELS[i]
    return AFP_LABELS[-1]

def assign_dq_segment(dq):
    for i in range(len(DQ_BINS) - 1):
        if DQ_BINS[i] < dq <= DQ_BINS[i + 1]:
            return DQ_LABELS[i]
    return DQ_LABELS[-1]

def threshold_by_dq(dq):
    return THRESHOLD_LE if dq <= DQ_THRESHOLD_SPLIT else THRESHOLD_GE

def assign_els(p):
    if p < ELS_BOUNDARIES[0]: return "ELS-1"
    elif p < ELS_BOUNDARIES[1]: return "ELS-2"
    elif p < ELS_BOUNDARIES[2]: return "ELS-3"
    elif p < ELS_BOUNDARIES[3]: return "ELS-4"
    else:                        return "ELS-5"

def els_zone(els):
    if els in ('ELS-1', 'ELS-2'): return "LOW"
    if els in ('ELS-4', 'ELS-5'): return "HIGH"
    return "MID"

def format_prob(p):
    if pd.isna(p):  return ""
    if p <= 0.02:   return "<2%"
    if p >= 0.98:   return ">98%"
    return f"{int(round(p * 100))}%"

def els_color(els):
    return {
        "ELS-1": "#2ecc71",
        "ELS-2": "#a8e6a3",
        "ELS-3": "#f39c12",
        "ELS-4": "#e67e22",
        "ELS-5": "#e74c3c"
    }.get(els, "#cccccc")

def els_meaning(els):
    return {
        "ELS-1": "Very low risk — HCC highly unlikely",
        "ELS-2": "Low risk — routine surveillance recommended",
        "ELS-3": "Intermediate risk — closer follow-up advised",
        "ELS-4": "High risk — further workup strongly recommended",
        "ELS-5": "Very high risk — HCC highly likely"
    }.get(els, "")

# =========================================================
# LOAD MODELS (cached so they load only once)
# =========================================================

@st.cache_resource
def load_models():
    required = ['global_fallback.pkl', 'afp_fn_model.pkl', 'afp_fp_model.pkl',
                'dq_fn_model.pkl', 'dq_fp_model.pkl']
    missing = [f for f in required if not os.path.exists(f"{MODEL_DIR}/{f}")]
    if missing:
        return None, f"Missing model files: {missing}. Run hcc_unified_pipeline.py first."

    models = {
        'afp':     {},
        'dq':      {},
        'global':  joblib.load(f"{MODEL_DIR}/global_fallback.pkl"),
        'afp_fn':  joblib.load(f"{MODEL_DIR}/afp_fn_model.pkl"),
        'afp_fp':  joblib.load(f"{MODEL_DIR}/afp_fp_model.pkl"),
        'dq_fn':   joblib.load(f"{MODEL_DIR}/dq_fn_model.pkl"),
        'dq_fp':   joblib.load(f"{MODEL_DIR}/dq_fp_model.pkl"),
        'arb':     None
    }
    for seg in AFP_LABELS:
        path = f"{MODEL_DIR}/afp_model_{seg}.pkl"
        if os.path.exists(path):
            models['afp'][seg] = joblib.load(path)
    for seg in DQ_LABELS:
        path = f"{MODEL_DIR}/dq_model_{seg}.pkl"
        if os.path.exists(path):
            models['dq'][seg] = joblib.load(path)
    arb_path = f"{MODEL_DIR}/arbitration_model.pkl"
    if os.path.exists(arb_path):
        models['arb'] = joblib.load(arb_path)

    return models, None

# =========================================================
# PREDICTOR
# =========================================================

def predict_one_stage(row_features, segment_fn, seg_models, global_model,
                      fn_model, fp_model, seg_key):
    seg   = segment_fn(row_features[seg_key])
    model = seg_models.get(seg, global_model)
    vals  = np.array([row_features[f] for f in FEATURES]).reshape(1, -1)
    orig_p    = model.predict_proba(vals)[0][1]
    threshold = threshold_by_dq(row_features['DQ'])
    orig_pred = int(orig_p >= threshold)
    final_p   = orig_p

    if orig_pred == 0 and orig_p > 0.10:
        fn_p    = fn_model.predict_proba(vals)[0][1]
        final_p = FNW * orig_p + (1 - FNW) * fn_p
    if orig_pred == 1:
        fp_p    = fp_model.predict_proba(vals)[0][1]
        final_p = FPW * orig_p + (1 - FPW) * (1 - fp_p)

    return orig_p, orig_pred, final_p


def predict_row(row_features, models):
    features_clean = {f: row_features[f] for f in FEATURES}

    afp_orig, afp_pred, afp_final = predict_one_stage(
        features_clean, assign_afp_segment,
        models['afp'], models['global'],
        models['afp_fn'], models['afp_fp'], 'AFP'
    )
    dq_orig, dq_pred, dq_final = predict_one_stage(
        features_clean, assign_dq_segment,
        models['dq'], models['global'],
        models['dq_fn'], models['dq_fp'], 'DQ'
    )

    els_afp  = assign_els(afp_final)
    els_dq   = assign_els(dq_final)
    is_hard  = (afp_pred != dq_pred) or (els_zone(els_afp) != els_zone(els_dq))
    fused_p  = (afp_final + dq_final) / 2

    if is_hard and models['arb'] is not None:
        arb_features = {**features_clean}
        if arb_features['DQ'] < DQ_RELIABLE_THRESHOLD:
            arb_features['DQ']  = np.nan
            arb_features['DQN'] = np.nan
        arb_vals = np.array([arb_features[f] for f in FEATURES]).reshape(1, -1)
        fused_p  = models['arb'].predict_proba(arb_vals)[0][1]

    els_final  = assign_els(fused_p)
    prediction = int(els_final in ('ELS-3', 'ELS-4', 'ELS-5'))

    return {
        'Prob_AFP':        round(afp_final, 4),
        'ELS_AFP':         els_afp,
        'Prob_DQ':         round(dq_final, 4),
        'ELS_DQ':          els_dq,
        'Is_Hard_Case':    is_hard,
        'Fused_Probability':        round(fused_p, 4),
        'Fused_Probability_Report': format_prob(fused_p),
        'ELS_Final':       els_final,
        'ELS_Meaning':     els_meaning(els_final),
        'Prediction':      prediction,
        'Prediction_Label':'HCC' if prediction else 'Non-HCC'
    }

# =========================================================
# STREAMLIT APP
# =========================================================

st.set_page_config(
    page_title="EpiScreen HCC Predictor",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 EpiScreen HCC Prediction System")
st.caption("Hepatocellular Carcinoma risk stratification using ELS (Epigenetic Liver Score)")

# Load models
models, error = load_models()
if error:
    st.error(f"⚠️ {error}")
    st.stop()
else:
    st.success(f"✅ Models loaded from `{MODEL_DIR}/`")

# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🧑‍⚕️ Single Patient", "📋 Batch Prediction", "📊 Model Evaluation"])

# =========================================================
# TAB 1 — SINGLE PATIENT
# =========================================================
with tab1:

    st.subheader("Single Patient Prediction")
    st.markdown("Enter patient values below to get an instant ELS result.")

    col1, col2, col3 = st.columns(3)

    with col1:
        dq    = st.number_input("DQ",    min_value=0.0,  max_value=500000.0, value=22.0, step=0.1)
        dqn   = st.number_input("DQN",   min_value=0.0,  max_value=200000.0, value=180.0, step=1.0)

    with col2:
        afp   = st.number_input("AFP",   min_value=0.0,  max_value=10000000.0, value=15.0, step=0.1)
        age   = st.number_input("Age",   min_value=18.0, max_value=100.0,   value=58.0, step=1.0)

    with col3:
        sex   = st.selectbox("Sex", ["Male", "Female"])
        cfdna = st.number_input("cfDNA", min_value=0.0,  max_value=10000.0, value=0.05, step=1.0, format="%.3f")

    if st.button("🔍 Predict", type="primary"):

        patient = {
            'DQ':          dq,
            'DQN':         dqn,
            'AFP':         afp,
            'Age':         age,
            'Sex_encoded': 1 if sex == 'Male' else 0,
            'cfDNA':       cfdna
        }

        result = predict_row(patient, models)

        st.markdown("---")

        # ELS result card
        els     = result['ELS_Final']
        color   = els_color(els)
        prob_pct = result['Fused_Probability_Report']

        st.markdown(f"""
        <div style="
            background-color: {color};
            padding: 20px 30px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 16px;
        ">
            <h1 style="color: white; margin: 0; font-size: 3em;">{els}</h1>
            <h3 style="color: white; margin: 4px 0;">{prob_pct} probability of HCC</h3>
            <p style="color: white; margin: 0; font-size: 1.1em;">{result['ELS_Meaning']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Detail columns
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("AFP Model", f"{result['Prob_AFP']:.3f}", result['ELS_AFP'])
        with c2:
            st.metric("DQ Model",  f"{result['Prob_DQ']:.3f}",  result['ELS_DQ'])
        with c3:
            st.metric("Fused Probability", f"{result['Fused_Probability']:.3f}", result['ELS_Final'])

        if result['Is_Hard_Case']:
            st.warning("⚠️ **Hard case detected** — models disagreed. Arbitration model was used.")
        else:
            st.info("✅ Easy case — both models agreed. Simple average used.")

        # ELS gauge bar
        st.markdown("#### ELS Probability Scale")
        fig, ax = plt.subplots(figsize=(8, 1.2))
        ax.barh([0], [0.10],               color='#2ecc71',  height=0.5)
        ax.barh([0], [0.25], left=[0.10],  color='#a8e6a3',  height=0.5)
        ax.barh([0], [0.30], left=[0.35],  color='#f39c12',  height=0.5)
        ax.barh([0], [0.25], left=[0.65],  color='#e67e22',  height=0.5)
        ax.barh([0], [0.10], left=[0.90],  color='#e74c3c',  height=0.5)
        ax.axvline(result['Fused_Probability'], color='black', linewidth=3, label='Patient')
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 0.10, 0.35, 0.65, 0.90, 1.0])
        ax.set_xticklabels(['0', 'ELS-1|2', 'ELS-2|3', 'ELS-3|4', 'ELS-4|5', '1'])
        ax.set_title("▲ Patient position on ELS scale", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# =========================================================
# TAB 2 — BATCH PREDICTION
# =========================================================
with tab2:

    st.subheader("Batch Prediction")
    st.markdown("""
    Upload a CSV file with the following columns:
    `DQ`, `DQN`, `AFP`, `Age`, `Sex` (Male/Female), `cfDNA`

    Optional: `TGKID` or any ID column for reference.
    """)

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)

        # Validate columns
        required_cols = ['DQ', 'DQN', 'AFP', 'Age', 'Sex', 'cfDNA']
        missing_cols  = [c for c in required_cols if c not in df.columns]

        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}")
        else:
            df['Sex_encoded'] = df['Sex'].map({'Male': 1, 'Female': 0})
            df[FEATURES]      = SimpleImputer(strategy='mean').fit_transform(df[FEATURES])

            st.info(f"Processing {len(df)} patients …")
            progress = st.progress(0)

            results = []
            for i, (_, row) in enumerate(df.iterrows()):
                results.append(predict_row(row, models))
                progress.progress((i + 1) / len(df))

            results_df = pd.DataFrame(results)
            out        = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

            st.success(f"✅ Done — {len(out)} predictions complete")

            # ELS distribution chart
            st.markdown("#### ELS Distribution")
            els_counts = out['ELS_Final'].value_counts().sort_index()
            colors_map = [els_color(e) for e in els_counts.index]

            fig, ax = plt.subplots(figsize=(7, 3.5))
            bars = ax.bar(els_counts.index, els_counts.values, color=colors_map, edgecolor='white')
            for bar, val in zip(bars, els_counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
            ax.set_ylabel("Number of patients")
            ax.set_title("ELS Distribution")
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Hard vs easy
            n_hard = out['Is_Hard_Case'].sum()
            n_easy = len(out) - n_hard
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Patients", len(out))
            c2.metric("Hard Cases (Arbitration)", n_hard)
            c3.metric("Easy Cases (Average)", n_easy)

            # Results table
            st.markdown("#### Results Table")
            display_cols = [c for c in [
                'TGKID', 'DQ', 'AFP', 'Age', 'Sex',
                'Prob_AFP', 'ELS_AFP',
                'Prob_DQ',  'ELS_DQ',
                'Fused_Probability', 'Fused_Probability_Report',
                'ELS_Final', 'Prediction_Label', 'Is_Hard_Case'
            ] if c in out.columns]

            st.dataframe(
                out[display_cols].style.applymap(
                    lambda v: f"background-color: {els_color(v)}; color: white"
                    if v in ('ELS-1','ELS-2','ELS-3','ELS-4','ELS-5') else "",
                    subset=['ELS_Final']
                ),
                use_container_width=True
            )

            # Download button
            csv_bytes = out.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Predictions CSV",
                data=csv_bytes,
                file_name="hcc_predictions.csv",
                mime="text/csv"
            )

# =========================================================
# TAB 3 — MODEL EVALUATION (labelled data)
# =========================================================
with tab3:

    st.subheader("Model Evaluation")
    st.markdown("""
    Upload a **labelled** CSV (must include a `Group` column: `HCC` or `Non-HCC`).
    """)

    eval_file = st.file_uploader("Upload labelled CSV", type=["csv"], key="eval")

    if eval_file is not None:
        df = pd.read_csv(eval_file)

        if 'Group' not in df.columns and 'Group_encoded' not in df.columns:
            st.error("❌ No 'Group' column found. Please include Group (HCC / Non-HCC).")
        else:
            if 'Group_encoded' not in df.columns:
                df['Group_encoded'] = df['Group'].map({'HCC': 1, 'Non-HCC': 0})
            df['Sex_encoded'] = df['Sex'].map({'Male': 1, 'Female': 0})
            df[FEATURES]      = SimpleImputer(strategy='mean').fit_transform(df[FEATURES])

            st.info(f"Evaluating {len(df)} patients …")
            progress2 = st.progress(0)

            results = []
            for i, (_, row) in enumerate(df.iterrows()):
                results.append(predict_row(row, models))
                progress2.progress((i + 1) / len(df))

            results_df = pd.DataFrame(results)
            out = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

            y_true = out['Group_encoded'].values
            y_prob = out['Fused_Probability'].values
            y_pred = out['Prediction'].values

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv  = tn / (tn + fn) if (tn + fn) > 0 else 0
            auc  = roc_auc_score(y_true, y_prob)

            st.success("✅ Evaluation complete")

            # Metrics row
            st.markdown("#### Performance Metrics")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Sensitivity", f"{sens:.3f}")
            m2.metric("Specificity", f"{spec:.3f}")
            m3.metric("PPV",         f"{ppv:.3f}")
            m4.metric("NPV",         f"{npv:.3f}")
            m5.metric("AUC",         f"{auc:.3f}")

            # Confusion matrix + ROC side by side
            col_cm, col_roc = st.columns(2)

            with col_cm:
                st.markdown("#### Confusion Matrix")
                fig, ax = plt.subplots(figsize=(4, 3.5))
                cm_data = np.array([[tn, fp], [fn, tp]])
                im = ax.imshow(cm_data, cmap='Blues')
                ax.set_xticks([0, 1]); ax.set_xticklabels(['Non-HCC', 'HCC'])
                ax.set_yticks([0, 1]); ax.set_yticklabels(['Non-HCC', 'HCC'])
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, cm_data[i, j], ha='center', va='center',
                                fontsize=16, fontweight='bold',
                                color='white' if cm_data[i, j] > cm_data.max() / 2 else 'black')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col_roc:
                st.markdown("#### ROC Curve")
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                fig, ax = plt.subplots(figsize=(4, 3.5))
                ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.4f}')
                ax.plot([0, 1], [0, 1], 'k--', lw=1)
                ax.scatter([1 - spec], [sens], color='red', zorder=5, s=80,
                           label=f'Sens={sens:.3f}\nSpec={spec:.3f}')
                ax.set_xlabel("1 − Specificity"); ax.set_ylabel("Sensitivity")
                ax.legend(loc='lower right', fontsize=8)
                ax.set_title("ROC Curve")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # ELS × Ground truth
            st.markdown("#### ELS × Ground Truth")
            ct = pd.crosstab(out['ELS_Final'], out['Group_encoded'],
                             colnames=['0 = Non-HCC  |  1 = HCC'])
            st.dataframe(ct, use_container_width=True)

            # Download
            csv_bytes = out.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Evaluation CSV",
                data=csv_bytes,
                file_name="hcc_evaluation.csv",
                mime="text/csv"
            )
