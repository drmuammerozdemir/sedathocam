import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyreadstat
from scipy import stats

st.set_page_config(page_title="Method Comparison", layout="wide")
st.title("📊 Method Comparison: Bland-Altman, Passing-Bablok, Deming")

# --- Dosya yükleme
uploaded_file = st.file_uploader("Upload CSV, TXT, or SPSS (.sav) file", type=["csv", "txt", "sav"])
if uploaded_file:
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext in ['csv', 'txt']:
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
    elif ext == 'sav':
        with open("temp.sav", "wb") as f:
            f.write(uploaded_file.read())
        df, meta = pyreadstat.read_sav("temp.sav")

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    cols = df.columns.tolist()
    x_col = st.sidebar.selectbox("Select Reference Method", cols)
    y_col = st.sidebar.selectbox("Select Test Method", cols, index=1)

    if x_col != y_col:
        x = df[x_col].dropna().reset_index(drop=True)
        y = df[y_col].dropna().reset_index(drop=True)

        n = min(len(x), len(y))
        x, y = x[:n], y[:n]

        method = st.sidebar.radio("Select Analysis", ["Bland-Altman", "Passing-Bablok", "Deming"])

        if method == "Bland-Altman":
            st.subheader("🔍 Bland-Altman Plot")
            mean = (x + y) / 2
            diff = x - y
            md = np.mean(diff)
            sd = np.std(diff)

            fig, ax = plt.subplots()
            ax.scatter(mean, diff)
            ax.axhline(md, color='gray', linestyle='--', label='Mean Diff')
            ax.axhline(md + 1.96*sd, color='red', linestyle='--', label='+1.96 SD')
            ax.axhline(md - 1.96*sd, color='red', linestyle='--', label='-1.96 SD')
            ax.set_title("Bland-Altman Plot")
            ax.set_xlabel("Mean of Methods")
            ax.set_ylabel("Difference")
            ax.legend()
            st.pyplot(fig)

        elif method == "Passing-Bablok":
            st.subheader("📈 Passing-Bablok Regression")

            # Hesaplamayı yapalım
            slopes = []
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    if dx != 0:
                        slopes.append(dy / dx)
            slopes = np.array(slopes)
            slope = np.median(slopes)

            intercepts = y - slope * x
            intercept = np.median(intercepts)

            st.markdown(f"**Slope:** {slope:.4f}  \n**Intercept:** {intercept:.4f}")

            fig, ax = plt.subplots()
            ax.scatter(x, y, label="Data")
            ax.plot(x, slope * x + intercept, color='blue', label='Passing-Bablok Fit')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title("Passing-Bablok Regression")
            ax.legend()
            st.pyplot(fig)

        elif method == "Deming":
            st.subheader("📉 Deming Regression")

            x_mean, y_mean = np.mean(x), np.mean(y)
            s_xx = np.sum((x - x_mean)**2)
            s_yy = np.sum((y - y_mean)**2)
            s_xy = np.sum((x - x_mean)*(y - y_mean))

            delta = 1  # ratio of variances (assume equal)
            beta = (s_yy - delta * s_xx + np.sqrt((s_yy - delta * s_xx)**2 + 4 * delta * s_xy**2)) / (2 * s_xy)
            alpha = y_mean - beta * x_mean

            st.markdown(f"**Slope:** {beta:.4f}  \n**Intercept:** {alpha:.4f}")

            fig, ax = plt.subplots()
            ax.scatter(x, y, label="Data")
            ax.plot(x, beta * x + alpha, color='green', label='Deming Fit')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title("Deming Regression")
            ax.legend()
            st.pyplot(fig)
    else:
        st.warning("Please select two different columns.")
else:
    st.info("Please upload a file.")