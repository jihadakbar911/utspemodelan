import streamlit as st
import pandas as pd
import numpy as np

# =========================================
# FUNGSI SIMULASI SISTEM DINAMIS HARGA EMAS
# =========================================
def simulate_gold_price(
    initial_price: float,
    n_days: int,
    demand_shock: float = 0.0,
    alpha: float = 0.2,
    beta: float = 0.3,
    gamma: float = 0.2,
    k_demand: float = 0.5,
    sigma_noise: float = 0.01,
    random_seed: int = 42
):
    np.random.seed(random_seed)

    price_t = initial_price
    trend_t = 0.0
    demand_index_t = 1.0
    base_price = initial_price

    prices, trends, demands, timesteps = [], [], [], []

    for t in range(n_days):
        fundamental_price_t = base_price * (1 + k_demand * (demand_index_t - 1))
        noise_t = np.random.normal(0, sigma_noise) * price_t

        delta_price_t = (
            alpha * (fundamental_price_t - price_t)
            + beta * trend_t
            + noise_t
        )

        price_t += delta_price_t
        trend_t = (1 - gamma) * trend_t + gamma * delta_price_t
        demand_index_t += demand_shock

        timesteps.append(t)
        prices.append(price_t)
        trends.append(trend_t)
        demands.append(demand_index_t)

    return pd.DataFrame({
        "t": timesteps,
        "sim_price": prices,
        "sim_trend": trends,
        "sim_demand_index": demands,
    })

def analyze_simulation(sim_df):
    price = sim_df["sim_price"]
    growth_rate = (price.iloc[-1] - price.iloc[0]) / price.iloc[0] * 100
    volatility = price.pct_change().std() * 100
    max_price = price.max()
    min_price = price.min()
    return growth_rate, volatility, max_price, min_price

# ========================
# LAYOUT STREAMLIT APP
# ========================
st.set_page_config(page_title="Simulasi Sistem Dinamis Harga Emas", layout="wide")

st.title("Simulasi Sistem Dinamis Harga Emas")
st.write("""
Aplikasi ini melakukan analisis dan simulasi harga emas menggunakan pendekatan **system dynamics** 
dan **skenario permintaan**.
Silakan upload dataset harga emas dan atur parameter simulasi di sidebar.
""")

# ------------------------
# SIDEBAR: INPUT PENGGUNA
# ------------------------
st.sidebar.header("Pengaturan Simulasi")

uploaded_file = st.sidebar.file_uploader("Upload dataset harga emas (CSV)", type=["csv"])

n_days_sim = st.sidebar.slider("Horizon simulasi (hari)", min_value=30, max_value=365, value=180, step=10)

scenario = st.sidebar.selectbox(
    "Pilih skenario permintaan",
    ["Baseline (permintaan stabil)", "Permintaan naik", "Permintaan turun", "Custom (manual)"]
)

if scenario == "Baseline (permintaan stabil)":
    demand_shock = 0.0
elif scenario == "Permintaan naik":
    demand_shock = 0.002
elif scenario == "Permintaan turun":
    demand_shock = -0.002
else:
    demand_shock = st.sidebar.slider("Demand shock per hari", -0.01, 0.01, 0.0, 0.001)

st.sidebar.subheader("Parameter Model")
alpha = st.sidebar.slider("alpha (penyesuaian ke harga fundamental)", 0.0, 1.0, 0.2, 0.05)
beta = st.sidebar.slider("beta (pengaruh tren)", 0.0, 1.0, 0.3, 0.05)
gamma = st.sidebar.slider("gamma (kecepatan update tren)", 0.0, 1.0, 0.2, 0.05)
k_demand = st.sidebar.slider("k_demand (sensitivitas terhadap permintaan)", 0.0, 2.0, 0.5, 0.05)
sigma_noise = st.sidebar.slider("sigma_noise (ketidakpastian/noise)", 0.0, 0.1, 0.01, 0.005)

random_seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

# =========================
# BAGIAN UTAMA: ANALISIS
# =========================
if uploaded_file is None:
    st.info("Silakan upload file CSV yang berisi data harga emas. Minimal ada kolom 'Date' dan 'Price'.")
else:
    # ----------------------
    # BACA & PREPROSES DATA
    # ----------------------
    df = pd.read_csv(uploaded_file)

    # Cek kolom
    if not {"Date", "Price"}.issubset(df.columns):
        st.error("Dataset harus memiliki kolom 'Date' dan 'Price'.")
    else:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        df["Price"] = df["Price"].astype(float)

        st.subheader("Data Historis Harga Emas")
        st.dataframe(df.head())

        st.line_chart(df.set_index("Date")["Price"])

        last_price = df["Price"].iloc[-1]

        st.write(f"**Harga emas terakhir dalam data historis: {last_price:.2f}**")

        # ----------------------
        # JALANKAN SIMULASI
        # ----------------------
        sim_result = simulate_gold_price(
            initial_price=last_price,
            n_days=n_days_sim,
            demand_shock=demand_shock,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            k_demand=k_demand,
            sigma_noise=sigma_noise,
            random_seed=random_seed
        )

        st.subheader("Hasil Simulasi Harga Emas (Skenario Terpilih)")
        st.line_chart(sim_result.set_index("t")["sim_price"])

        growth_rate, volatility, max_price, min_price = analyze_simulation(sim_result)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Harga awal simulasi", f"{sim_result['sim_price'].iloc[0]:.2f}")
        with col2:
            st.metric("Harga akhir simulasi", f"{sim_result['sim_price'].iloc[-1]:.2f}")
        with col3:
            st.metric("Pertumbuhan (%)", f"{growth_rate:.2f}")
        with col4:
            st.metric("Volatilitas (%)", f"{volatility:.2f}")

        st.markdown("**Rentang harga simulasi:**")
        st.write(f"Min: `{min_price:.2f}`, Max: `{max_price:.2f}`")

        # ----------------------
        # INTERPRETASI SINGKAT
        # ----------------------
        st.subheader("Interpretasi Singkat")

        if growth_rate > 0:
            st.write(f"- Harga emas cenderung **naik** selama periode simulasi (sekitar {growth_rate:.2f}%).")
        elif growth_rate < 0:
            st.write(f"- Harga emas cenderung **turun** selama periode simulasi (sekitar {abs(growth_rate):.2f}%).")
        else:
            st.write("- Harga emas relatif **stagnan** selama periode simulasi.")

        if volatility > 3:
            st.write(f"- Volatilitas cukup **tinggi** (~{volatility:.2f}%), mencerminkan ketidakpastian pasar yang besar.")
        elif volatility > 1:
            st.write(f"- Volatilitas berada di tingkat **sedang** (~{volatility:.2f}%).")
        else:
            st.write(f"- Volatilitas relatif **rendah** (~{volatility:.2f}%), pasar cenderung stabil.")

        st.write(
            "- Parameter `demand_shock` merepresentasikan perubahan permintaan harian akibat faktor eksternal "
            "(misalnya krisis, perubahan persepsi safe haven, dan lain-lain)."
        )
