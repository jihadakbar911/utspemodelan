import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Bagian 1: Fungsi Model (Logika Inti dari Colab) ---
# Kita pakai @st.cache_data agar data hanya di-load sekali

@st.cache_data
def muat_data_dan_parameter():
    """
    Fungsi ini memuat data, mengolahnya, dan menghitung parameter historis.
    Fungsi ini hanya akan dijalankan sekali berkat @st.cache_data.
    """
    file_path = 'gold-price.csv'
    df = pd.read_csv(file_path)
    
    # Konversi dan urutkan data
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)
    
    # Hitung 'Aliran' (Flow)
    df['Perubahan_Harga'] = df['Price'].diff()

    # Analisis 'Aliran' (Parameter Historis)
    rata_rata_perubahan = df['Perubahan_Harga'].dropna().mean()
    std_dev_perubahan = df['Perubahan_Harga'].dropna().std()
    
    # Ambil harga terakhir sebagai titik awal simulasi
    harga_awal_simulasi = df['Price'].iloc[-1]
    
    return harga_awal_simulasi, rata_rata_perubahan, std_dev_perubahan

def jalankan_simulasi_sistem_dinamis(start_price, mean_change, std_change, num_days):
    """
    Fungsi simulasi yang sama persis seperti di Colab.
    """
    stok_harga = [start_price]
    harga_sekarang = start_price
    
    for hari in range(num_days - 1):
        aliran_perubahan = np.random.normal(loc=mean_change, scale=std_change)
        harga_sekarang = harga_sekarang + aliran_perubahan
        stok_harga.append(max(0, harga_sekarang)) # Pastikan harga tidak negatif
        
    return stok_harga

# --- Bagian 2: Tampilan Aplikasi Web (Streamlit) ---

st.title('Aplikasi Web Pemodelan & Simulasi')
st.header('Studi Kasus: Simulasi Sistem Dinamis Harga Emas')
st.write("""
Aplikasi ini adalah implementasi dari tugas UTS Pemodelan dan Simulasi.
Model ini menjalankan "analisis dan pengujian" dengan 3 skenario berbeda
untuk memprediksi pergerakan harga emas berdasarkan data historis.
""")

# Tombol untuk menjalankan simulasi
if st.button('Jalankan Analisis & Uji Skenario'):
    
    st.write("Memuat data dan parameter historis...")
    try:
        # 1. Muat parameter
        harga_awal, mean_hist, std_hist = muat_data_dan_parameter()
        
        st.write(f"Data historis di-load. Harga awal untuk simulasi: ${harga_awal:,.2f}")
        st.write(f"Parameter Historis -> Rata-rata: ${mean_hist:.2f}, Std Dev: ${std_hist:.2f}")

        # 2. Tentukan parameter simulasi
        jumlah_hari_simulasi = 365
        mean_krisis = -50  # Parameter skenario krisis
        mean_boom = 60     # Parameter skenario boom
        
        st.write(f"Menjalankan 3 skenario simulasi untuk {jumlah_hari_simulasi} hari ke depan...")

        # 3. Jalankan semua simulasi
        simulasi_baseline = jalankan_simulasi_sistem_dinamis(harga_awal, mean_hist, std_hist, jumlah_hari_simulasi)
        simulasi_krisis = jalankan_simulasi_sistem_dinamis(harga_awal, mean_krisis, std_hist, jumlah_hari_simulasi)
        simulasi_boom = jalankan_simulasi_sistem_dinamis(harga_awal, mean_boom, std_hist, jumlah_hari_simulasi)

        # 4. Buat Grafik
        st.write("Membuat grafik hasil simulasi...")
        
        fig, ax = plt.subplots(figsize=(12, 7)) # 'fig' dan 'ax' adalah cara standar membuat plot
        
        hari_simulasi = np.arange(1, jumlah_hari_simulasi + 1)
        
        ax.plot(hari_simulasi, simulasi_baseline, label=f'Skenario 1: Baseline (Rata-rata ${mean_hist:.2f})', color='blue', linestyle='--')
        ax.plot(hari_simulasi, simulasi_krisis, label=f'Skenario 2: Krisis (Rata-rata ${mean_krisis})', color='red')
        ax.plot(hari_simulasi, simulasi_boom, label=f'Skenario 3: Boom (Rata-rata ${mean_boom})', color='green')
        ax.axhline(y=harga_awal, color='grey', linestyle=':', label=f'Harga Awal: ${harga_awal:,.2f}')
        
        ax.set_title('Simulasi Model Sistem Dinamis Harga Emas (3 Skenario)', fontsize=16)
        ax.set_xlabel('Jumlah Hari ke Depan', fontsize=12)
        ax.set_ylabel('Simulasi Harga Emas', fontsize=12)
        ax.legend()
        ax.grid(True)
        
        # 5. Tampilkan Grafik di Streamlit
        st.pyplot(fig)
        
        st.success("Analisis dan simulasi skenario telah selesai!")

    except FileNotFoundError:
        st.error(f"Error: File 'Gold Price.csv' tidak ditemukan.")
        st.error("Pastikan file 'Gold Price.csv' berada di folder yang sama dengan file 'app.py' ini.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")