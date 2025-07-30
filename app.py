import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# KONFIGURASI HALAMAN & FUNGSI BANTUAN
# ==============================================================================
st.set_page_config(layout="wide", page_title="Dashboard Hasil Analisis Sentimen")
st.title("Dashboard Visualisasi Hasil Analisis Sentimen SVM")

# Fungsi untuk memplot confusion matrix dari data array yang sudah ada
def plot_hardcoded_confusion_matrix(cm_data, labels, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    return fig

# ==============================================================================
# 1. LOAD DATA (Hanya untuk konteks)
# ==============================================================================
with st.container(border=True):
    st.header("📁 Pemuatan Data")
    st.write("Bagian ini menampilkan cuplikan data yang digunakan dalam analisis untuk memberikan konteks.")
    try:
        df = pd.read_csv("hasil_proses.csv")
        st.dataframe(df.head())
        st.info(f"Total data yang dianalisis: **{len(df)} baris**.")
    except FileNotFoundError:
        st.error("File 'hasil_proses.csv' tidak ditemukan. Silakan letakkan di folder yang sama.")
        st.stop()

# ==============================================================================
# 2. EVALUASI AWAL (BERDASARKAN GAMBAR)
# ==============================================================================
with st.container(border=True):
    st.header("📊 Tahap 1: Evaluasi Awal SVM")
    st.write("Tahap ini menampilkan hasil evaluasi awal sebelum penyetelan parameter")

    # Data dari image_7db465.png: Akurasi: 0.7858, Presisi: 0.8138, Recall: 0.7858
    # Confusion Matrix: TN=83, FP=107, FN=8, TP=339
   
    col1, col2, col3 = st.columns(3)
    col1.metric("Akurasi", "78.58%")
    col2.metric("Presisi", "76.00%")
    col3.metric("Recall", "97.69%")

    cm_initial_data = np.array([[83, 107], [8, 339]])
    fig_initial = plot_hardcoded_confusion_matrix(cm_initial_data, ['negatif', 'positif'], "Confusion Matrix Tahap Awal")
    st.pyplot(fig_initial)

# ==============================================================================
# 3. Hyperparameter & PERBANDINGAN KERNEL 
# ==============================================================================
with st.container(border=True):
    st.header("⚙️ Tahap 2: Pencarian Hyperparameter & Perbandingan Kinerja Kernel")
    st.write("Menampilkan parameter terbaik yang ditemukan untuk setiap kernel dan perbandingan kinerjanya")

    st.write("#### Hasil Teks Grid Search:")
    grid_search_text = """
    Hasil Grid Search untuk Kernel Linear:
    Best Params: {'C': 1}
    Best Score: 0.788
    Accuracy: 81.38%
    Precision: 81.36%
    Recall: 81.38%

    Hasil Grid Search untuk Kernel RBF:
    Best Params: {'C': 10, 'gamma': 0.1}
    Best Score: 0.7896000000000001
    Accuracy: 82.12%
    Precision: 82.04%
    Recall: 82.12%

    Hasil Grid Search untuk Kernel Polynomial:
    Best Params: {'C': 1, 'degree': 1}
    Best Score: 0.788
    Accuracy: 81.38%
    Precision: 81.36%
    Recall: 81.38%
    """
    st.code(grid_search_text)

    # --- Visualisasi Grafik Batang (hardcoded dari gambar) ---
    st.write("#### Grafik Perbandingan Kinerja Kernel")
    results_data = {
        'Kernel': ['Linear', 'RBF', 'Polynomial'],
        'Accuracy': [81.38, 82.12, 81.38],
        'Precision': [81.36, 82.04, 81.36],
        'Recall': [81.38, 82.12, 81.38]
    }
    results_df = pd.DataFrame(results_data)
   
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    kernels = results_df['Kernel']
    x_pos = np.arange(len(kernels))
    width = 0.25

    rects1 = ax_bar.bar(x_pos - width, results_df['Accuracy'], width, label='Accuracy', color='skyblue')
    rects2 = ax_bar.bar(x_pos, results_df['Precision'], width, label='Precision', color='orange')
    rects3 = ax_bar.bar(x_pos + width, results_df['Recall'], width, label='Recall', color='green')

    ax_bar.set_ylabel('Performance (%)')
    ax_bar.set_title('Perbandingan Kinerja Model SVM Berdasarkan Kernel')
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(kernels)
    ax_bar.legend()
    ax_bar.set_ylim(75, 85)
    ax_bar.grid(axis='y', linestyle='--')
    ax_bar.bar_label(rects1, padding=3, fmt='%.2f%%')
    ax_bar.bar_label(rects2, padding=3, fmt='%.2f%%')
    ax_bar.bar_label(rects3, padding=3, fmt='%.2f%%')
    st.pyplot(fig_bar)

# ==============================================================================
# 4. HASIL TERBAIK & VISUALISASI AKHIR
# ==============================================================================
with st.container(border=True):
    st.header("🏆 Tahap 3: Visualisasi Hasil Terbaik (Kernel RBF, Rasio 90:10)")
    st.write("Setelah diketahui kernel RBF adalah yang terbaik, model dilatih kembali. Hasil terbaik didapat pada rasio 90:10")

    col1, col2, col3 = st.columns(3)
    col1.metric("Akurasi", "84.35%")
    col2.metric("Presisi", "84.09%")
    col3.metric("Recall", "94.06%")

    cm_best_data = np.array([[40, 21], [7, 111]])
    fig_best = plot_hardcoded_confusion_matrix(cm_best_data, ['negatif', 'positif'], "Confusion Matrix Hasil Akhir Terbaik")
    st.pyplot(fig_best)

    st.info(f"**Total Prediksi Benar:** 151\n\n**Total Keseluruhan Data:** 179")

# ==============================================================================
# 5. K-FOLD CROSS VALIDATION 
# ==============================================================================
with st.container(border=True):
    st.header("🔁 Tahap 4: Hasil 10-Fold Cross Validation")
    st.write("Menampilkan hasil pengujian konsistensi model terbaik menggunakan validasi silang 10-Fold")

    kfold_text = """
Fold: 1
Akurasi: 83.80%
Presisi: 83.91%
Recall: 83.80%

Fold: 2
Akurasi: 82.68%
Presisi: 83.83%
Recall: 82.68%

Fold: 3
Akurasi: 79.89%
Presisi: 80.07%
Recall: 79.89%

Fold: 4
Akurasi: 81.01%
Presisi: 80.73%
Recall: 81.01%

Fold: 5
Akurasi: 74.30%
Presisi: 73.75%
Recall: 74.30%

Fold: 6
Akurasi: 81.56%
Presisi: 81.18%
Recall: 81.56%

Fold: 7
Akurasi: 87.15%
Presisi: 87.01%
Recall: 87.15%

Fold: 8
Akurasi: 79.78%
Presisi: 79.33%
Recall: 79.78%

Fold: 9
Akurasi: 78.09%
Presisi: 78.44%
Recall: 78.09%

Fold: 10
Akurasi: 77.53%
Presisi: 79.25%
Recall: 77.53%
=======================
Rata-rata Akurasi: 80.58%
Rata-rata Presisi: 80.75%
Rata-rata Recall: 80.58%
    
Fold dengan performa terbaik: 7
Akurasi terbaik: 87.15%
Presisi terbaik: 87.01%
Recall terbaik: 87.15%
    """
    st.code(kfold_text)