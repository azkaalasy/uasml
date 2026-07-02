# ==========================================================
# IMPORT LIBRARY
# ==========================================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from analisis import (
    load_prediction_models,
    predict_student_status
)

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(
    page_title="Student Academic Dashboard",
    page_icon="🎓",
    layout="wide"
)

sns.set_style("darkgrid")

# ==========================================================
# LOAD MODEL
# ==========================================================
try:
    model, scaler, pca = load_prediction_models()
except FileNotFoundError:
    st.error("Model belum ditemukan.")
    st.stop()

# ==========================================================
# LOAD DATASET
# ==========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv", sep=";")
    return df


df = load_data()

# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135755.png",
        width=120
    )

    st.title("Filter Dashboard")

    gender_filter = st.selectbox(
        "Gender",
        ["Semua", "Female", "Male"]
    )

    debtor_filter = st.selectbox(
        "Status Debtor",
        ["Semua", "Bukan Debtor", "Debtor"]
    )

# ==========================================================
# FILTER DATA
# ==========================================================
main_df = df.copy()

if gender_filter != "Semua":

    gender_value = 1 if gender_filter == "Male" else 0

    main_df = main_df[
        main_df["Gender"] == gender_value
    ]

if debtor_filter != "Semua":

    debtor_value = 1 if debtor_filter == "Debtor" else 0

    main_df = main_df[
        main_df["Debtor"] == debtor_value
    ]

# ==========================================================
# HEADER
# ==========================================================
st.title("🎓 Student Academic Dashboard")

st.markdown("""
Dashboard ini digunakan untuk menganalisis data akademik mahasiswa
serta melakukan prediksi status akademik menggunakan Machine Learning.
""")

# ==========================================================
# MEMBUAT TAB
# ==========================================================
tab1, tab2 = st.tabs([
    "📊 Analisis Data",
    "🔮 Prediksi Mahasiswa"
])

# ==========================================================
# TAB 1 : ANALISIS DATA
# ==========================================================
with tab1:

    st.subheader("📊 Overview Data Mahasiswa")

    # ======================================================
    # METRIC
    # ======================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Mahasiswa",
            len(main_df)
        )

    with col2:

        avg_age = round(
            main_df["Age at enrollment"].mean(),
            1
        )

        st.metric(
            "Rata-rata Umur",
            f"{avg_age} Tahun"
        )

    with col3:

        if len(main_df) > 0:

            fees_rate = round(
                main_df["Tuition fees up to date"].mean() * 100,
                1
            )

        else:
            fees_rate = 0

        st.metric(
            "UKT Lunas",
            f"{fees_rate}%"
        )

    st.divider()

    # ======================================================
    # BAR CHART TARGET
    # ======================================================

    col4, col5 = st.columns(2)

    with col4:

        st.markdown("### Status Akademik Mahasiswa")

        fig, ax = plt.subplots(figsize=(7,5))

        sns.countplot(
            data=main_df,
            x="Target",
            palette="Set2",
            ax=ax
        )

        ax.set_xlabel("Status")
        ax.set_ylabel("Jumlah")

        st.pyplot(fig)

    # ======================================================
    # SCATTER PLOT
    # ======================================================

    with col5:

        st.markdown("### Usia vs Nilai Semester 1")

        fig, ax = plt.subplots(figsize=(7,5))

        sns.scatterplot(
            data=main_df,
            x="Age at enrollment",
            y="Curricular units 1st sem (grade)",
            hue="Target",
            palette="Set2",
            ax=ax
        )

        ax.set_xlabel("Age at Enrollment")

        ax.set_ylabel("Grade Semester 1")

        st.pyplot(fig)

    st.divider()

    # ======================================================
    # PIE CHART
    # ======================================================

    col6, col7 = st.columns(2)

    with col6:

        st.markdown("### Distribusi Gender")

        gender_count = (
            main_df["Gender"]
            .replace({
                0: "Female",
                1: "Male"
            })
            .value_counts()
        )

        fig, ax = plt.subplots(figsize=(6,6))

        ax.pie(
            gender_count,
            labels=gender_count.index,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.axis("equal")

        st.pyplot(fig)

    # ======================================================
    # HISTOGRAM
    # ======================================================

    with col7:

        st.markdown("### Admission Grade")

        fig, ax = plt.subplots(figsize=(7,5))

        sns.histplot(
            data=main_df,
            x="Admission grade",
            bins=20,
            kde=True,
            ax=ax
        )

        st.pyplot(fig)

    st.divider()

    # ======================================================
    # TABEL DATA
    # ======================================================

    st.subheader("Preview Dataset")

    st.dataframe(
        main_df.head(20),
        use_container_width=True
    )
    
# ==========================================================
# TAB 2 : PREDIKSI STATUS MAHASASISWA
# ==========================================================
with tab2:

    st.subheader("🔮 Prediksi Status Akademik Mahasiswa")

    st.write(
        "Masukkan data mahasiswa untuk memprediksi status akademiknya."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        marital_dict = {
            "Single": 1,
            "Menikah": 2,
            "Duda/Janda": 3,
            "Cerai": 4,
            "Hubungan Sipil (Facto Union)": 5,
            "Pisah Secara Hukum": 6
        }
        marital = st.selectbox(
            "Status Pernikahan",
            marital_dict.keys()
            )
        
        marital_status = marital_dict[marital]
        
        qualification_dict = {
            "SMA/SMK Sederajat (Lulus)": 1,
            "Sarjana (Bachelor)": 2,
            "Sarjana (Degree)": 3,
            "Magister (S2)": 4,
            "Doktor (S3)": 5,
            "Pernah Kuliah Sebelumnya": 6,
            "Belum Lulus SMA Kelas 12": 9,
            "Belum Lulus SMA Kelas 11": 10,
            "Sekolah Lama Kelas 7": 11,
            "Pendidikan Lain Setara Kelas 11": 12,
            "Program SMA Tambahan Tahun ke-2": 13,
            "Lulus Kelas 10": 14,
            "Jurusan Perdagangan Umum": 18,
            "SMP/Sederajat (Kelas 9-11)": 19,
            "Program SMA Pelengkap": 20,
            "SMK Teknik/Profesional": 22,
            "Program SMA Pelengkap (Belum Lulus)": 25,
            "Lulus Kelas 7": 26,
            "Program Umum SMA Siklus 2": 27,
            "Belum Lulus Kelas 9": 29,
            "Lulus Kelas 8": 30,
            "Administrasi & Perdagangan": 31,
            "Akuntansi & Administrasi Tambahan": 33,
            "Tidak Diketahui": 34,
            "Tidak Bisa Membaca dan Menulis": 35,
            "Bisa Membaca, Tidak Tamat SD": 36,
            "SD/Sederajat": 37,
            "SMP/Sederajat": 38,
            "Program Spesialisasi Teknologi": 39,
            "Sarjana Siklus 1": 40,
            "Program Studi Lanjutan": 41,
            "Diploma Tinggi Profesional": 42,
            "Magister Siklus 2": 43,
            "Doktor Siklus 3": 44
        }
        
        qualification = st.selectbox(
            "Pendidikan Sebelumnya",
            qualification_dict.keys()
        )
        
        previous_qualification = qualification_dict[qualification]

        # PERBAIKAN: Mengubah min_value agar tidak 0.0 (Data outlier bagi model)
        previous_qualification_grade = st.slider(
            "Previous Qualification Grade",
            95.0,
            200.0,
            120.0
        )

        nationality_dict = {
            "Portuguese": 1,
            "German": 2,
            "Spanish": 6,
            "Italian": 11,
            "Dutch": 13,
            "English": 14,
            "Lithuanian": 17,
            "Angolan": 21,
            "Cape Verdean": 22,
            "Guinean": 24,
            "Mozambican": 25,
            "Brazilian": 26,
            "Romanian": 32,
            "Moldovan": 41,
            "Mexican": 62,
            "Ukrainian": 100,
            "Russian": 101,
            "Cuban": 103,
            "Colombian": 105
        }
        
        nationality_name = st.selectbox(
            "Nationality",
            list(nationality_dict.keys())
        )
        
        nationality = nationality_dict[nationality_name]

        admission_grade = st.slider(
            "Admission Grade",
            95.0,
            200.0,
            120.0
        )

        debtor = st.selectbox(
            "Debtor",
            [0,1],
            format_func=lambda x:"Tidak" if x==0 else "Ya"
        )

        tuition = st.selectbox(
            "Tuition Fees Up To Date",
            [1,0],
            format_func=lambda x:"Ya" if x==1 else "Tidak"
        )

        gender = st.selectbox(
            "Gender",
            [0,1],
            format_func=lambda x:"Female" if x==0 else "Male"
        )

    with col2:

        scholarship = st.selectbox(
            "Scholarship Holder",
            [0,1],
            format_func=lambda x:"Tidak" if x==0 else "Ya"
        )

        age = st.number_input(
            "Age at Enrollment",
            min_value=15,
            max_value=70,
            value=19
        )

        approved1 = st.number_input(
            "Curricular Units 1st Sem (Approved)",
            min_value=0,
            max_value=30,
            value=6
        )

        grade1 = st.slider(
            "Curricular Units 1st Sem (Grade)",
            0.0,
            20.0,
            13.0
        )

        approved2 = st.number_input(
            "Curricular Units 2nd Sem (Approved)",
            min_value=0,
            max_value=30,
            value=6
        )

        grade2 = st.slider(
            "Curricular Units 2nd Sem (Grade)",
            0.0,
            20.0,
            13.0
        )

        unemployment = st.slider(
            "Unemployment Rate",
            0.0,
            20.0,
            10.8
        )

    # PERBAIKAN: Menyamakan format penulisan kunci dictionary dengan dataset latih
   # Ganti kembali ke huruf 'c' agar sesuai dengan bentukan pas training model dulu
    input_data = {
        "Marital Status": marital_status,
        "Previous qualification": previous_qualification,
        "Previous qualification (grade)": previous_qualification_grade,
        "Nacionality": nationality,  # <-- CUKUP UBAH BAGIAN INI SAJA
        "Admission grade": admission_grade,
        "Debtor": debtor,
        "Tuition fees up to date": tuition,
        "Gender": gender,
        "Scholarship holder": scholarship,
        "Age at enrollment": age,
        "Curricular units 1st sem (approved)": approved1,
        "Curricular units 1st sem (grade)": grade1,
        "Curricular units 2nd sem (approved)": approved2,
        "Curricular units 2nd sem (grade)": grade2,
        "Unemployment rate": unemployment
    }

    st.divider()

    if st.button(
        "🚀 Jalankan Prediksi",
        use_container_width=True
    ):

        hasil = predict_student_status(
            input_data,
            model,
            scaler,
            pca
        )

        st.subheader("Hasil Prediksi")

        if hasil == "Graduate":

            st.success(
                "🎓 Mahasiswa diprediksi **Graduate**."
            )

        elif hasil == "Enrolled":

            st.info(
                "📘 Mahasiswa diprediksi masih **Enrolled**."
            )

        else:

            st.error(
                "⚠️ Mahasiswa diprediksi **Dropout**."
            )

st.divider()

st.caption("© 2026 Student Academic Dashboard")