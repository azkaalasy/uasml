import pickle
import pandas as pd

def load_prediction_models():
    """
    Fungsi untuk memuat model, scaler, dan PCA
    """
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler(4).pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("pca(4).pkl", "rb") as f:
        pca = pickle.load(f)

    return model, scaler, pca

def predict_student_status(input_data, model, scaler, pca):
    # 1. Mengubah data input dari Streamlit menjadi DataFrame
    df_input = pd.DataFrame([input_data])

    # 2. Paksa urutan kolom agar SAMA PERSIS dengan milik scaler/model
    feature_names = scaler.feature_names_in_
    df_input = df_input[feature_names]

    # 3. Skalasi data menggunakan Scaler yang urutannya sudah benar
    scaled_data = scaler.transform(df_input)

    # 4. Langsung prediksi menggunakan data hasil scale
    prediction = model.predict(scaled_data)

    # 5. Mapping class hasil prediksi menjadi teks label
    label = {
        0: "Dropout",
        1: "Enrolled",
        2: "Graduate"
    }

    return label[prediction[0]]