from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
from scipy.signal import find_peaks, welch
from pywt import wavedec

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

def calculate_features(row):
    zero_crossing_rate = ((np.diff(np.sign(row)) != 0).sum()) / len(row)
    wavelet_coeffs = wavedec(row, 'db4', level=5)
    wavelet_band_21_mean = np.mean(wavelet_coeffs[4])
    std_freq_domain = np.std(np.abs(np.fft.fft(row)))
    median_time_domain = np.median(row)
    return [zero_crossing_rate, wavelet_band_21_mean, std_freq_domain, median_time_domain]

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'csv_file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['csv_file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        data = pd.read_csv(file, header=None)

        features = []
        for index, row in data.iterrows():
            row_values = calculate_features(row)
            features.append(row_values)

        input_query = np.array(features)
        results_dict = {}
        for i, sublist in enumerate(input_query):
            result = model.predict([sublist])
            results_dict[f'Day{i}'] = str(result)

        return jsonify(results_dict)


        

if __name__ == '__main__':
    app.run(debug=True)
