import numpy as np

### TAHAP PREPROCESSING DATA DAN INISIALISASI ###

# Fungsi sigmoid dan turunannya
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
# Fungsi Normalisasi data
def normalize(data):
    min_vals = np.min(data)
    max_vals = np.max(data)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data, min_vals, max_vals
# Fungsi Denormalisasi data
def denormalize(normalized_data, min_vals, max_vals):
    return normalized_data * (max_vals - min_vals) + min_vals
# Membaca data dari file
with open('data_training.txt', 'r') as file:
    lines = file.readlines()
data = []
output_data = []
for line in lines:
    values = line.strip().split()                     # Menggunakan spasi sebagai pemisah
    data.append([float(val) for val in values[:-1]])  # Mengambil semua kecuali yang terakhir sebagai input
    output_data.append([float(values[-1])])           # Mengambil yang terakhir sebagai target
input_data = np.array(data)
output_data = np.array(output_data)
# Normalisasi data input dan output
normalized_input_data, input_min_vals, input_max_vals = normalize(input_data)
normalized_output_data, output_min_vals, output_max_vals = normalize(output_data)
# Inisialisasi bobot dan bias
print("\n\nTugas Akhir Jaringan Syaraf Tiruan")
print("\nIMPLEMENTASI ALGORITMA BACKPROPAGATION UNTUK MEMPREDIKSI HARGA SAHAM PT. TELEKOMUNIKASI INDONESIA, TBK\n")
print("Dibuat oleh: 1. Ilham Farhan Z. (082011233070)")
print("             2. Akrom Fuadi     (082011233079)\n")
print("\nINISIALISASI\n")
input_neurons   = int(input("Masukkan jumlah node pada Input Layer  : "))
hidden_neurons  = int(input("Masukkan jumlah node pada Hidden Layer : "))
output_neurons  = int(input("Masukkan jumlah node pada Output Layer : "))
learning_rate   = float(input("Masukkan Learning Rate (Alpha)         : "))
max_epochs      = int(input("Masukkan Maksimum Epoch                : "))
error_threshold = float(input("Masukkan Maksimum Error                : "))
# Menampilkan hasil normalisasi
print("\n\nPREPROCESSING DATA")
print("\nContoh hasil Normalisasi data :")
print(normalized_input_data[:3])   # Contoh menampilkan 3 baris pertama
print("\nContoh hasil Normalisasi data (Target):")
print(normalized_output_data[:3])  # Contoh menampilkan 3 baris pertama


### TAHAP TRAINING DATA ###

print("\n\nTAHAP TRAINING DATA")
# Generate bobot random
np.random.seed(1)
# Bobot untuk lapisan tersembunyi dan lapisan keluaran
hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
output_weights = np.random.uniform(size=(hidden_neurons, output_neurons))
# Bias untuk lapisan tersembunyi dan lapisan keluaran
hidden_bias = np.random.uniform(size=(1, hidden_neurons))
output_bias = np.random.uniform(size=(1, output_neurons))
# Menghitung MSE
print("\nMSE dari setiap 100 epoch adalah:")
for epoch in range(max_epochs):
    # Feed Forward
    hidden_layer_input = np.dot(normalized_input_data, hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, output_weights) + output_bias
    output = sigmoid(output_layer_input)
    # Menghitung error dan Mean Squared Error (MSE)
    error = normalized_output_data - output
    mse = np.mean(np.square(error))
    # Menghentikan pelatihan jika MSE kurang dari threshold
    if mse < error_threshold:
        print(f"Mencapai MSE kurang dari {error_threshold} setelah {epoch} epochs")
        break
    # Backpropagation
    d_output = error * sigmoid_derivative(output)
    error_hidden_layer = d_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    # Memperbarui bobot dan bias
    output_weights += hidden_layer_output.T.dot(d_output) * learning_rate
    output_bias += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    hidden_weights += normalized_input_data.T.dot(d_hidden_layer) * learning_rate
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
    # Menampilkan MSE
    if epoch % 100 == 0:
        print(f"Epoch {epoch} - MSE: {mse:.7f}")
# Jika mencapai batas max_epochs
if epoch == max_epochs - 1:
    print("Mencapai batas maksimum epoch tanpa mencapai MSE yang diinginkan.")
# Feed Forward untuk prediksi
def predict(input):
    hidden_layer_input = np.dot(input, hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, output_weights) + output_bias
    output = sigmoid(output_layer_input)
    return output
# Prediksi
predictions = predict(normalized_input_data)
# Denormalisasi hasil prediksi
denormalized_predictions = denormalize(predictions, output_min_vals, output_max_vals)
# Menghitung MSE setelah denormalisasi
mse_denormalized = np.mean(np.square(denormalized_predictions - output_data))
# Menampilkan MSE terakhir
print(f"\nMSE terakhir setelah Denormalisasi: {mse:.7f}")
# Menampilkan hasil prediksi setelah denormalisasi
print("\nHasil Prediksi Setelah Denormalisasi:")
for i in range(len(normalized_input_data)):
    predicted_value = denormalized_predictions[i][0]
    actual_value = output_data[i][0]
    error = abs(predicted_value - actual_value)
    print(f"Input: {input_data[i]} -> Prediksi: {predicted_value:.4f} | Aktual: {actual_value} | Selisih: {error:.4f}")
# Menampilkan Output bobot dan bias terakhir
print("\nBobot Terakhir untuk Hidden Layer:")
print(hidden_weights)
print("\nBobot Bias Terakhir untuk Hidden Layer:")
print(hidden_bias)
print("\nBobot Terakhir untuk Output Layer:")
print(output_weights)
print("\nBobot Bias Terakhir untuk Output Layer:")
print(output_bias)


### TAHAP TESTING DATA ###

print("\n\nTAHAP TESTING DATA\n")
# Normalisasi data
def normalize2(data2):
    min_vals2 = np.min(data2)
    max_vals2 = np.max(data2)
    normalized_data2 = (data2 - min_vals2) / (max_vals2 - min_vals2)
    return normalized_data2, min_vals2, max_vals2
# Denormalisasi data
def denormalize2(normalized_data2, min_vals2, max_vals2):
    return normalized_data2 * (max_vals2 - min_vals2) + min_vals2
# Membaca data dari file
with open('data_testing.txt', 'r') as file:
    lines = file.readlines()
data2 = []
output_data2 = []
for line in lines:
    values = line.strip().split()                      # Menggunakan spasi sebagai pemisah
    data2.append([float(val) for val in values[:-1]])  # Mengambil semua kecuali yang terakhir sebagai input
    output_data2.append([float(values[-1])])           # Mengambil yang terakhir sebagai target
input_data2 = np.array(data2)
output_data2 = np.array(output_data2)
# Normalisasi data input dan output
normalized_input_data2, input_min_vals2, input_max_vals2 = normalize2(input_data2)
normalized_output_data2, output_min_vals2, output_max_vals2 = normalize2(output_data2)
# Inisialisasi bobot dan bias
input_neurons2   = input_data2.shape[1]
# Bobot dan bobot bias untuk lapisan tersembunyi 
hidden_weights2 = hidden_weights
hidden_bias2 = hidden_bias
# Bobot dan bobot Bias untuk lapisan keluaran
output_weights2 = output_weights
output_bias2 = output_bias
# Feed Forward
hidden_layer_input2 = np.dot(normalized_input_data2, hidden_weights2) + hidden_bias2
hidden_layer_output2 = sigmoid(hidden_layer_input2)
output_layer_input2 = np.dot(hidden_layer_output2, output_weights2) + output_bias2
output2 = sigmoid(output_layer_input2)
# Menghitung error dan Mean Squared Error (MSE)
error2 = normalized_output_data2 - output2
mse2 = np.mean(np.square(error2))
# Backpropagation
d_output2 = error2 * sigmoid_derivative(output2)
error_hidden_layer2 = d_output2.dot(output_weights2.T)
d_hidden_layer2 = error_hidden_layer2 * sigmoid_derivative(hidden_layer_output2)
# Feed Forward untuk prediksi
def predict2(input):
    hidden_layer_input2 = np.dot(input, hidden_weights2) + hidden_bias2
    hidden_layer_output2 = sigmoid(hidden_layer_input2)
    output_layer_input2 = np.dot(hidden_layer_output2, output_weights2) + output_bias2
    output2 = sigmoid(output_layer_input2)
    return output2
# Prediksi
predictions2 = predict2(normalized_input_data2)
# Denormalisasi hasil prediksi
denormalized_predictions2 = denormalize2(predictions2, output_min_vals2, output_max_vals2)
# Menghitung MSE setelah denormalisasi
mse_denormalized2 = np.mean(np.square(denormalized_predictions2 - output_data2))
# Menampilkan hasil prediksi setelah denormalisasi
print("Hasil Prediksi Setelah Denormalisasi:")
for i in range(len(normalized_input_data2)):
    predicted_value2 = denormalized_predictions2[i][0]
    actual_value2 = output_data2[i][0]
    error2 = abs(predicted_value2 - actual_value2)
    print(f"Input: {input_data2[i]} -> Prediksi: {predicted_value2:.4f} | Aktual: {actual_value2} | Selisih: {error2:.4f}")
# Menampilkan MSE
print(f"\nMSE: {mse2:.7f}")


### TAHAP PREDIKSI ###

# Menggunakan bobot terakhir dari training
hidden_weights3 = hidden_weights
hidden_bias3 = hidden_bias
output_weights3 = output_weights
output_bias3 = output_bias

new_data1 = np.loadtxt('data_prediksi.txt')
def normalize_new_data1(new_data1):
    min_vals1 = np.min(new_data1)
    max_vals1 = np.max(new_data1)
    normalized_new_data1 = (new_data1 - min_vals1) / (max_vals1 - min_vals1)
    return normalized_new_data1, min_vals1, max_vals1
def denormalize_new_data1(value, min_vals1, max_vals1):
    return value * (max_vals1 - min_vals1) + min_vals1
normalized_new_data1, min_vals1, max_vals1 = normalize_new_data1(new_data1)
hidden_layer_input1 = np.dot(normalized_new_data1, hidden_weights3) + hidden_bias3
hidden_layer_output1 = sigmoid(hidden_layer_input1)
output_layer_input1 = np.dot(hidden_layer_output1, output_weights3) + output_bias3
output1 = sigmoid(output_layer_input1)
denormalized_output1 = denormalize_new_data1(output1, min_vals1, max_vals1)
print("\n\nPREDIKSI DATA SELANJUTNYA\n")
print(new_data1)
print(f"Output prediksi nilai saham 1 hari selanjutnya yang telah didenormalisasi : {denormalized_output1}\n")

# Looping untuk prediksi berkelanjutan
for i in range(4):
    new_data2 = new_data1[1:10]
    new_data2 = np.append(new_data2, denormalized_output1)
    normalized_new_data2, min_vals2, max_vals2 = normalize_new_data1(new_data2)
    hidden_layer_input2 = np.dot(normalized_new_data2, hidden_weights3) + hidden_bias3
    hidden_layer_output2 = sigmoid(hidden_layer_input2)
    output_layer_input2 = np.dot(hidden_layer_output2, output_weights3) + output_bias3
    output2 = sigmoid(output_layer_input2)
    denormalized_output2 = denormalize_new_data1(output2, min_vals2, max_vals2)
    new_data1 = np.roll(new_data1, -1)
    new_data1[-1] = denormalized_output2
    denormalized_output1 = denormalized_output2
    print(new_data2)
    print(f"Output prediksi nilai saham {i+2} hari selanjutnya yang telah didenormalisasi : {denormalized_output2}\n")