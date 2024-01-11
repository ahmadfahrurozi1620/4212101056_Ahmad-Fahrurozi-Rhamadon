import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from skimage.feature import hog
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mlxtend.plotting import plot_confusion_matrix

# Muat dataset MNIST
mnist = datasets.fetch_openml('mnist_784', version=1)
x, y = mnist.data.astype('float32'), mnist.target.astype('int')

# Bagi dataset menjadi data latih dan uji
x_train, y_train = np.array(x[:1000]), np.array(y[:1000])
x_tes, y_tes = np.array(x[:20]), np.array(y[:20])


# Ekstraksi fitur HOG untuk data latih
hog_features_train = []
hog_images_train = []
for image in x_train:
    feature, hog_img = hog(image.reshape((28, 28)), 
                           orientations = 9, 
                           pixels_per_cell = (8,8), 
                           cells_per_block = (2,2), 
                           visualize = True, 
                           block_norm = 'L2')
    hog_features_train.append(feature)
    hog_images_train.append(hog_img)

hog_features_train_np = np.array(hog_features_train)
hog_images_train_np = np.array(hog_images_train)
    

# Ekstraksi fitur HOG untuk data uji
hog_features_tes = []
hog_images_tes = []
for image in x_tes:
    feature, hog_img = hog(image.reshape((28, 28)), 
                           orientations = 9, 
                           pixels_per_cell = (8,8), 
                           cells_per_block = (2,2), 
                           visualize = True, 
                           block_norm = 'L2')
    hog_features_tes.append(feature)
    hog_images_tes.append(hog_img)

hog_features_tes_np = np.array(hog_features_tes)
hog_images_tes_np = np.array(hog_images_tes)    
 

# Normalisasi fitur HOG
scaler = StandardScaler()
hog_features_train_scaled = scaler.fit_transform(hog_features_train_np)
hog_features_tes_scaled = scaler.transform(hog_features_tes_np)

# Latih model SVM
svm_model = SVC(random_state=0)
svm_model.fit(hog_features_train_scaled, y_train)

# Lakukan prediksi pada data uji
predictions = svm_model.predict(hog_features_tes_scaled)

# Evaluasi performa
conf_matrix = confusion_matrix(y_tes, predictions)
accuracy = accuracy_score(y_tes, predictions)
precision = precision_score(y_tes, predictions, average='weighted')

# Tampilkan hasil evaluasi
print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy:", accuracy)
print("Precision:", precision)

def plot_combined(x_tes, hog_images_tes):
    fig, axes = plt.subplots(2, 20, figsize=(10, 5))

    # Plot untuk gambar dataset
    for i in range(min(len(x_tes), 20)):  
        axes[0, i].imshow(x_tes[i].reshape((28, 28)), cmap='gray')
        axes[0, i].axis('off')

    # Plot untuk gambar hasil HOG extraction
    for i in range(min(len(hog_images_tes), 20)):  
        axes[1, i].imshow(hog_images_tes[i].reshape((28, 28)), cmap='gray')
        axes[1, i].axis('off')

    class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, class_names=class_name)

    plt.show()

plot_combined(x_tes, hog_images_tes)


# Menampilkan Plot Confussion Matrix






