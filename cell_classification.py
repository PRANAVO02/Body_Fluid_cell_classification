import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# # Suppress TensorFlow informational messages
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # Define paths
# dataset_path = r'C:\Users\dhars\Downloads\Body fluid cell Dataset\Body fluid cell\Dataset'  # Path where your dataset is located
# output_path = r'C:\Users\dhars\Downloads\Body fluid cell Dataset\Body fluid cell\Preprossed_data'  # Path where you want to save preprocessed data

# # Create output directories
# os.makedirs(output_path, exist_ok=True)

# # Function to load images and labels from a given directory
# def load_data(data_dir):
#     images = []
#     labels = []
#     for class_dir in os.listdir(data_dir):
#         class_path = os.path.join(data_dir, class_dir)
#         if os.path.isdir(class_path):
#             for img_name in os.listdir(class_path):
#                 img_path = os.path.join(class_path, img_name)
#                 img = cv2.imread(img_path)
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 images.append(img)
#                 labels.append(class_dir)
#     return np.array(images), np.array(labels)

# # Load training data
# train_dir = os.path.join(dataset_path, 'train')
# X_train, y_train = load_data(train_dir)

# # Load validation data
# val_dir = os.path.join(dataset_path, 'valid')
# X_val, y_val = load_data(val_dir)

# # Load test data
# test_dir = os.path.join(dataset_path, 'test')
# X_test, y_test = load_data(test_dir)

# # Normalize images
# X_train = X_train.astype('float32') / 255.0
# X_val = X_val.astype('float32') / 255.0
# X_test = X_test.astype('float32') / 255.0

# # Encode labels
# label_encoder = LabelEncoder()
# y_train = label_encoder.fit_transform(y_train)
# y_val = label_encoder.transform(y_val)
# y_test = label_encoder.transform(y_test)

# # Data augmentation
# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True,
#     vertical_flip=True,
#     zoom_range=0.2
# )

# # Save preprocessed data
# np.save(os.path.join(output_path, 'X_train.npy'), X_train)
# np.save(os.path.join(output_path, 'y_train.npy'), y_train)
# np.save(os.path.join(output_path, 'X_val.npy'), X_val)
# np.save(os.path.join(output_path, 'y_val.npy'), y_val)
# np.save(os.path.join(output_path, 'X_test.npy'), X_test)
# np.save(os.path.join(output_path, 'y_test.npy'), y_test)

# # Save label encoder
# with open(os.path.join(output_path, 'label_encoder.npy'), 'wb') as f:
#     np.save(f, label_encoder.classes_)
    
    #train and load model
    


# # Load preprocessed data
# output_path = r'C:\Users\dhars\Downloads\Body fluid cell Dataset\Body fluid cell\Preprossed_data'

# X_train = np.load(os.path.join(output_path, 'X_train.npy'))
# y_train = np.load(os.path.join(output_path, 'y_train.npy'))
# X_val = np.load(os.path.join(output_path, 'X_val.npy'))
# y_val = np.load(os.path.join(output_path, 'y_val.npy'))
# X_test = np.load(os.path.join(output_path, 'X_test.npy'))
# y_test = np.load(os.path.join(output_path, 'y_test.npy'))

# # Convert labels to categorical
# num_classes = len(np.unique(y_train))
# y_train = to_categorical(y_train, num_classes)
# y_val = to_categorical(y_val, num_classes)
# y_test = to_categorical(y_test, num_classes)

# # Define the model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(num_classes, activation='softmax')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Define early stopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # Train the model
# history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# # Evaluate the model
# test_loss, test_acc = model.evaluate(X_test, y_test)
# print(f'Test accuracy: {test_acc:.2f}')

# # Save the model
# model.save(os.path.join(output_path, 'cell_classification_model.h5')) 

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load preprocessed data
output_path = r'C:\Users\dhars\Downloads\Body fluid cell Dataset\Body fluid cell\Preprossed_data'

X_test = np.load(os.path.join(output_path, 'X_test.npy'))
y_test = np.load(os.path.join(output_path, 'y_test.npy'))

# Load the trained model
model = tf.keras.models.load_model(os.path.join(output_path, 'cell_classification_model.h5'))

# Predict on the test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()