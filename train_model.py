import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models #type: ignore

DATA_DIR = "gesture_dataset"

X = []
y = []
labels = []


for idx, file in enumerate(os.listdir(DATA_DIR)):
    if file.endswith(".npy"):
        data = np.load(os.path.join(DATA_DIR, file))
        X.extend(data)
        y.extend([idx] * len(data))
        labels.append(file.replace(".npy", ""))

X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))
print("Gestures:", labels)


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y
)


model = models.Sequential([
    layers.Input(shape=(63,)),

    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(64, activation='relu'),

    layers.Dense(len(labels), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32
)


model.save("gesture_model.h5")
np.save("labels.npy", labels)

print("\nModel saved as gesture_model.h5")

