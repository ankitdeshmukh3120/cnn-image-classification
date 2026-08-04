import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import (
    Input, Dense, Conv2D, Flatten, Rescaling, MaxPool2D, 
    Dropout, RandomFlip, RandomRotation, RandomZoom
)
from keras.utils import image_dataset_from_directory
from keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# ==========================================
# 1. LOAD DATASETS
# ==========================================
training_dataset = image_dataset_from_directory(
    "dataset/train",
    image_size=(128, 128),
    color_mode="rgb",
    validation_split=0.1,
    subset="training",
    seed=17
)

validation_dataset = image_dataset_from_directory(
    "dataset/train",
    image_size=(128, 128),
    color_mode="rgb",
    validation_split=0.1,
    subset="validation",
    seed=17
)

# shuffle=False ensures clean alignment for test set metrics
testing_dataset = image_dataset_from_directory(
    "dataset/test",
    image_size=(128, 128),
    color_mode="rgb",
    shuffle=False
)

class_names = testing_dataset.class_names
print("Classes:", class_names)

# Optimize pipeline for faster epoch times
AUTOTUNE = 1
training_dataset = training_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
testing_dataset = testing_dataset.prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 2. BUILD CNN MODEL
# ==========================================
model = Sequential([
    Input(shape=(128, 128, 3)),
    
    # Data Augmentation to prevent overfitting
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
    
    # Normalization (0-255 to 0-1)
    Rescaling(1/255),

    Conv2D(32, (3, 3), activation="relu"),
    MaxPool2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPool2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPool2D(2, 2),

    Conv2D(256, (3, 3), activation="relu"),
    MaxPool2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.summary()

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Monitor val_loss to catch overfitting early
stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# ==========================================
# 3. TRAIN MODEL
# ==========================================
history = model.fit(
    training_dataset,
    epochs=40,
    validation_data=validation_dataset,
    callbacks=[stop]
)

# ==========================================
# 4. SAVE MODEL
# ==========================================
model.save("cats_vs_dogs_cnn.keras")

# ==========================================
# 5. GRAPH 1: ACCURACY PLOT
# ==========================================
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy", linewidth=2)
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
plt.title("Training vs Validation Accuracy", fontsize=14, fontweight="bold")
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("accuracy_graph.png", dpi=300)
plt.show()

# ==========================================
# 6. GRAPH 2: LOSS PLOT
# ==========================================
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss", linewidth=2)
plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
plt.title("Training vs Validation Loss", fontsize=14, fontweight="bold")
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("loss_graph.png", dpi=300)
plt.show()

# ==========================================
# 7. EVALUATE ON TEST DATASET
# ==========================================
print("\n" + "=" * 45)
print("   EVALUATING MODEL ON UNSEEN TEST DATASET   ")
print("=" * 45)

test_loss, test_accuracy = model.evaluate(testing_dataset)

print(f"\nFinal Test Loss:     {test_loss:.4f}")
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%\n")

# ==========================================
# 8. GRAPH 3: TRAIN VS VAL VS TEST BAR CHART
# ==========================================
final_train_acc = history.history["accuracy"][-1]
final_val_acc = history.history["val_accuracy"][-1]

categories = ["Train", "Validation", "Test"]
accuracies = [final_train_acc, final_val_acc, test_accuracy]

plt.figure(figsize=(7, 5))
bars = plt.bar(categories, accuracies, color=["#3498db", "#f39c12", "#2ecc71"], width=0.45)
plt.title("Model Performance Across Datasets", fontsize=14, fontweight="bold")
plt.ylabel("Accuracy", fontsize=12)
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle="--", alpha=0.5)

# Display numerical percentages above each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval * 100:.1f}%", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("train_val_test_comparison.png", dpi=300)
plt.show()

# ==========================================
# 9. GRAPH 4: CONFUSION MATRIX
# ==========================================
y_true = []
y_pred = []

for images, labels in testing_dataset:
    preds = model.predict(images, verbose=0)
    preds = (preds > 0.5).astype(int).flatten()
    y_true.extend(labels.numpy().flatten())
    y_pred.extend(preds)

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
plt.title("Confusion Matrix (Test Set)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# ==========================================
# 10. GRAPH 5: SAMPLE PREDICTIONS VISUAL GRID
# ==========================================
plt.figure(figsize=(10, 10))
for images, labels in testing_dataset.take(1):
    preds = model.predict(images, verbose=0)
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        # Convert float image back to uint8 for proper visualization display
        plt.imshow(images[i].numpy().astype("uint8"))
        actual = int(labels[i])
        predicted = 1 if preds[i] > 0.5 else 0
        
        color = "green" if actual == predicted else "red"
        plt.title(f"Actual: {class_names[actual]}\nPred: {class_names[predicted]}", color=color, fontweight="bold")
        plt.axis("off")

plt.tight_layout()
plt.savefig("prediction_results.png", dpi=300)
plt.show()