# %% [markdown]
# # 1. Import Libraries

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import image_dataset_from_directory
import shutil
import random
import os

# %% [markdown]
# # 2. Load Dataset

# %% [markdown]
# ## 2.2 Loading Dataset into Notebook.

# %%
# Defining the root path given by Kaggle
root_dir = "/kaggle/input/datasets/pacificrm/skindiseasedataset/SkinDisease/SkinDisease"

# Listing the contents to see if 'train' and 'test' are present
print(os.listdir(root_dir))

# %%

train_dir = os.path.join(root_dir, "train")
test_dir = os.path.join(root_dir, "test")

allowed_classes = sorted([f for f in os.listdir(train_dir)])
# Loading clean raw training data
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(256, 256),
    color_mode='rgb',
    label_mode='categorical',
    batch_size=32,
    shuffle=True
)


# Loading clean raw testing data
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size = (256, 256),
    color_mode = 'rgb',
    label_mode = 'categorical',
    batch_size = 32,
    shuffle = False
)

# %% [markdown]
# ### 2.3 Displaying Training Dataset Classes

# %%
train_class_names = train_dataset.class_names
train_class_names

# %% [markdown]
# ## 2.4 Displaying Test Dataset Classes

# %%
test_class_names = test_dataset.class_names
test_class_names

# %% [markdown]
# # 3. Visualising the Dataset (EDA)

# %%
#  Extracting exactly one batch (32 images and labels) from the pipeline
for images, labels in train_dataset.take(1):
    plt.figure(figsize=(12, 12))

    #  Plotting the first 9 images from this batch in a 3x3 grid
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)

        # Converting the image tensor to integers (0-255) so matplotlib can render it
        img = images[i].numpy().astype("uint8")
        plt.imshow(img)

        # Finding the active class index from your one-hot encoded label vector
        label_index = np.argmax(labels[i].numpy())
        current_class_name = train_class_names[label_index]

        plt.title(current_class_name, fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 3.1 Observation(s) after raw data visualization: 
# ### A class name "Unknown_Normal", present in both Train and Test folders, was found to contain largely images of objects e.g cars, pressing iron, basket e.t.c. and of normal people and animals with no skin diseases. we decided to keep it. it will enable our models to classify images other than skin diseases in the rest 21 classes as "Unknown_Normal".

# %% [markdown]
# # 4. Data Preprocessing

# %% [markdown]
# ## 4.1 Scan Folder Counts and Find the Smallest Subfolder

# %%
root_dir = "/kaggle/input/datasets/pacificrm/skindiseasedataset/SkinDisease/SkinDisease"
# Using the original train directory inside Kaggle input
train_dir = os.path.join(root_dir, "train")

print("--- Counting Images in Each Allowed Category ---")
folder_counts = {}

# Scanning all subfolders except the one we dropped
for folder in sorted(os.listdir(train_dir)):
    folder_path = os.path.join(train_dir, folder)
    if os.path.isdir(folder_path):
        # Count only image files
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        folder_counts[folder] = len(images)
        print(f"Category: {folder:<30} | Images: {len(images)}")

# Finding the category with the least amount of content
least_folder = min(folder_counts, key=folder_counts.get)
min_count = folder_counts[least_folder]

print("\n" + "="*50)
print(f"LEAST CONTENT FOLDER: '{least_folder}'")
print(f"EXACT NUMBER TO EXTRACT FROM EACH CLASS: {min_count} images")
print("="*50)

# %% [markdown]
# # 5. Model Building

# %% [markdown]
# ## 5.1 Model (CNN,Conv2D)

# %%
model = tf.keras.models.Sequential([
    # Input shape explicitly accepts 256x256 RGB images (Pixels are already 0.0-1.0)
    tf.keras.layers.Input(shape=(256, 256, 3)),

    # Block 1
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),


    # Block 2
    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', padding = 'same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),


    # Block 3
    tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', padding = 'same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),


    # Block 4
    tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu', padding = 'same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Flatten & Dense Classifier
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5), # Strong dropout to stop overfitting

    # Output Layer for 22 Classes
    tf.keras.layers.Dense(22, activation = 'softmax')
])


model.summary()

# %% [markdown]
# ### 5.1.1 Compile Model (CNN,Conv2D)

# %%
# Compiling the model with a precise learning rate
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), # Low learning rate for fine details
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

print("Model Compile Completed!")

# %% [markdown]
# ### 5.1.2 Defining the Callbacks (CNN,Conv2D)

# %%
#  Stopping the training early to save time and prevent overfitting
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=4,                  # Stops if val_loss doesn't improve for 4 epochs
    restore_best_weights=True    # Keeps the absolute best version of your model
)

#  This lowers learning rate automatically if the model plateaus
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.2,                  # Cuts learning rate by 80% (e.g., 0.0001 -> 0.00002)
    patience = 2,                  # Waits 2 epochs before dropping the rate
    min_lr = 1e-6                  # Do not drop lower than this
)

callbacks_list = [early_stop, reduce_lr]

# %% [markdown]
# ### 5.1.3 Executing the Fit Command (CNN,Conv2D)

# %%
print("Starting training execution on Kaggle environment...")

# This automatically streams the preprocessed x and y batches to the network
history = model.fit(
    train_dataset,
    validation_data = test_dataset,
    epochs = 20,
    callbacks = callbacks_list
)

print("\nModel training execution finished successfully!")

# %% [markdown]
# ### 5.1.4 Fine-Tuning Fit Command (CNN,Conv2D)

# %%
print("Starting fine-tuning training...")

# Continue training from where the previous history left off
fine_tune_history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs = 15,
    callbacks = callbacks_list # Keeps EarlyStopping and ReduceLROnPlateau active
)

print("Fine-tuning complete!")

# %% [markdown]
# ### 5.1.5 Generate the Confusion Matrix (CNN,Conv2D)

# %%

print("Extracting predictions from the validation dataset...")

allowed_classes = sorted([f for f in os.listdir(train_dir)])

#  Gathering all true labels and model predictions
all_true_labels = []
all_pred_labels = []

for images, labels in test_dataset:
    preds = model.predict(images, verbose = 0)
    
    # Converting one-hot vectors into integer class indices
    all_true_labels.extend(np.argmax(labels.numpy(), axis = 1))
    all_pred_labels.extend(np.argmax(preds, axis=1))

#  Computing the confusion matrix metrics
cm = confusion_matrix(all_true_labels, all_pred_labels)

#  Plotting the matrix using Seaborn
plt.figure(figsize = (16, 14))
sns.heatmap(
    cm, 
    annot = True, 
    fmt = 'd', 
    cmap = 'Blues',
    xticklabels = allowed_classes, 
    yticklabels = allowed_classes
)
plt.title('Skin Disease Classification Confusion Matrix')
plt.ylabel('Actual Medical Disease')
plt.xlabel('Predicted Medical Disease')
plt.xticks(rotation = 45, ha = 'right')
plt.yticks(rotation = 0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.1.6 Print the Classification Report (CNN,Conv2D)

# %%
# Printting a full performance breakdown text report
print("\n" + "="*60)
print("             DETAILED CLASSIFICATION REPORT")
print("="*60)
print(classification_report(all_true_labels, all_pred_labels, target_names = allowed_classes))

# %% [markdown]
# ### 5.1.7 Saving the Model (CNN,Conv2D)

# %%
# Saving the final, optimized model file
model.save("/kaggle/working/optimized_skin_2_Conv2D.keras")
print("\nSuccess! Your final model is saved as 'optimized_skin_2_Conv2D.keras'.")
print("You can download it right now from the 'Output' section in the Kaggle right sidebar.")

# %% [markdown]
# ## 5.2 ResNet50V2 Model

# %%
#  Loading the pre-trained ResNet50V2 base model
base_resnet = tf.keras.applications.ResNet50V2(
    input_shape = (256, 256, 3),
    include_top = False,
    weights = 'imagenet'
)
base_resnet.trainable = False # Freeze weights first

#  Building the model structure
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape  =(256, 256, 3)),

    # This automatically rescales the raw pixels to exactly what ResNet requires
    tf.keras.layers.Lambda(tf.keras.applications.resnet_v2.preprocess_input),

    base_resnet,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(22, activation = 'softmax') # 22 skin disease categories
])


model.summary

# %% [markdown]
# ### 5.2.1 Compile Model (ResNet50V2)

# %%
#  Compiling with a standard learning rate
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

print("Model Compile Completed")

# %% [markdown]
# ### 5.2.2 Executing the Fit Command (ResNet50V2)

# %%
# Resetting callbacks to catch performance plateaus accurately
callbacks_list = [
    tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 2)
]

print("Executing fresh training run...")
history = model.fit(
    train_dataset,
    validation_data = test_dataset,
    epochs = 10,
    callbacks = callbacks_list
)

print("Model execution Completed!!!")

# %% [markdown]
# ### 5.2.3 Generate the Confusion Matrix (ResNet50V2)

# %%

print("Extracting predictions from the validation dataset...")

allowed_classes = sorted([f for f in os.listdir(train_dir)])

#  Gathering all true labels and model predictions
all_true_labels = []
all_pred_labels = []

for images, labels in test_dataset:
    preds = model.predict(images, verbose = 0)
    
    # Converting one-hot vectors into integer class indices
    all_true_labels.extend(np.argmax(labels.numpy(), axis = 1))
    all_pred_labels.extend(np.argmax(preds, axis=1))

#  Computing the confusion matrix metrics
cm = confusion_matrix(all_true_labels, all_pred_labels)

#  Plotting the matrix using Seaborn
plt.figure(figsize = (16, 14))
sns.heatmap(
    cm, 
    annot = True, 
    fmt = 'd', 
    cmap = 'Blues',
    xticklabels = allowed_classes, 
    yticklabels = allowed_classes
)
plt.title('Skin Disease Classification Confusion Matrix')
plt.ylabel('Actual Medical Disease')
plt.xlabel('Predicted Medical Disease')
plt.xticks(rotation = 45, ha = 'right')
plt.yticks(rotation = 0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.2.4 Print the Classification Report (ResNet50V2)

# %%
# Printting a full performance breakdown text report
print("\n" + "="*60)
print("             DETAILED CLASSIFICATION REPORT")
print("="*60)
print(classification_report(all_true_labels, all_pred_labels, target_names = allowed_classes))

# %% [markdown]
# ### 5.2.5 Calculate and Apply Class Weights (ResNet50V2)

# %%
print("Calculating class weights to resolve mixing issues...")

#  Extractting all raw integer labels from the training set
train_labels = []
for _, labels in train_dataset:
    train_labels.extend(np.argmax(labels.numpy(), axis = 1))
train_labels = np.array(train_labels)

#  Computting balanced weights for the 22 categories
unique_classes = np.unique(train_labels)
computed_weights = class_weight.compute_class_weight(
    class_weight = 'balanced',
    classes = unique_classes,
    y = train_labels
)

#  Formatting into a dictionary required by Keras
class_weight_dict = dict(zip(unique_classes, computed_weights))

# Verifying the weights (confusing/rare classes will have values > 1.0)
print("\nClass weights calculated successfully!")
for idx, class_name in enumerate(allowed_classes):
    print(f"Class {idx} ({class_name}): Weight = {class_weight_dict[idx]:.2f}")

# %% [markdown]
# ### 5.2.6 Retrain with the Class Weights (ResNet50V2)

# %%
print("\nRestarting training with targeted Class Weights...")

#  Using the stable ResNet50V2 model architecture
history_weighted = model.fit(
    train_dataset,
    validation_data =test_dataset,
    epochs = 12,
    class_weight = class_weight_dict, # This forces the model to fix the mix-ups!
    callbacks = callbacks_list
)

# %% [markdown]
# ### 5.2.7 Saving the Model with Final Model Weights (ResNet50V2)

# %%
# Saving the final, optimized model file
model.save("/kaggle/working/optimized_skin_2_resnet50v2.keras")
print("\nSuccess! Your final model is saved as 'optimized_skin_2_resnet50v2.keras'.")
print("You can download it right now from the 'Output' section in the Kaggle right sidebar.")

# %% [markdown]
# # 6. Visualise Loss and Accuracy (ResNet50V2)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
ax1.plot(history.history["loss"],     label="Training Loss",   linewidth = 2)
ax1.plot(history.history["val_loss"], label="Validation Loss", linewidth = 2, linestyle = "--")
ax1.set_title("Loss over Epochs", fontweight="normal")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, alpha = 0.3)

# Accuracy curves
ax2.plot(history.history["accuracy"],     label = "Training Accuracy",   linewidth = 2)
ax2.plot(history.history["val_accuracy"], label = "Validation Accuracy", linewidth = 2, linestyle = "--")
ax2.set_title("Accuracy over Epochs", fontweight = "normal")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()
ax2.grid(True, alpha = 0.3)

plt.suptitle("Training History", fontsize = 14, fontweight = "normal")
plt.tight_layout()
plt.show()

# %% [markdown]
# # 7. Evaluate on the Test Set (ResNet50V2)

# %%
test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

# %% [markdown]
# # 8. Inference (ResNet50V2)

# %%
def run_inference_on_random_samples(model, dataset, class_names, num_samples=12, seed=42):
    """
    Picks a random selection of samples from the test set,
    runs them through the model, and displays the results
    in a grid showing the image, true label, predicted label,
    and the model's confidence score.
    """
    # ── Step 1: Collect all images and true labels from the test set ───────────
    all_images      = []
    all_true_labels = []

    for images, labels in dataset:
        for i in range(len(images)):
            all_images.append(images[i].numpy().astype("uint8"))
            all_true_labels.append(np.argmax(labels[i].numpy()))

    # ── Step 2: Randomly pick num_samples indices ──────────────────────────────
    random.seed(seed)
    selected_indices = random.sample(range(len(all_images)), min(num_samples, len(all_images)))

    selected_images      = [all_images[i]      for i in selected_indices]
    selected_true_labels = [all_true_labels[i] for i in selected_indices]

    # ── Step 3: Batch-predict all selected images in one forward pass ──────────
    batch = np.stack(selected_images, axis=0).astype("float32")  # Shape: (N, 256, 256, 3)
    predictions = model.predict(batch, verbose=0)                 # Shape: (N, num_classes)

    pred_indices    = np.argmax(predictions, axis=1)
    confidences     = np.max(predictions, axis=1) * 100

    # ── Step 4: Display results in a grid ─────────────────────────────────────
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4.2))
    axes = axes.flatten()

    for i in range(len(selected_images)):
        ax = axes[i]
        ax.imshow(selected_images[i])
        ax.axis("off")

        true_name = class_names[selected_true_labels[i]]
        pred_name = class_names[pred_indices[i]]
        confidence = confidences[i]
        is_correct = selected_true_labels[i] == pred_indices[i]

        # True label always in green
        ax.text(
            0.5, 1.10,
            f"True:  {true_name}",
            transform  = ax.transAxes,
            ha         = "center",
            va         = "bottom",
            fontsize   = 8.5,
            color      = "green",
            fontweight = "bold"
        )

        # Predicted label: green if correct, red if wrong
        ax.text(
            0.5, 1.03,
            f"Pred:  {pred_name}",
            transform  = ax.transAxes,
            ha         = "center",
            va         = "bottom",
            fontsize   = 8.5,
            color      = "green" if is_correct else "red",
            fontweight = "bold"
        )

        # Confidence score below the predicted label
        ax.text(
            0.5, -0.04,
            f"Confidence: {confidence:.1f}%",
            transform  = ax.transAxes,
            ha         = "center",
            va         = "top",
            fontsize   = 8,
            color      = "black"
        )

    # Hide unused subplot slots
    for j in range(len(selected_images), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"Inference on {len(selected_images)} Randomly Sampled Test Images",
        fontsize   = 13,
        fontweight = "normal",
        y          = 1.02
    )
    plt.tight_layout()
    plt.show()

    # ── Step 5: Print a quick summary ─────────────────────────────────────────
    correct = sum(
        1 for i in range(len(selected_images))
        if selected_true_labels[i] == pred_indices[i]
    )
    print(f"\nSample Accuracy : {correct}/{len(selected_images)} correct  ({correct/len(selected_images)*100:.1f}%)")


# ── Run inference ──────────────────────────────────────────────────────────────
run_inference_on_random_samples(
    model       = model,
    dataset     = test_dataset,
    class_names = allowed_classes,
    num_samples = 12,
    seed        = 42        # Change seed for a different random selection
)

# %% [markdown]
# # 9. Inspect Misclassfied Examples (ResNet50V2)

# %%
print("Collecting predictions from the test set to find misclassified examples...")

misclassified_images      = []
misclassified_true_labels = []
misclassified_pred_labels = []

for images, labels in test_dataset:
    preds = model.predict(images, verbose=0)

    true_indices = np.argmax(labels.numpy(), axis=1)
    pred_indices = np.argmax(preds, axis=1)

    # Identify positions where prediction did NOT match the true label
    wrong_mask = true_indices != pred_indices

    for i in np.where(wrong_mask)[0]:
        misclassified_images.append(images[i].numpy().astype("uint8"))
        misclassified_true_labels.append(true_indices[i])
        misclassified_pred_labels.append(pred_indices[i])

print(f"Total misclassified images found: {len(misclassified_images)}")


# ── Display up to 12 misclassified examples in a 3 × 4 grid ───────────────────
num_to_show = min(12, len(misclassified_images))

if num_to_show == 0:
    print("No misclassified images to display — the model got everything right!")
else:
    fig, axes = plt.subplots(3, 4, figsize=(18, 14))
    axes = axes.flatten()

    for i in range(num_to_show):
        ax = axes[i]
        ax.imshow(misclassified_images[i])
        ax.axis("off")

        true_name = allowed_classes[misclassified_true_labels[i]]
        pred_name = allowed_classes[misclassified_pred_labels[i]]

        # True label in green, predicted label in red
        ax.set_title(
            f"True:  {true_name}\nPred:  {pred_name}",
            fontsize  = 9,
            loc       = "center",
            color     = "black"          # base colour (overridden per-line below)
        )

        # Re-draw the title using two separate text objects for per-line colouring
        ax.set_title("")                 # clear the single-colour title first
        ax.text(
            0.5, 1.07,
            f"True:  {true_name}",
            transform  = ax.transAxes,
            ha         = "center",
            va         = "bottom",
            fontsize   = 9,
            color      = "green",
            fontweight = "bold"
        )
        ax.text(
            0.5, 1.00,
            f"Pred:  {pred_name}",
            transform  = ax.transAxes,
            ha         = "center",
            va         = "bottom",
            fontsize   = 9,
            color      = "red",
            fontweight = "bold"
        )

    # Hide any unused subplot slots
    for j in range(num_to_show, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"Misclassified Examples  ({num_to_show} of {len(misclassified_images)} shown)",
        fontsize   = 13,
        fontweight = "normal",
        y          = 1.02
    )
    plt.tight_layout()
    plt.show()


