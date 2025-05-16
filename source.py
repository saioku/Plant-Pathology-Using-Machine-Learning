import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# random seed for replicability
tf.random.set_seed(42)
np.random.seed(42)

preprocessed_dir = 'preprocessed_images'
os.makedirs(preprocessed_dir, exist_ok=True)

# DATA PREPARATION

df = pd.read_csv('train.csv')

df['image_id'] = df['image_id'].astype(str) + '.jpg'

# labels to binary (0 or 1)
label_cols = ['healthy', 'multiple_diseases', 'rust', 'scab']
for col in label_cols:
    df[col] = (df[col] > 0).astype(int)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# DATA GENERATION 

train_dir = 'images'  # original images folder
target_size = (256, 256)
batch_size = 32

def save_preprocessed_images(generator, subset_name):
    generator.reset()
    os.makedirs(os.path.join(preprocessed_dir, subset_name), exist_ok=True)
    
    for i in range(len(generator)):
        x_batch, y_batch = next(generator)
        for j in range(len(x_batch)):
            img_id = generator.filenames[generator.index_array[i * generator.batch_size + j]]
            img_array = x_batch[j]
            img_path = os.path.join(preprocessed_dir, subset_name, img_id)
            tf.keras.preprocessing.image.save_img(img_path, img_array)

# augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=lambda x: x 
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=lambda x: x 
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col='image_id',
    y_col=label_cols,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='raw',
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=train_dir,
    x_col='image_id',
    y_col=label_cols,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='raw',
    shuffle=False
)

save_preprocessed_images(train_generator, 'train')
save_preprocessed_images(val_generator, 'val')

train_generator.reset()
val_generator.reset()

# MODEL ARCHITECTURE 

def create_model(input_shape=(256, 256, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='sigmoid')  # sigmoid for multi-label classification
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    
    return model

model = create_model()
model.summary()

# TRAINING
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')
]

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=30,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=callbacks
)

#EVALUATION

def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)

results = model.evaluate(val_generator)
print(f"Validation Loss: {results[0]:.4f}")
print(f"Validation Accuracy: {results[1]:.4f}")
print(f"Validation AUC: {results[2]:.4f}")

def visualize_predictions(generator, model, num_samples=8, max_batches=5):
    generator.reset()
    class_names = ['healthy', 'multiple_diseases', 'rust', 'scab']
    
    correct_samples = []
    incorrect_samples = []
    
    batches_checked = 0
    
    while (len(correct_samples) < num_samples//2 or len(incorrect_samples) < num_samples//2) and batches_checked < max_batches:
        x_batch, y_true = next(generator)
        y_pred = model.predict(x_batch, verbose=0)
        
        for i in range(len(x_batch)):
            correct = np.array_equal((y_true[i] > 0.5), (y_pred[i] > 0.5))
            true_labels = [class_names[j] for j in range(4) if y_true[i][j] > 0.5]
            pred_labels = [class_names[j] for j in range(4) if y_pred[i][j] > 0.5]
            
            sample_data = {
                'image': x_batch[i],
                'true_labels': true_labels,
                'pred_labels': pred_labels,
                'correct': correct
            }
            
            if correct and len(correct_samples) < num_samples//2:
                correct_samples.append(sample_data)
            elif not correct and len(incorrect_samples) < num_samples//2:
                incorrect_samples.append(sample_data)
            
            if len(correct_samples) >= num_samples//2 and len(incorrect_samples) >= num_samples//2:
                break
        
        batches_checked += 1
    
    # combine samples (alternating correct and incorrect)
    all_samples = []
    for i in range(max(len(correct_samples), len(incorrect_samples))):
        if i < len(incorrect_samples):
            all_samples.append(incorrect_samples[i])
        if i < len(correct_samples):
            all_samples.append(correct_samples[i])
    
    plt.figure(figsize=(15, 15))
    for i, sample in enumerate(all_samples[:num_samples]):
        plt.subplot((num_samples+1)//2, 2, i+1)
        plt.imshow(sample['image'])
        
        title = f"True: {', '.join(sample['true_labels']) if sample['true_labels'] else 'none'}\n"
        title += f"Pred: {', '.join(sample['pred_labels']) if sample['pred_labels'] else 'none'}\n"
        title += "CORRECT" if sample['correct'] else "INCORRECT"
        
        plt.title(title, fontsize=10, color='green' if sample['correct'] else 'red')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_predictions(val_generator, model, num_samples=8)