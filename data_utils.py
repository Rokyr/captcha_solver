import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import string


def load_dataset(dataset_dir, target_size=(50, 200)):
    images = []
    labels = []
    for i in os.listdir(dataset_dir):
        img_path = os.path.join(dataset_dir, i)
        # Load as grayscale
        img = load_img(
            img_path, target_size=target_size, color_mode="grayscale"
        )
        img_array = img_to_array(img) / 255.0
        images.append(img_array)

        base = os.path.splitext(i)[0]
        label = base
        labels.append(label)

    return np.array(images), labels


def encode_labels(labels, num_chars=5):
    # Collect all unique characters from labels
    char_list = sorted({c for label in labels for c in label})
    char_to_int = {c: i for i, c in enumerate(char_list)}
    num_classes = len(char_list)

    encoded = []
    for pos in range(num_chars):
        pos_labels = [char_to_int[label[pos]] for label in labels]
        encoded.append(to_categorical(pos_labels, num_classes=num_classes))
    return encoded, char_list


def load_and_preprocess_image(image_path, target_size=(50, 200)):
    img = load_img(image_path, target_size=target_size, color_mode="grayscale")
    img_array = img_to_array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)


def split_dataset(images, encoded_labels, test_size=0.1, val_size=0.1):
    num_samples = images.shape[0]
    indices = np.arange(num_samples)

    trainval_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=42
    )

    val_fraction = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=val_fraction, random_state=42
    )

    X_train = images[train_idx]
    X_val = images[val_idx]
    X_test = images[test_idx]

    y_train = [arr[train_idx] for arr in encoded_labels]
    y_val = [arr[val_idx] for arr in encoded_labels]
    y_test = [arr[test_idx] for arr in encoded_labels]

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
