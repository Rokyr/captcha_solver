import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def train_model(
    model,
    train_data,
    val_data,
    epochs=50,
    batch_size=64,
    model_path="best_captcha_model.h5",
):

    X_train, y_train = train_data
    X_val, y_val = val_data

    early_stop = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    )  # 30 ignores all stoppage
    checkpoint = ModelCheckpoint(
        model_path, monitor="val_loss", save_best_only=True, verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint],
    )
    return history


def plot_training_history(history, num_chars=5):

    import matplotlib.pyplot as plt

    # Plot loss curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy for each character head
    plt.subplot(1, 2, 2)
    for i in range(num_chars):
        train_acc_key = f"char_{i+1}_accuracy"
        val_acc_key = f"val_char_{i+1}_accuracy"
        if train_acc_key in history.history:
            plt.plot(
                history.history[train_acc_key], label=f"Train Acc Char {i+1}"
            )
            plt.plot(history.history[val_acc_key], label=f"Val Acc Char {i+1}")
    plt.title("Accuracy per Character Head")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()


def save_model(model, path):
    model.save(path)


def load_model(path):
    from tensorflow.keras.models import load_model

    return load_model(path)
