import numpy as np
import string

CHAR_LIST = list(string.ascii_lowercase) + list(string.digits)


def decode_predictions(preds):
    decoded = []
    num_samples = preds[0].shape[0]
    num_chars = len(preds)

    for i in range(num_samples):
        pred_str = ""
        for char_idx in range(num_chars):
            class_idx = np.argmax(preds[char_idx][i])
            pred_str += CHAR_LIST[class_idx]
        decoded.append(pred_str)

    return decoded


def evaluate_model(model, X_test, y_test, verbose=1):
    results = model.evaluate(X_test, y_test, verbose=verbose)

    num_chars = len(y_test)
    total_loss = results[0]
    losses = results[1 : 1 + num_chars]
    accuracies = results[1 + num_chars : 1 + 2 * num_chars]

    print(f"Total loss: {total_loss:.4f}")
    for i, (loss, acc) in enumerate(zip(losses, accuracies)):
        print(f"Char {i+1} - Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    preds = model.predict(X_test)

    decoded_preds = decode_predictions(preds)
    decoded_labels = decode_predictions(y_test)

    correct = sum(p == t for p, t in zip(decoded_preds, decoded_labels))
    captcha_accuracy = correct / len(decoded_preds)

    print(f"Overall CAPTCHA accuracy: {captcha_accuracy:.4f}")

    return {
        "total_loss": total_loss,
        "per_char_loss": losses,
        "per_char_accuracy": accuracies,
        "captcha_accuracy": captcha_accuracy,
    }
