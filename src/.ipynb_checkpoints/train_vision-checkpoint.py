
"""Module 3 -- Computer Vision (surface defect classification).

Fine-tunes a pretrained ResNet-50 on the NEU surface-defect images. The
convolutional backbone is frozen and only a small classification head is
trained -- with 1,800 images, training the full network would overfit badly.

Note on preprocessing: ResNet-50's preprocess_input is applied in the tf.data
pipeline, NOT inside the model. A Lambda layer wrapping it would save only the
function's name, and the model would then fail to load in a fresh process.
Keeping the model free of custom objects means the .keras file loads anywhere.

Run:  python src/train_vision.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

try:
    from src import config
except ImportError:
    import config


def load_datasets():
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.applications.resnet50 import preprocess_input

    train_dir = config.NEU_IMAGES / "train"
    val_dir = config.NEU_IMAGES / "val"
    if not train_dir.exists():
        raise SystemExit("Image folders not found. Run: python src/prepare_data.py")

    common = dict(
        labels="inferred",
        label_mode="int",
        class_names=config.DEFECT_CLASSES,
        color_mode="rgb",              # grayscale BMPs are expanded to 3 channels
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, shuffle=True, seed=config.RANDOM_STATE, **common
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, shuffle=False, **common
    )

    # Augment first (on raw 0-255 pixels), then preprocess for ResNet.
    augment = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.1),
        ],
        name="augment",
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(
        lambda x, y: (preprocess_input(augment(x, training=True)), y),
        num_parallel_calls=autotune,
    ).prefetch(autotune)
    val_ds = val_ds.map(
        lambda x, y: (preprocess_input(x), y), num_parallel_calls=autotune
    ).prefetch(autotune)

    return train_ds, val_ds


def build_model():
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import ResNet50

    base = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    )
    base.trainable = False  # frozen backbone -- we only train the head

    model = models.Sequential(
        [
            layers.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3)),
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(len(config.DEFECT_CLASSES), activation="softmax"),
        ],
        name="defect_classifier",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    print("=" * 60)
    print("MODULE 3 -- Surface Defect Classification")
    print("=" * 60)

    import mlflow
    from sklearn.metrics import classification_report, confusion_matrix

    train_ds, val_ds = load_datasets()

    model = build_model()
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    print(f"\ntrainable params: {trainable:,}")
    print(f"training for {config.EPOCHS} epochs (10-20 min on CPU)\n")

    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment("defect_classification")

    with mlflow.start_run(run_name="resnet50_frozen"):
        mlflow.log_params(
            {
                "backbone": "ResNet50 (ImageNet, frozen)",
                "img_size": config.IMG_SIZE,
                "batch_size": config.BATCH_SIZE,
                "epochs": config.EPOCHS,
            }
        )

        history = model.fit(train_ds, validation_data=val_ds, epochs=config.EPOCHS)

        y_true = np.concatenate([y.numpy() for _, y in val_ds])
        y_prob = model.predict(val_ds, verbose=0)
        y_pred = y_prob.argmax(axis=1)

        acc = float((y_pred == y_true).mean())
        mlflow.log_metric("val_accuracy", acc)
        print(f"\nValidation accuracy: {acc:.3f}")
        print(
            classification_report(
                y_true, y_pred,
                target_names=[config.DEFECT_NAMES[c] for c in config.DEFECT_CLASSES],
                zero_division=0,
            )
        )

    model.save(config.VISION_MODEL)
    with open(config.VISION_LABELS, "w") as fh:
        json.dump(
            {"classes": config.DEFECT_CLASSES, "names": config.DEFECT_NAMES,
             "val_accuracy": acc},
            fh, indent=2,
        )

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(
        cm,
        index=[config.DEFECT_NAMES[c] for c in config.DEFECT_CLASSES],
        columns=[config.DEFECT_NAMES[c] for c in config.DEFECT_CLASSES],
    ).to_csv(config.DATA_PROCESSED / "vision_confusion_matrix.csv")

    pd.DataFrame(history.history).to_csv(
        config.DATA_PROCESSED / "vision_history.csv", index=False
    )

    # Sanity check: reload from disk exactly as the Streamlit app will.
    import tensorflow as tf

    tf.keras.models.load_model(config.VISION_MODEL)
    print(f"\nSaved and reload-verified -> {config.VISION_MODEL.relative_to(config.ROOT)}")
    print("Done. Next: python src/build_rag.py\n")


if __name__ == "__main__":
    main()