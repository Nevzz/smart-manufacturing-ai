import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from src import config

st.set_page_config(page_title="Defect Detection", page_icon="🔍", layout="wide")
st.title("🔍 Surface Defect Detection")
st.caption("ResNet-50 (ImageNet pretrained) fine-tuned on the NEU surface defect dataset")

if not config.VISION_MODEL.exists():
    st.warning("Defect classifier not trained yet. Run `python src/train_vision.py` first.")
    st.stop()


@st.cache_resource
def load_model():
    import tensorflow as tf

    model = tf.keras.models.load_model(config.VISION_MODEL)
    meta = json.loads(config.VISION_LABELS.read_text())
    return model, meta


def predict(model, image: Image.Image):
    # preprocess_input lives here rather than inside the model, so the saved
    # model stays free of custom objects and loads cleanly.
    from tensorflow.keras.applications.resnet50 import preprocess_input

    img = image.convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE))
    arr = np.expand_dims(np.array(img, dtype="float32"), 0)
    probs = model.predict(preprocess_input(arr), verbose=0)[0]
    return probs


model, meta = load_model()
val_acc = meta.get("val_accuracy")

tab1, tab2 = st.tabs(["Classify an image", "Model performance"])

# ------------------------------------------------------------------ tab 1
with tab1:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Input")
        uploaded = st.file_uploader(
            "Upload a steel surface image", type=["bmp", "png", "jpg", "jpeg"]
        )

        image = None
        true_label = None

        if uploaded is not None:
            image = Image.open(uploaded)
        elif (config.NEU_IMAGES / "val").exists():
            st.caption("No upload yet — try a random image from the validation set.")
            if st.button("Pick a random validation image"):
                cls = random.choice(config.DEFECT_CLASSES)
                files = list((config.NEU_IMAGES / "val" / cls).glob("*.bmp"))
                if files:
                    chosen = random.choice(files)
                    st.session_state["sample_path"] = str(chosen)
                    st.session_state["sample_true"] = cls

            if "sample_path" in st.session_state:
                image = Image.open(st.session_state["sample_path"])
                true_label = st.session_state["sample_true"]

        if image is not None:
            st.image(image, caption="Input image", width=320)

    with right:
        st.subheader("Prediction")
        if image is None:
            st.info("Upload an image or pick a random validation image.")
        else:
            probs = predict(model, image)
            idx = int(np.argmax(probs))
            code = config.DEFECT_CLASSES[idx]
            name = config.DEFECT_NAMES[code]
            conf = float(probs[idx])

            st.metric("Predicted defect", name, f"{conf:.1%} confidence")

            if true_label is not None:
                if true_label == code:
                    st.success(f"Correct — actual label is {config.DEFECT_NAMES[true_label]}.")
                else:
                    st.error(f"Incorrect — actual label is {config.DEFECT_NAMES[true_label]}.")

            chart_df = pd.DataFrame({
                "Defect": [config.DEFECT_NAMES[c] for c in config.DEFECT_CLASSES],
                "Probability": probs,
            }).sort_values("Probability")

            fig = px.bar(chart_df, x="Probability", y="Defect", orientation="h",
                         height=330, title="Class probabilities")
            fig.update_layout(xaxis_tickformat=".0%", yaxis_title="", xaxis_title="")
            st.plotly_chart(fig, width="stretch")

            if conf < 0.6:
                st.warning(
                    "Low confidence. Route this coil to manual inspection rather "
                    "than acting on the prediction."
                )

    st.divider()
    st.markdown("**Defect classes in this dataset**")
    st.markdown(
        " · ".join(f"**{config.DEFECT_NAMES[c]}** ({c})" for c in config.DEFECT_CLASSES)
    )

# ------------------------------------------------------------------ tab 2
with tab2:
    if val_acc is not None:
        st.metric("Validation accuracy", f"{val_acc:.1%}", help="360 held-out images, 60 per class")

    cm_path = config.DATA_PROCESSED / "vision_confusion_matrix.csv"
    if cm_path.exists():
        cm = pd.read_csv(cm_path, index_col=0)
        fig = px.imshow(
            cm.values, x=cm.columns, y=cm.index, text_auto=True,
            color_continuous_scale="Blues", height=520,
            labels=dict(x="Predicted", y="Actual", color="Images"),
            title="Confusion matrix (validation set)",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Off-diagonal cells show which defects the model confuses. Crazing and "
            "rolled-in scale are the usual pair, since both appear as dark irregular "
            "texture at this resolution."
        )

    hist_path = config.DATA_PROCESSED / "vision_history.csv"
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        hist["epoch"] = hist.index + 1
        acc_cols = [c for c in hist.columns if "accuracy" in c]
        fig = px.line(hist, x="epoch", y=acc_cols, markers=True,
                      title="Training history", height=380)
        fig.update_layout(yaxis_title="Accuracy", legend_title="")
        st.plotly_chart(fig, width="stretch")

    st.markdown(
        """
**Why the backbone is frozen.** With 1,800 images across six classes, training
all 25 million ResNet-50 parameters would memorise the training set. Freezing
the convolutional layers and training only the classification head keeps the
trainable parameter count small and the model honest.
"""
    )