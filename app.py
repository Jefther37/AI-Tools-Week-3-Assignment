import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.datasets import mnist

#  Helpers
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return tf.keras.models.load_model("mnist_cnn.keras")

def preprocess(img: Image.Image) -> np.ndarray:
    """Convert (H,W) PIL‑image → (1,28,28,1) float32/255."""
    img = img.resize((28, 28)).convert("L")       # gray 28×28
    arr = np.array(img).reshape(1, 28, 28, 1)
    return arr.astype("float32") / 255.0


st.set_page_config(page_title="MNIST CNN Demo", page_icon="🔢", layout="centered")
st.title("🔢 Hand‑written Digit Classifier")

model = load_model()

tab_draw, tab_upload, tab_sample = st.tabs(
    ["✏️ Draw a digit", "📤 Upload image", "🎲 Random test sample"]
)

with tab_draw:
    from streamlit_drawable_canvas import st_canvas

    st.caption("Use the left mouse button (or finger on mobile) to draw **one digit**. \
                Clear and draw again any time.")
    canvas = st_canvas(
        fill_color="white",
        stroke_color="white",
        background_color="black",
        stroke_width=12,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Predict", disabled=canvas.image_data is None):
        img = Image.fromarray(
            (255 - canvas.image_data[:, :, 0]).astype("uint8")
        )                                             # invert → white bg, black digit
        pred = model.predict(preprocess(img)).argmax()
        st.success(f"Prediction: **{pred}**")

with tab_upload:
    up = st.file_uploader("Choose PNG/JPG", type=["png", "jpg", "jpeg"])
    if up:
        img = Image.open(up)
        st.image(img, caption="Your upload", width=160)
        pred = model.predict(preprocess(img)).argmax()
        st.success(f"Prediction: **{pred}**")

with tab_sample:
    (_,_), (x_test, y_test) = mnist.load_data()
    i = st.slider("Test‑set index", 0, 9999, 0)
    img = Image.fromarray(x_test[i])
    st.image(img, caption=f"Ground truth: {y_test[i]}", width=160)
    pred = model.predict(preprocess(img)).argmax()
    st.success(f"Prediction: **{pred}**")

# Sidebar
with st.sidebar:
    st.header("Model")
    st.write("Trained for 5 epochs, Adam optimiser, cross‑entropy loss.")
    if st.checkbox("Show architecture"):
        model.summary(print_fn=lambda x: st.text(x))
