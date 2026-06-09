0import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skin Disease Classifier",
    page_icon="🔬",
    layout="wide",
)

# ── TensorFlow import ──────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    TF_OK = True
except Exception as e:
    TF_OK = False
    TF_ERROR = str(e)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH = "optimized_skin_2_resnet50v2.keras"

CLASS_NAMES = [
    'Acne', 'Actinic_Keratosis', 'Benign_tumors', 'Bullous', 'Candidiasis',
    'DrugEruption', 'Eczema', 'Infestations_Bites', 'Lichen', 'Lupus',
    'Moles', 'Psoriasis', 'Rosacea', 'Seborrh_Keratoses', 'SkinCancer',
    'Sun_Sunlight_Damage', 'Tinea', 'Unknown_Normal', 'Vascular_Tumors',
    'Vasculitis', 'Vitiligo', 'Warts'
]

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("ℹ️ About This App")
    st.divider()
    st.markdown("#### What this app does")
    st.success("✅ Loads a ResNet50V2 model trained on 22 skin conditions")
    st.caption("Group 29 — Skin Disease Detection Project")
    st.success("✅ Accepts image uploads (JPG, PNG, WEBP)")
    st.success("✅ Predicts the correct skin condition")
    st.success("✅ Shows confidence score with a High / Moderate / Low tier")
    st.success("✅ Plots the top 8 class probabilities as a bar chart")
    st.success("✅ Generates Grad-CAM heatmap showing model focus area")
    st.divider()
    st.markdown("#### 22 Skin Conditions")
    for name in CLASS_NAMES:
        st.markdown(f"- {name.replace('_', ' ')}")
    st.divider()
    st.warning("⚠️ For educational purposes only. Always consult a qualified dermatologist.")

# ══════════════════════════════════════════════════════════════════════════════
#  TITLE
# ══════════════════════════════════════════════════════════════════════════════
# ── Header: logo + title side by side ─────────────────────────────────────────
col_logo, col_title = st.columns([0.12, 0.88], gap="small")

with col_logo:
    if os.path.exists("skin disease logo.png"):
        st.image("skin disease logo.png", width=90)

with col_title:
    st.markdown("## 🔬 Skin Disease Classifier (Group 29)")
    st.caption("ResNet50V2  ·  22 skin conditions  ·  Upload an image for instant analysis")

st.divider()

# ── Guard: TensorFlow ──────────────────────────────────────────────────────────
if not TF_OK:
    st.error("**TensorFlow failed to import.**")
    st.code(TF_ERROR)
    st.info("In your activated environment run:  `pip install tensorflow`  then restart the app.")
    st.stop()

# ── Guard: model file ──────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    st.error(f"**Model file not found:** `{MODEL_PATH}`")
    st.info(
        f"The app is looking in: **`{os.getcwd()}`**\n\n"
        f"Copy `{MODEL_PATH}` into that folder, then refresh the browser."
    )
    st.stop()

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def load_model():
    from tensorflow.keras.applications.resnet_v2 import preprocess_input
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": preprocess_input}
    )

try:
    model = load_model()
except Exception as e:
    st.error("**Model failed to load.**")
    st.code(str(e))
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def predict(pil_image):
    img   = pil_image.convert("RGB").resize((256, 256))
    arr   = np.array(img, dtype="float32")
    arr   = np.expand_dims(arr, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]) * 100, probs, idx


def make_gradcam(pil_image, pred_index):
    """
    Generates a Grad-CAM heatmap overlay using GradientTape directly
    on the full model — no sub-model needed, avoids "never been called" error.
    """
    img = pil_image.convert("RGB").resize((256, 256))
    arr = np.array(img, dtype="float32")
    arr = np.expand_dims(arr, axis=0)                      # (1, 256, 256, 3)

    # Find the ResNet50V2 Functional sub-model by type
    resnet_base = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            resnet_base = layer
            break
    if resnet_base is None:
        raise ValueError("Could not find ResNet base inside the model.")

    # Build a model: input -> last conv layer output + full model output
    # Call the model once first to ensure all outputs are defined
    _ = model(arr, training=False)
    last_conv_layer  = resnet_base.get_layer("post_bn")

    grad_model = tf.keras.models.Model(
        inputs  = resnet_base.inputs,
        outputs = [last_conv_layer.output, resnet_base.output]
    )

    # Pass image through Lambda + resnet preprocessing manually
    from tensorflow.keras.applications.resnet_v2 import preprocess_input as rn_preprocess
    arr_preprocessed = rn_preprocess(arr.copy())

    with tf.GradientTape() as tape:
        conv_outputs, _ = grad_model(arr_preprocessed)
        tape.watch(conv_outputs)
        # Pass conv_outputs through remaining post-resnet layers
        x = conv_outputs
        found_resnet = False
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                found_resnet = True
                continue
            if found_resnet:
                x = layer(x, training=False)
        loss = x[:, pred_index]

    grads        = tape.gradient(loss, conv_outputs)       # (1, H, W, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # (C,)
    conv_outputs = conv_outputs[0]                         # (H, W, C)
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]  # (H, W, 1)
    heatmap      = tf.squeeze(heatmap).numpy()             # (H, W)

    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    heatmap_resized = np.array(
        Image.fromarray(np.uint8(heatmap * 255)).resize((256, 256), Image.LANCZOS)
    ) / 255.0

    orig = np.array(img) / 255.0

    # ── Smooth color gradient: blue (low) → cyan → green → yellow → red (high)
    colormap = plt.cm.RdYlGn_r(heatmap_resized)[:, :, :3]   # red=hot, green=cold
    overlay  = np.clip(0.5 * orig + 0.5 * colormap, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(orig)
    axes[0].set_title("Original", fontsize=9, fontweight="bold")
    axes[0].axis("off")

    im = axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=9, fontweight="bold")
    axes[1].axis("off")

    # ── Colorbar legend showing gradient scale ─────────────────────────────────
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.RdYlGn_r,
        norm=plt.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1], fraction=0.035, pad=0.04)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Mid", "High"], fontsize=7)
    cbar.set_label("Activation", fontsize=7)

    plt.suptitle("🔍 Region the model focused on", fontsize=10, fontweight="bold")
    plt.tight_layout()
    return fig


def confidence_chart(probs, top_n=8):
    top_idx = np.argsort(probs)[::-1][:top_n]
    names   = [CLASS_NAMES[i] for i in top_idx][::-1]
    values  = [float(probs[i]) * 100 for i in top_idx][::-1]
    colors  = ["#4CAF50" if i == top_n - 1 else "#90CAF9" for i in range(top_n)]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(names, values, color=colors, height=0.55, edgecolor="none")

    for j, val in enumerate(values):
        ax.text(val + 0.3, j, f"{val:.1f}%", va="center", ha="left", fontsize=8)

    ax.set_xlim(0, max(values) * 1.22)
    ax.set_xlabel("Confidence (%)", fontsize=8)
    ax.set_title(f"Top {top_n} Predictions", fontsize=9, fontweight="bold", loc="left")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════════════════════
uploaded = st.file_uploader(
    "Upload a skin image (JPG, PNG or WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded is None:
    st.info("👆 Upload an image above to begin analysis.")
    st.stop()

image = Image.open(uploaded)

# ── Run inference first so pred_index is available for Grad-CAM ───────────────
with st.spinner("Running inference…"):
    try:
        pred_class, confidence, probs, pred_index = predict(image)
    except Exception as e:
        st.error("**Inference failed.**")
        st.code(str(e))
        st.stop()

col_img, col_results = st.columns([1, 1.5], gap="large")

# ── Left column: uploaded image + Grad-CAM ────────────────────────────────────
with col_img:
    st.image(image, caption="Uploaded image", use_container_width=True)

    st.markdown("#### 🔍 Grad-CAM — Where the model looked")
    with st.spinner("Generating Grad-CAM heatmap…"):
        try:
            gradcam_fig = make_gradcam(image, pred_index)
            st.pyplot(gradcam_fig, use_container_width=True)
            plt.close(gradcam_fig)
        except Exception as e:
            st.warning(f"Grad-CAM could not be generated: {e}")

# ── Right column: prediction results + confidence chart ───────────────────────
with col_results:
    if confidence >= 70:
        tier, color = "High confidence", "green"
    elif confidence >= 45:
        tier, color = "Moderate confidence", "orange"
    else:
        tier, color = "Low confidence — interpret with caution", "red"

    st.subheader("Predicted Condition")
    st.markdown(f"### {pred_class.replace('_', ' ')}")
    st.metric("Model Confidence", f"{confidence:.1f}%")
    st.markdown(f":{color}[● {tier}]")
    st.divider()

    st.subheader("Confidence Distribution")
    fig = confidence_chart(probs, top_n=8)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.divider()
st.warning(
    "⚠️ **Medical Disclaimer:** This tool is for educational purposes only. "
    "It is not a substitute for professional medical advice or diagnosis. "
    "Always consult a qualified dermatologist."
)

st.markdown(
    """
    <div style="text-align:center; margin-top:2rem; padding-top:1rem;
                border-top:1px solid #e0e0e0;">
        <p style="color:#aaaaaa; font-size:0.72rem; margin:0;">
            Developed by &nbsp;·&nbsp;
            Tajudeen Kazeem &nbsp;·&nbsp; Oluwakayode Alabi &nbsp;·&nbsp;
            Oladele Paul &nbsp;·&nbsp; Peter Olajide &nbsp;·&nbsp; Adunola Amole Oladele
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
