"""SEM nanostructure classifier: demo (SPEC section 11, demo entry 2026-10-07).

  streamlit run app/streamlit_app.py

Uploads are processed in memory only: nothing is written to disk or logged."""
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from semcls import infer as I  # noqa: E402
from semcls.fetch import DownloadError, ensure_checkpoint  # noqa: E402

CKPT = Path(os.environ.get("SEM_CHECKPOINT", ROOT / "data" / "models" / "convnext_tiny_group094_s2_best.pt"))
SAMPLES = ROOT / "app" / "samples"

st.set_page_config(page_title="SEM nanostructure classifier", layout="wide")


@st.cache_resource(show_spinner="Loading model (the first start downloads 111 MB, checked against a recorded sha256)...")
def load_model():
    return I.Classifier.load(ensure_checkpoint(CKPT, I.CHECKPOINT_SHA256))


def overlay(image, cam, strength=0.6):
    v = np.clip(cam, 0, 1)[..., None]
    jet = np.stack([np.clip(1.5 - np.abs(4 * v[..., 0] - 3), 0, 1),
                    np.clip(1.5 - np.abs(4 * v[..., 0] - 2), 0, 1),
                    np.clip(1.5 - np.abs(4 * v[..., 0] - 1), 0, 1)], axis=-1) * 255
    a = strength * v
    return ((1 - a) * image + a * jet).astype(np.uint8)


def sample_files():
    return sorted(SAMPLES.rglob("*.jpg")) if SAMPLES.exists() else []


st.title("SEM nanostructure classifier")
st.caption("ConvNeXt-Tiny fine-tuned on the NFFA-EUROPE SEM dataset (10 classes), evaluated with a leakage-free split. "
           "Research demo, not a measuring instrument.")

try:
    clf = load_model()
except (I.ChecksumError, DownloadError) as e:
    st.error(f"Model could not be loaded: {e}")
    st.stop()

with st.sidebar:
    st.subheader("Model")
    st.write("ConvNeXt-Tiny, group split tau = 0.94, seed 2")
    st.write(f"checkpoint sha256: `{I.CHECKPOINT_SHA256[:12]}...`")
    st.write(f"temperature T = {I.TEMPERATURE}")
    st.write(f"\"uncertain\" below confidence {I.THETA:.4f}")
    show_cam = st.checkbox("Show Grad-CAM", value=True)

data, true_label = None, None
tab_example, tab_upload = st.tabs(["Example image (validation split)", "Upload your own"])
with tab_example:
    files = sample_files()
    if files:
        choice = st.selectbox("Image", files, format_func=lambda p: f"{p.parent.name} / {p.stem[:14]}")
        if st.button("Classify the example", type="primary"):
            data, true_label = choice.read_bytes(), choice.parent.name
    else:
        st.info("No example images found in app/samples.")
with tab_upload:
    up = st.file_uploader("PNG or JPEG, up to 10 MB, up to 4096 x 4096 pixels", type=["png", "jpg", "jpeg"])
    if up is not None and st.button("Classify the upload", type="primary"):
        data, true_label = up.getvalue(), None

if data is not None:
    try:
        r = clf.predict(data, top_k=3, with_cam=show_cam)
    except I.UploadError as e:
        st.error(f"Upload not accepted: {e}")
        st.stop()
    left, right = st.columns([3, 2])
    with left:
        if show_cam and r.cam is not None:
            c1, c2 = st.columns(2)
            c1.image(r.image, caption="Image as seen by the model (512 x 384)", width="stretch")
            c2.image(overlay(r.image, r.cam), caption="Grad-CAM of the predicted class (illustrative)", width="stretch")
        else:
            st.image(r.image, caption="Image as seen by the model (512 x 384)", width="stretch")
    with right:
        if r.uncertain:
            st.warning(f"Uncertain: top confidence {r.confidence:.1%} is below the threshold {I.THETA:.1%}. "
                       "Treat the label as a suggestion.")
        else:
            st.success(f"Prediction: {r.top[0][0]} ({r.confidence:.1%})")
        for name, p in r.top:
            st.write(f"{name}")
            st.progress(min(max(p, 0.0), 1.0), text=f"{p:.1%}")
        if true_label is not None:
            st.caption(f"Label in the dataset: {true_label}")

with st.expander("What this demo can and cannot do", expanded=False):
    st.markdown(
        "- The model knows only the 10 classes of the NFFA-EUROPE SEM dataset and returns one of them for **any** image, "
        "including photographs or other microscopy. There is no out-of-distribution detector: low confidence is not a "
        "reliable alarm for such inputs.\n"
        "- Probabilities are temperature-scaled (T fitted on the validation split); the \"uncertain\" flag follows the "
        "selective-prediction rule fixed before the test analysis. On the test split about 94% of images are accepted "
        "with ~99% accuracy for this model family; the rest are flagged. Rare classes (Films_Coated_Surface, Porous_Sponge, "
        "Fibres) are flagged much more often (about 25-45%).\n"
        "- Overall test macro-F1 of the 3-seed mean is 0.956 (ConvNeXt-Tiny, group-aware split); individual seeds differ by "
        "about 1 point and the rare classes have only 22-45 test images, so their numbers are uncertain.\n"
        "- Uploads are resized to 512 x 384 with the same code as in evaluation, but the training images were resized from "
        "1024 x 768 originals by a separate cache step, so results on raw uploads can differ slightly from the reported ones.\n"
        "- Grad-CAM shows where the gradient of the predicted score is large; it is an illustration, not proof of what the "
        "model uses. The information bar at the bottom of SEM images was shown not to be necessary for accuracy (small "
        "possible effect of about 1 point in rare classes).\n"
        "- Uploads are processed in memory only; nothing is saved, logged or used for training.")
    st.caption("Data: NFFA-EUROPE 100% SEM Dataset (Aversa et al., Scientific Data 2018). Example images come from the validation split only.")
