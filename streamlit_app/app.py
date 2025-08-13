import streamlit as st, torch, os, tempfile, pandas as pd
from streamlit_app.inference import load_model, predict

st.title("SignLink — 1D-CNN + Transformer (MVP)")

with st.sidebar:
    ckpt = st.text_input("Checkpoint path", "checkpoints/slt_best.pth")
    spm  = st.text_input("SentencePiece model", "artifacts/spm.model")
    pose_root = st.text_input("Pose data root", "datasets/training_data") # New input for pose root
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"Device: {device}")

tab1, tab2 = st.tabs(["Demo", "Evaluate"])

with tab1:
    st.header("Demo")
    file = st.file_uploader("Upload a video (mp4)", type=["mp4","mov","avi"])
    
    # New input for SENTENCE_NAME
    sentence_name_input = st.text_input("Enter SENTENCE_NAME for the uploaded video (e.g., _-adcxjm1R4_0-8-rgb_front)")

    if st.button("Load model"):
        st.session_state.model, st.session_state.tok = load_model(ckpt, spm, device=device)
        st.success("Model loaded")
    if file and "model" in st.session_state:
        if not sentence_name_input:
            st.warning("Please enter the SENTENCE_NAME for the uploaded video to find pose data.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(file.read()); path = tmp.name
            st.video(path)
            
            # Use provided SENTENCE_NAME for pose_dir construction
            # Assuming demo videos are from 'test' split for pose path construction
            pose_dir = os.path.join(pose_root, f"test_2D_keypoints/openpose_output/json/{sentence_name_input}")
            
            st.write(f"Debug: SENTENCE_NAME input: {sentence_name_input}")
            st.write(f"Debug: Constructed pose_dir: {pose_dir}")
            st.write(f"Debug: Pose directory exists: {os.path.isdir(pose_dir)}")

            pred = predict(path, pose_dir, st.session_state.model, st.session_state.tok, device=device) # Pass pose_dir
            st.subheader("Prediction")
            st.write(pred)

with tab2:
    st.header("Evaluate TSV")
    tsv = st.file_uploader("Upload manifest TSV", type=["tsv"])
    if tsv and "model" in st.session_state:
        df = pd.read_csv(tsv, sep="\t")
        names, gts, preds = [], [], []
        for _, row in df.iterrows():
            video_path = row["VIDEO_PATH"]
            sentence_name = row.get("SENTENCE_NAME","")
            # Construct pose_dir based on pose_root and SENTENCE_NAME
            # This assumes the manifest's VIDEO_PATH contains enough info to infer split (e.g., "train", "val", "test")
            # For simplicity, let's assume the manifest is for a single split (e.g., test.tsv)
            # A more robust solution might involve adding 'split' to the manifest or inferring it from video_path
            
            # For now, let's assume the manifest is for the 'test' split for pose path construction
            # This needs to be flexible if user uploads train/val manifests for evaluation
            # A better approach would be to have POSE_DIR directly in the manifest, which build_manifests.py already does.
            # So, we should use row["POSE_DIR"] directly if it exists.
            
            pose_dir = str(row["POSE_DIR"]) if "POSE_DIR" in row and pd.notna(row["POSE_DIR"]) else ""

            if not os.path.isfile(video_path):
                st.warning(f"Missing video file: {video_path}"); continue
            if pose_dir and not os.path.isdir(pose_dir):
                st.warning(f"Missing pose directory: {pose_dir}"); continue

            pred = predict(video_path, pose_dir, st.session_state.model, st.session_state.tok, device=device) # Pass pose_dir
            names.append(sentence_name); gts.append(row["SENTENCE"]); preds.append(pred)
        out = pd.DataFrame({"SENTENCE_NAME": names, "GT": gts, "PRED": preds})
        st.dataframe(out)
        st.download_button("Download CSV", out.to_csv(index=False).encode("utf-8"), "preds.csv", "text/csv")
