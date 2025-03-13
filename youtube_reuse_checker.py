import cv2
import numpy as np
import librosa
import streamlit as st
from moviepy.editor import VideoFileClip
import os

# Streamlit Web App
st.title("YouTube Reused Content Checker (Upload Version)")
uploaded_file = st.file_uploader("Upload your video file", type=["mp4", "mov", "avi"])

if uploaded_file:
    st.write("Processing video...")

    # Save uploaded file temporarily
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    # Extract audio
    video = VideoFileClip(video_path)
    audio_path = "uploaded_audio.wav"
    video.audio.write_audiofile(audio_path)

    # Function to detect repetitive frames
    def detect_repetitive_frames(video_path, sample_rate=300, threshold=0.98):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count == 0:
            st.error("Error: Could not read the video file.")
            return 0
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        prev_frame = None
        duplicate_count = 0
        total_samples = frame_count // sample_rate

        for i in range(0, frame_count, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                similarity = np.sum(gray_frame == prev_frame) / gray_frame.size
                if similarity > threshold:
                    duplicate_count += 1

            prev_frame = gray_frame

        cap.release()
        repetition_percentage = (duplicate_count / total_samples) * 100
        return repetition_percentage

    # Function to detect repetitive audio segments
    def detect_repetitive_audio(audio_path, sample_rate=10, threshold=0.95):
        y, sr = librosa.load(audio_path, sr=None)
        chunk_size = sr * sample_rate  # Analyzes every 10 seconds
        chunks = [y[i:i+chunk_size] for i in range(0, len(y), chunk_size)]
        
        duplicate_count = 0
        total_chunks = len(chunks)
        
        for i in range(1, total_chunks):
            similarity = np.corrcoef(chunks[i], chunks[i-1])[0,1]
            if similarity > threshold:
                duplicate_count += 1

        repetition_percentage = (duplicate_count / total_chunks) * 100
        return repetition_percentage

    # Run analysis
    video_repetition = detect_repetitive_frames(video_path)
    audio_repetition = detect_repetitive_audio(audio_path)

    # Display results
    st.write(f"### Video Repetition Percentage: {video_repetition:.2f}%")
    st.write(f"### Audio Repetition Percentage: {audio_repetition:.2f}%")

    if video_repetition > 80 or audio_repetition > 80:
        st.write("⚠️ Warning: High repetition detected. YouTube may flag this as reused content.")
    else:
        st.write("✅ Safe: Low repetition detected. This should be safe for monetization.")

    # Clean up files
    os.remove(video_path)
    os.remove(audio_path)
