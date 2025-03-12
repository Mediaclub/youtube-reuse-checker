import cv2
import numpy as np
import librosa
import streamlit as st
from pytube import YouTube
from moviepy.video.io.VideoFileClip import VideoFileClip
import os

def download_youtube_video(youtube_url):
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(file_extension="mp4").first()
    video_path = stream.download()
    return video_path

# Function to detect repetitive frames
def detect_repetitive_frames(video_path, sample_rate=300, threshold=0.98):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

# Streamlit Web App
def main():
    st.title("YouTube Reused Content Checker")
    youtube_url = st.text_input("Enter YouTube Video URL")
    
    if youtube_url:
        st.write("Downloading video... This may take a while.")
        video_path = download_youtube_video(youtube_url)
        
        st.write("Analyzing for repeated content...")
        
        # Extract audio
        video = VideoFileClip(video_path)
        audio_path = video_path.replace(".mp4", ".wav")
        video.audio.write_audiofile(audio_path)
        
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

        # Clean up downloaded files
        os.remove(video_path)
        os.remove(audio_path)

if __name__ == "__main__":
    main()
