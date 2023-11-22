import streamlit as st
import subprocess

def open_python_file(file_path):
    subprocess.Popen(["python", file_path])

# Tiêu đề của ứng dụng
st.title("Xử Lý Ảnh")

# Nút button để mở file thuchanh.py
if st.button("Run Source"):
    open_python_file("utility/6_DigitalImageProcessing/ThucHanh.py")
