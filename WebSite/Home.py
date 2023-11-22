import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Trang Chủ",
    page_icon="👋",
)

logo = Image.open('images\logo.png')
st.image(logo, width=800)

st.markdown(
    """
    ### Website Xử Lý Ảnh Số
    - Thực hiện bởi: Ngô Quang Nghĩa và Trần Thanh Hiếu
    - Giảng viên hướng dẫn: ThS. Trần Tiến Đức
    - Lớp Xử Lý Ảnh Số nhóm 01: DIPR430685_23_1_01
    """
)

st.markdown("""### Thành viên thực hiện""")
left, right = st.columns(2)
with left: 
    st.image(Image.open('images\member_nghia.jpg'), "Ngô Quang Nghĩa, 21110559", width=350)
with right:
    st.image(Image.open('images\member_hieu.jpg'), "Trần Thanh Hiếu, 21110448", width=350)

st.markdown(
    """
    ### Thông tin liên hệ
    - Facebook: https://www.facebook.com/quangnghia559/ hoặc https://www.facebook.com/TTHieu29
    - Email: 21110559@student.hcmute.edu.vn hoặc 21110448@student.hcmute.edu.vn
    - Lấy source code tại [đây](https://github.com/quangnghia1110/XuLyAnhSo.git)
    """
)

st.markdown("""### Video giới thiệu về Website""")
st.markdown("""[Video giới thiệu website Xử Lý Ảnh Số]()""")