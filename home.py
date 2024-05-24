import streamlit as st 

st.set_page_config(page_title="Home",
                   layout='wide',
                   page_icon='./images/home.png')

st.title("YOLO V5 Object Detection App")
st.caption('This web application demostrate Object Detection')

# Content
st.markdown("""
### This App detects weapons from Images
- [Click here for Image Detection](/Image_detection/)  
- [Click here for Real Time Detection](/Real_time_detection/)  

the app detects 3 objects: 
1. gun - רובה ארוך
2. knife - סכין
3. pistol  - אקדח



""")