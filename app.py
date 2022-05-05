import streamlit as st
from SteamJeeraMasino_Page import show_steamJeeraMasino_page
from SonaMansuli_Page import show_SonaMansuli_page
from Mota_Page import show_Mota_page


page = st.sidebar.selectbox(
    "Select Rice Type", ("Steam Jeera Masino", "Sona Mansuli", "Mota"))

if page == "Steam Jeera Masino":
    show_steamJeeraMasino_page()
elif page == "Sona Mansuli":
    show_SonaMansuli_page()
else:
    show_Mota_page()
