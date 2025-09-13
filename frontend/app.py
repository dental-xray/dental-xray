import streamlit as st

# streamlit run frontend/app.py

# https://docs.ultralytics.com/modes/predict/#working-with-results


main_page = st.Page("main_page.py", title="Main Page")
page_2 = st.Page("page_2.py", title="Prediction")
page_3 = st.Page("page_3.py", title="Conditions")
# Set up navigation
pg = st.navigation([main_page, page_2, page_3])

# Run the selected page
pg.run()
