import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_icon="💹",
    page_title="project_2"
)

st.title("Annual Salary prediction model based on :red[Age] and :red[Experience]")
st.markdown(">This is my first :blue[linear Regression Model]")
st.divider()

st.write("""### Fill the requierements to proceed""")
Age=st.number_input("Enter your Age: ", min_value=17, max_value=80, value=17)
Experience=st.number_input("Enter the number of experience on the job: ", max_value=27, value=0)

submitBtn=st.button("Predict the estimated Salary")

x=np.array([[Age, Experience]])
loaded_reg=joblib.load("Salary_reg.joblib")
if submitBtn:
    def add_commas(value):
        return "{:,}".format(value)
       
    st.balloons()
    pred=loaded_reg.predict(x)[0].round(3)
    formatted_salaries = np.array([add_commas(pred)])
    st.info(f'Predicted Annual Salary: {formatted_salaries[0]} /=')
