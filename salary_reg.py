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

with st.expander("Explore"):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns 

    data=pd.read_csv("Employee_Salary_Dataset.csv")
    st.table(data.head())
    
    data0=data.groupby("Gender")['Salary'].sum()
    st.write(data0)
    

    st.subheader("Scatter_Chart of the Dataset")
    data_bar=data.groupby("Gender")['Salary'].sum()
    st.scatter_chart(data.head())
    data['Gender']=data['Gender'].replace("Male", 1)
    data['Gender']=data['Gender'].replace("Female", 0)

    corr_matrix=data.corr()
    st.subheader("Correlation Matrix Between variables")
    st.table(corr_matrix)

    df=data.drop("Gender", axis=1)
    df.head()
    st.metric("Maximum Age: ", df['Age'].max())
    st.metric("Minimum Age: ", df['Age'].min())
    st.subheader("Train Dataset Overview")
    fig=sns.pairplot(df, hue="Salary")
    st.pyplot(fig)

    from sklearn.model_selection import train_test_split
    X=df.drop("Salary", axis=1)
    y=df['Salary']
    X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=44, test_size=0.2)

    from sklearn.linear_model import LinearRegression
    lr=LinearRegression()
    lr.fit(X_train, y_train)
    y_pred=lr.predict(X_test)
    from sklearn.metrics import  mean_squared_error
    import numpy as np

    error=np.sqrt(mean_squared_error(y_pred, y_test))
    st.metric("Error:", error.round(4))

