import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import pandas as pd

st.set_page_config(
    page_icon="⚒️",
    page_title="project_1"
)

st.header(":red[My machine learning classifier project]")
st.write("""These displays the various common classifier type
    \nThat have interact with through my studies course
         """)

name = st.sidebar.selectbox("Select Dataset", ['Iris Dataset', 'Breast_Cancer_Dataset', 'Wine_Dataset'])
classifier = st.sidebar.selectbox("Select Classifier", ["KNN", 'SVC', 'Random Forest Classifier'])

st.subheader(name)
st.write(f"#### Classifier:  {classifier}")


def get_dataset(name):
    if name == 'Iris Dataset':
        data = datasets.load_iris()
    elif name == 'Breast_Cancer_Dataset':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    return data


final_data = get_dataset(name)


def get_params(classifier):
    params = dict()
    if classifier == "KNN":
        K = st.sidebar.slider("n_neighbors", 1, 20)
        params["K"] = K
        L = st.sidebar.slider("leaf_size", 1, 30)
        params["leaf_size"] = L
    elif classifier == 'SVC':
        C = st.sidebar.slider("C", 0.001, 1.0)
        params['C'] = C
    else:
        n_estimator = st.sidebar.slider("n_estimators: ", 1, 100)
        params["n_estimators"] = n_estimator
        min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 10)
        params["min_samples_leaf"] = min_samples_leaf
        max_depth = st.sidebar.slider("max_depth", 1, 10)
        params["max_depth"] = max_depth
    return params


params = get_params(classifier)
st.write(params)


def get_cl_name(classifier, params):
    if classifier == 'KNN':
        cl_name = KNeighborsClassifier(n_neighbors=params["K"],
                                       leaf_size=params["K"])
    elif classifier == 'Random Forest Classifier':
        cl_name = RandomForestClassifier(n_estimators=params["n_estimators"],
                                         min_samples_leaf=params["min_samples_leaf"],
                                         max_depth=params["max_depth"], random_state=45)
    else:
        cl_name = SVC(C=params['C'])
    return cl_name


X = pd.DataFrame(final_data.data)
y = final_data.target
st.metric("**Length:**", value=len(X))
st.write("shape of dataset: ", X.shape)
st.write("number of classes:", len(np.unique(y)))
with st.expander("Show table"):
    st.table(X.head())

#fig=sns.pairplot(X, hue="")
#st.pyplot(fig)
pca = PCA(2)

X_projected = pca.fit_transform(X)
x1 = X_projected[:, 0]

x2 = X_projected[:, 1]
fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")

plt.colorbar()

st.pyplot(fig)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=45, test_size=0.3)
##fitting the data
model = get_cl_name(classifier, params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#####

from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import numpy as np

st.metric(" :red[Accuracy_score:]", accuracy_score(y_pred, y_test))
st.metric(" :red[mean_absolute_error:]", mean_absolute_error(y_pred, y_test))
st.metric(" :red[_error:]", np.sqrt(mean_squared_error(y_pred, y_test)).round(3))
