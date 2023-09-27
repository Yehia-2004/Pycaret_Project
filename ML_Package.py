import pandas as pd
import numpy as np
import sqlite3 as sq
import sklearn.preprocessing as pr
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pycaret.classification as clf
import pycaret.regression as reg
import streamlit as st

st.title("Estimate with us")

class ML:

    def __init__(self) -> None:
        pass

    def load_data(self,filepath):
        file_extension = filepath.name.split('.')[-1]

        if file_extension == "csv":
            self.data = pd.read_csv(filepath)

        elif file_extension == "xlsx":
            self.data = pd.read_excel(filepath)

        elif file_extension == "parquet":
            self.data = pd.read_parquet(filepath)

        elif file_extension == "db":
            con = sq.connect(filepath)
            self.data = pd.read_sql(sql=f"SELECT * FROM {filepath}", con=con)

        elif file_extension == "html":
            dfs = pd.read_html(filepath)
            self.data = dfs[0]

        elif file_extension == "json":
            self.data = pd.read_json(filepath)

        else: st.write("This file is not supported")

    def eda(self):
        explore = st.sidebar.selectbox("What is you wanna know about your data Sir",[None,"Data Type","Shape","Columns","Number of missy values in each column","Statistical Analysis"])
        if explore == "Data Type":
            st.write(self.data.dtypes)

        elif explore == "Shape":
            st.write(self.data.shape)

        elif explore == "Columns":
            st.write(self.data.columns)

        elif explore == "Number of missy values in each column":
            st.write(self.data.isnull().sum())

        elif explore == "Statistical Analysis":
            st.write(self.data.describe())

    def preprocess(self):
       
        column = st.sidebar.multiselect("If you wanna drop any column(s), please select it below",self.data.columns)
        if column:
            self.data = self.data.drop(columns=column,errors="ignore")
            st.write(self.data.columns)
        numeric = self.data.select_dtypes(include='number')
        cat = self.data.select_dtypes(exclude='number')

        missy = st.sidebar.selectbox("How you wanna deal with Missing values, Sir",[None,"Drop Them","Fill Them"])
        if missy == "Drop Them":
            self.data = self.data.dropna()
            st.write(self.data.isnull().sum())
    
        if missy == "Fill Them":
            for column in self.data.select_dtypes(exclude= "number").columns:
                self.data[column] = SimpleImputer(strategy='most_frequent').fit_transform(self.data[column].values.reshape(-1,1))
            for num in self.data.select_dtypes(include = "number").columns:
                self.data[num] = self.data[num].fillna(self.data[num].mean())

            st.write(self.data.isnull().sum())
            
        scale = st.sidebar.selectbox("How you wanna scale your Data, Sir",[None,"Standard Scaling","Rubost Scaling","MinMax Scaling","MaxAbs Scaling"])
        
        if scale == "Standard Scaling":
            self.data[numeric.columns] = pr.StandardScaler().fit_transform(self.data[numeric.columns])

        elif scale == "Rubost Scaling":
            self.data[numeric.columns] = pr.RobustScaler().fit_transform(self.data[numeric.columns])

        elif scale == "MinMax Scaling":
            self.data[numeric.columns] = pr.MinMaxScaler().fit_transform(self.data[numeric.columns])

        elif scale == "MaxAbs Scaling":
            self.data[numeric.columns] = pr.MaxAbsScaler().fit_transform(self.data[numeric.columns])

        encode = st.sidebar.selectbox("How you wanna transform your Categorical column to Numeric, Sir",[None,"OneHot Encoding","Label Encoding","Ordinal Encoding"])
        
        if encode == "OneHot Encoding":
            self.data = pd.get_dummies(self.data, columns=cat.columns,drop_first=True)

        elif encode == "Label Encoding":
            self.data[cat.columns] = self.data[cat.columns].apply(pr.LabelEncoder().fit_transform)

        elif encode == "Ordinal Encoding":
            self.data[cat.columns] = pr.OrdinalEncoder().fit_transform(self.data[cat.columns])

        st.write(self.data.head())

    def auto(self):

        target = st.sidebar.selectbox("Choose you Target Column, Sir",self.data.columns.insert(0,None))

        if target in self.data.select_dtypes(include='number'):
            try:
                s = reg.setup(self.data, target=target)
                best = s.compare_models()
                st.write(best)
                st.write(s.evaluate_model(best))
            except ValueError:
                st.subheader("This Target isn't availble in prediction, please select another Target")

        if target in self.data.select_dtypes(exclude='number'):
            try:
                s = clf.setup(self.data, target=target)
                best = s.compare_models()
                st.write(best)
                st.write(s.evaluate_model(best))
            except ValueError:
                st.subheader("This Target isn't availble in prediction, please select another Target")

file_uploader = st.file_uploader(label="Upload your file, Sir")

if file_uploader is not None:
    ml = ML()
    ml.load_data(file_uploader)
    manual = st.sidebar.radio("Do you wanna make your EDA and Preprocessing manually?",("No","Yes"))
    if manual == "Yes":
        ml.eda()
        ml.preprocess()
    
    ml.auto()

else:
    pass