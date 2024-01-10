import streamlit as st
import pandas as pd



@st.cache
def load_default_data():
    data = pd.read_csv("data/orbit_source.csv")
    columns = data.columns.to_list()
    return data


@st.cache
def load_prepared_data():
    data = pd.read_csv("data/orbit_prepared.csv")
    columns = data.columns.to_list()
    return data[columns[1:]]


@st.cache
def load_undersampled_data():
    data = pd.read_csv("data/orbit_iht.csv")
    columns = data.columns.to_list()
    return data[columns[1:]]

@st.cache
def load_oversampled_data():
    data = pd.read_csv("data/orbit_adasyn.csv")
    columns = data.columns.to_list()
    return data[columns[1:]]