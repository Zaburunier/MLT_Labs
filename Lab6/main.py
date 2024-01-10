import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from data_loader import load_prepared_data
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.over_sampling import ADASYN


@st.cache
def split_data(data):
    columns = data.columns.to_list()
    return train_test_split(data[columns[:-1]], data[columns[-1]], train_size=0.7, random_state=6)


@st.cache
def get_classification_metrics(model, data):
    x_train, x_test, y_train, y_test = split_data(data)

    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    return {
        "Accuracy Score = " : accuracy_score(y_test, prediction),
        "Precision Score  = " : precision_score(y_test, prediction, average="macro"),
        "Recall Score = " : recall_score(y_test, prediction, average="macro"),
        "F1 Score = " : f1_score(y_test, prediction, average="macro")
    }



dataset = load_prepared_data()

if st.checkbox("Основные характеристики набора данных") == True:
    st.subheader("Структура")
    st.dataframe(dataset[:10])

    st.subheader("Число записей")
    labels_dict = {"Общее" : dataset.shape[0],}
    last_column = dataset.columns.to_list()[-1]
    for label in dataset[last_column].unique():
        labels_dict.update([(str(label), dataset[dataset[last_column] == label].shape[0])])
    st.write(labels_dict)

    st.subheader("Корреляционная матрица")
    fig, ax = plt.subplots(figsize=(25, 25))
    sns.set(font_scale=2)
    sns.heatmap(dataset.corr(), ax=ax, annot=True, fmt=".2f", linewidths=0.3, linecolor="black", vmin=-1, vmax=1)
    st.pyplot(fig)


if st.checkbox("Искусственное внесение данных в выборку") == True:
    st.write("Перед настройкой числа образцов рекомендуется посмотреть соотношение в исходных данных")
    generating_option = st.selectbox("Способ восстановления баланса", ["IHT (Undersampling)", "ADASYN (Oversampling)"])
    samples = dict()
    last_column = dataset.columns.to_list()[-1]

    dataset_x, dataset_y = (None, None)
    if generating_option == "IHT (Undersampling)":
        for label in dataset[last_column].unique():
            base_amount = dataset[dataset[last_column] == label].shape[0]
            samples.update([(label, st.slider(str(label), min_value=1, max_value=base_amount, value=base_amount)), ])
        dataset_x, dataset_y = InstanceHardnessThreshold(sampling_strategy=samples,
                                            random_state=16,
                                            n_jobs=6,
                                            cv = 5)\
                .fit_resample(dataset[dataset.columns.to_list()[:-1]],
                              dataset[dataset.columns.to_list()[-1]])
    else: # option == ADASYN
        for label in dataset[last_column].unique():
            base_amount = dataset[dataset[last_column] == label].shape[0]
            samples.update([(label, st.slider(str(label), min_value=base_amount, max_value=1500, value=base_amount)), ])
        dataset_x, dataset_y = ADASYN(sampling_strategy=samples,
                         random_state=16,
                         n_jobs=6,
                         n_neighbors=3) \
                .fit_resample(dataset[dataset.columns.to_list()[:-1]],
                              dataset[dataset.columns.to_list()[-1]])

    dataset = dataset_x.join(dataset_y)

st.header("Модель для обучения")
kernel_option = st.selectbox("Ядро модели", ["linear", "poly", "rbf", "sigmoid"])

st.header("Гипепараметры модели")
st.markdown("Параметр C")
c_option = st.slider("", min_value=0.1, max_value=1e+2)
st.markdown("Степень полинома (актуально только для ядра ***poly***)")
degree_option = st.slider("", min_value=2, max_value=8)
st.markdown("Параметр $$\gamma$$ (для ядер ***poly***, ***rbf*** и ***sigmoid***)")
gamma_option = st.slider("", min_value=0.001, max_value=1.)
st.markdown("Свободный член (*coef0*) (для ядер ***poly*** и ***sigmoid***)")
coef0_option = st.slider("", min_value=0., max_value=3.)

st.header("Показатели качества модели")
st.write(get_classification_metrics(SVC(kernel=kernel_option,
                                          C=c_option,
                                          degree=degree_option,
                                          gamma=gamma_option,
                                          coef0=coef0_option),
                                    dataset))
