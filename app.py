import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import hstack
import os

import warnings
warnings.filterwarnings('ignore')

# changing page main title and main icon(logo)
PAGE_CONFIG = {"page_title":"Personalized Cancer Diagnosis", "page_icon":":cancer:", "layout":"centered"}
st.set_page_config(**PAGE_CONFIG)   

st.sidebar.text("Created on Sat, Jan 30 2021")
st.sidebar.markdown("**@author:Sumit Kumar** :monkey_face:")
st.sidebar.markdown("[My Github](https://github.com/IMsumitkumar) :penguin:")
st.sidebar.markdown("[findingdata.ml](https://www.findingdata.ml/) :spider_web:")
st.sidebar.markdown("[Data & Description](https://www.kaggle.com/c/msk-redefining-cancer-treatment) :house:")
st.sidebar.markdown("coded with :heart:")

# sidebar header
st.sidebar.subheader("Cancer Diagnosis")

# sidebar : choose analysis or prediction page
option = st.sidebar.selectbox(
    'prediction? Select From here...',
     ("Please Select here", "Cancer Diagnosis"))

@st.cache
def load_data():
    test_data = pd.read_csv("input/test.csv")
    return test_data[:50]
    
if option == "Please Select here":
    st.title("Personalized Cancer Diagnosis")
    st.text("Predict the effect of Genetic Variants to enable Personalized Medicine")
    st.markdown("A lot has been said during the past several years about how precision medicine and, more concretely, how genetic testing is going to disrupt the way diseases like cancer are treated. But this is only partially happening due to the huge amount of manual work still required. Memorial Sloan Kettering Cancer Center (MSKCC) launched this competition, accepted by the NIPS 2017 Competition Track,  because they need our help to take personalized medicine to its full potential.")
    # st.title("")
    st.image("https://i.imgur.com/Yg5cyhl.jpg", width=700)
    st.markdown("Once sequenced, a cancer tumor can have thousands of genetic mutations. But the challenge is distinguishing the mutations that contribute to tumor growth (drivers) from the neutral mutations (passengers). \
                Currently this interpretation of genetic mutations is being done manually. This is a very time-consuming task where a clinical pathologist has to manually review and classify every single genetic mutation based on evidence from text-based clinical literature.\
                For this competition MSKCC is making available an expert-annotated knowledge base where world-class researchers and oncologists have manually annotated thousands of mutations.\
                using this knowledge base as a baseline, automatically classifies genetic variations.")
    

elif option == 'Cancer Diagnosis':
    if st.checkbox("Use Sample Data"):
        # st.image("https://i.imgur.com/Yg5cyhl.jpg", width=700)
        data = load_data()
        k = np.random.randint(0, 49)
        auto_gene = data['Gene'][k]
        variation_auto = data['Variation'][k]
        text_auto = data['TEXT'][k]
        three, four = st.beta_columns(2)
        Gene = [str(three.text_input("Gene", auto_gene))]
        Var = [str(four.text_input("Variation", variation_auto))]
        TEXT = [str(st.text_area("Write your research content...", text_auto, height=250))]
    else:
        st.image("https://i.imgur.com/Yg5cyhl.jpg", width=700)
        one, two = st.beta_columns(2)
        Gene = [str(one.text_input("Gene"))]
        Var = [str(two.text_input("Variation"))]
        TEXT = [str(st.text_area("Write your research content...", height=250))]

    gene_vect = pickle.load(open('models/diag_gene_vect.pkl', 'rb'))
    var_vect = pickle.load(open('models/diag_variation_vect.pkl', 'rb'))
    text_vect = pickle.load(open('models/diag_text_vect.pkl', 'rb'))
    model = pickle.load(open('models/diag_model.pkl', 'rb')) 

    gene = gene_vect.transform(Gene)
    var = var_vect.transform(Var)
    text = text_vect.transform(TEXT)

    query_vector = hstack((gene, var, text)).tocsr()


    if st.button("Predict"):
        predicted_cls = model.predict(query_vector)
        st.markdown("Predicted Class")
        st.success(predicted_cls[0])
        predicted_probas = np.round(model.predict_proba(query_vector), 3)
        st.markdown("Predicted class probabilities")
        st.success(predicted_probas[0])

