import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pycaret.classification import *
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Configuração inicial da página
st.set_page_config(page_title='Simulador - Case Ifood',
                   page_icon='./images/logo_fiap.png',
                   layout='wide',
                   initial_sidebar_state='expanded')

st.title('Simulador - Conversão de Vendas')

# Descrição do App
with st.expander('Descrição do App', expanded=False):
    st.write('O objetivo principal deste app é .....')

# Sidebar com informações e escolha do tipo de entrada
with st.sidebar:
    c1, c2 = st.columns(2)
    c1.image('./images/logo_fiap.png', width=100)
    c2.write('')
    c2.subheader('Auto ML - Fiap [v1]')

    database = st.radio('Fonte dos dados de entrada (X):', ('CSV', 'Online'))

# Abas principais
tab1, tab2 = st.tabs(["Predições", "Análise Detalhada"])

with tab1:
    if database == 'CSV':
        file = st.file_uploader('Selecione o arquivo CSV', type='csv')
        if file:
            Xtest = pd.read_csv(file)
            mdl_rf = load_model('./pickle/ifood_sales_prospecting_model')
            ypred = predict_model(mdl_rf, data=Xtest, raw_score=True)

            with st.expander('Visualizar CSV carregado:', expanded=False):
                qtd_linhas = st.slider('Visualizar quantas linhas do CSV:',
                                       min_value=5,
                                       max_value=Xtest.shape[0],
                                       step=10,
                                       value=5)
                st.dataframe(Xtest.head(qtd_linhas))

            with st.expander('Visualizar Predições:', expanded=True):
                threshold = st.slider('Threshold (ponto de corte para considerar predição como True)',
                                      min_value=0.0,
                                      max_value=1.0,
                                      step=0.1,
                                      value=0.5)
                Xtest['Predicted_Class'] = (ypred['prediction_score_1'] > threshold).astype(int)
                qtd_true = Xtest[Xtest['Predicted_Class'] == 1].shape[0]
                qtd_false = Xtest[Xtest['Predicted_Class'] == 0].shape[0]

                st.metric('Qtd clientes True', value=qtd_true)
                st.metric('Qtd clientes False', value=qtd_false)

                def color_pred(val):
                    color = 'olive' if val > threshold else 'orangered'
                    return f'background-color: {color}'

                df_view = pd.DataFrame({'prediction_score_1': ypred['prediction_score_1'], 'Predicted_Class': Xtest['Predicted_Class']})
                st.dataframe(df_view.style.applymap(color_pred, subset=['prediction_score_1']))

                csv = df_view.to_csv(sep=';', decimal=',', index=True)
                st.download_button(label='Download CSV',
                                   data=csv,
                                   file_name='Predicoes.csv',
                                   mime='text/csv')
        else:
            st.warning('Arquivo CSV não foi carregado')
    else:
        # Layout do aplicativo
        st.title('Predição de Propensão de Compra')

        # Recolher os valores das features do usuário
        accepted_cmp1 = st.number_input('AcceptedCmp1', min_value=0, max_value=1)
        accepted_cmp2 = st.number_input('AcceptedCmp2', min_value=0, max_value=1)
        accepted_cmp3 = st.number_input('AcceptedCmp3', min_value=0, max_value=1)
        accepted_cmp4 = st.number_input('AcceptedCmp4', min_value=0, max_value=1)
        accepted_cmp5 = st.number_input('AcceptedCmp5', min_value=0, max_value=1)
        age = st.number_input('Age', min_value=0)
        complain = st.number_input('Complain', min_value=0, max_value=1)
        education = st.selectbox('Education', ['2n Cycle', 'Basic', 'Graduation', 'Master', 'PhD'])
        income = st.number_input('Income')
        kidhome = st.number_input('Kidhome', min_value=0)
        marital_status = st.selectbox('Marital Status', ['Married', 'Single', 'Together', 'Widow'])
        mnt_fish_products = st.number_input('MntFishProducts')
        mnt_fruits = st.number_input('MntFruits')
        mnt_gold_prods = st.number_input('MntGoldProds')
        mnt_meat_products = st.number_input('MntMeatProducts')
        mnt_sweet_products = st.number_input('MntSweetProducts')
        mnt_wines = st.number_input('MntWines')
        num_catalog_purchases = st.number_input('NumCatalogPurchases', min_value=0)
        num_deals_purchases = st.number_input('NumDealsPurchases', min_value=0)
        num_store_purchases = st.number_input('NumStorePurchases', min_value=0)
        num_web_purchases = st.number_input('NumWebPurchases', min_value=0)
        num_web_visits_month = st.number_input('NumWebVisitsMonth', min_value=0)
        recency = st.number_input('Recency', min_value=0)
        teenhome = st.number_input('Teenhome', min_value=0)
        time_customer = st.number_input('Time_Customer')
        # Slider para escolher o threshold
        threshold = st.slider('Escolha o Threshold', min_value=0.0, max_value=1.0, step=0.01, value=0.5)

        # Criar DataFrame com os valores inseridos pelo usuário
        user_data = pd.DataFrame({
            'AcceptedCmp1': [accepted_cmp1],
            'AcceptedCmp2': [accepted_cmp2],
            'AcceptedCmp3': [accepted_cmp3],
            'AcceptedCmp4': [accepted_cmp4],
            'AcceptedCmp5': [accepted_cmp5],
            'Age': [age],
            'Complain': [complain],
            'Education': [education],
            'Income': [income],
            'Kidhome': [kidhome],
            'Marital_Status': [marital_status],
            'MntFishProducts': [mnt_fish_products],
            'MntFruits': [mnt_fruits],
            'MntGoldProds': [mnt_gold_prods],
            'MntMeatProducts': [mnt_meat_products],
            'MntSweetProducts': [mnt_sweet_products],
            'MntWines': [mnt_wines],
            'NumCatalogPurchases': [num_catalog_purchases],
            'NumDealsPurchases': [num_deals_purchases],
            'NumStorePurchases': [num_store_purchases],
            'NumWebPurchases': [num_web_purchases],
            'NumWebVisitsMonth': [num_web_visits_month],
            'Recency': [recency],
            'Teenhome': [teenhome],
            'Time_Customer': [time_customer]
        })

        # Criar variáveis dummy para o estado civil
        marital_status_dummy = pd.DataFrame({f'Marital_Status_{marital_status}': 1}, 
                                            index=user_data.index)
        user_data = pd.concat([user_data, marital_status_dummy], axis=1)

        user_data['Time_Customer'] = pd.to_datetime(user_data['Time_Customer'])
        user_data['Time_Days_Customer'] = (datetime.now() - user_data['Time_Customer']).dt.days

        label_encoder = LabelEncoder()
        user_data['Education'] = \
            label_encoder.fit_transform(user_data['Education'])

        # Botão para fazer a predição
        if st.button('Prever Propensão de Compra'):
            mdl_rf = load_model('./pickle/ifood_sales_prospecting_model')
            ypred = predict_model(mdl_rf, data=user_data, raw_score=True)
            prediction_proba = mdl_rf.predict_proba(user_data)[:, 1]
            prediction = (prediction_proba > threshold).astype(int)
            st.subheader('Resultado da Predição')
            if prediction == 1:
                st.success('Este cliente é propenso a comprar o produto da campanha.')
            else:
                st.error('Este cliente não é propenso a comprar o produto da campanha.')

with tab2:
    if database == 'CSV' and file:
        st.header("Análise Detalhada das Características dos Clientes")
        threshold = st.slider("Ajuste o Threshold para Análise", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        Xtest['Predicted_Class'] = (ypred['prediction_score_1'] > threshold).astype(int)

        features_to_plot = Xtest.columns.difference(['Predicted_Class', 'prediction_score_1'])
        for feature in features_to_plot:
            fig, ax = plt.subplots()
            sns.boxplot(data=Xtest, x='Predicted_Class', y=feature, ax=ax)
            st.pyplot(fig)
    else:
        st.error('Nenhuma predição disponível para análise. Por favor, carregue e processe um CSV primeiro.')
