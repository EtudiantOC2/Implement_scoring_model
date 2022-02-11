#------------- Configuration de l'environnement streamlit et des packages python -------------

# Configuration préalable de Streamlit via la commande "streamlit run dashboard.py" 

import streamlit as st
import numpy as np
import matplotlib as plt
import pandas as pd
import seaborn as sns
import pickle
import altair as alt
from PIL import Image


#------------- Configuration des titres de notre dashboard -------------

st.set_page_config(layout = "wide") # Configuration largeur & titre dashboard

def formatting_title_1(title): # Format des grands titres
    return st.markdown(f"<h1 style='text-align: center; color: #900C3F;font-size:45px'>{title}</h1>", unsafe_allow_html=True)

def formatting_title_2(title):
    return st.markdown(f"<h1 style='text-align: left; color: white;font-size:25px'>{title}</h1>", unsafe_allow_html=True)

def formatting_title_2_green(title):
    return st.markdown(f"<h1 style='text-align: left; color: green;font-size:25px'>{title}</h1>", unsafe_allow_html=True)

def formatting_title_2_red(title):
    return st.markdown(f"<h1 style='text-align: left; color: ;font-size:25px'>{title}</h1>", unsafe_allow_html=True)

def formatting_title_3(title):
    return st.markdown(f"<h1 style='text-align: left; color: white;font-size:20px'>{title}</h1>", unsafe_allow_html=True)

def formatting_title_4(title):
    return st.markdown(f"<h1 style='text-align: left; color: white;font-size:10px'>{title}</h1>", unsafe_allow_html=True)

formatting_title_1('Dashboard interactif à destination des gestionnaires de la relation client ')


#------------- Chargement de toutes les données nécessaires à la formalisation -------------

def loading_csv_data(path): # Chargement dataset
    '''
    Retourne le dataframe à partir du chemin qu'on lui désigne
    '''
    df = pd.read_csv(path)
    df.drop("Unnamed: 0", axis = 1, inplace = True)
    return df

data = loading_csv_data(r"C:\Users\pauline_castoriadis\Documents\implement_scoring_model\data\df_test.csv") # Dataset
applicant_id_list = data['applicant_loan_id'].tolist() # Liste des id clients
selected_id = st.text_input('Veuillez saisir l\'identifiant d\'un client:', 100001) # Id sélectionné
loaded_model = pickle.load(open('C:\\Users\\pauline_castoriadis\\Documents\\implement_scoring_model\\model\\best_model.pkl','rb')) # Modèle ML

def loading_predicted_data(df,id_col): # Chargement prédiction
    '''
    Réalise la prédiction pour l'ensemble des clients contenus dans la base
    '''
    selection_col =  list(df.loc[:, (df.columns != id_col)])
    data_selection = df[selection_col].values
    prediction =  pd.DataFrame(data = loaded_model.predict_proba(data_selection)[:, 0],columns=['prediction'])
    result = pd.concat([df, prediction], axis = 1)
    return result

predicted_data = loading_predicted_data(data,'applicant_loan_id') # Données tous clients

def defining_group(df): # Définir le groupe auquel appartient le client étudié
    '''
    Rajouter une colonne de groupe selon la probabilité de remboursement
    '''
    if (df['prediction'] < 0.71):
        return "restudy"
    elif (df['prediction'] > 0.75):
        return "refuse"
    else:
        return "monitor"

predicted_data['group'] = predicted_data.apply(defining_group, axis = 1)

id_data = predicted_data[predicted_data['applicant_loan_id'] == int(selected_id)] # Données client sélectionné
prediction_id = round(id_data['prediction'].iat[0,],2)
group_id = id_data['group'].iat[0,]

#------------- Présentation de la prédiction réalisée sur le client -------------

def checking_id_format(id,id_list): # Vérification id sélectionné
    '''
    Retourne une erreur si l'utilisateur a rentré un mauvais code client
    '''
    if len(id) > 0:
        if (len(id) != 6) or (id == '') or (id not in id_list) :
            message = st.error("Attention le code client rentré est invalide")
        else:
            message = st.write("Découvrez votre client")
    return message

checking_id_format(selected_id,applicant_id_list)

title_probability,selected_id_probability = st.columns(2)

with title_probability:
    formatting_title_2('La probabilité du client de ne pas rembourser son prêt est de :')

def displaying_id_score(id,threshold):# Affichage de la proba/client
    if id < threshold :
        formatting_title_2_red(id)
    else:
        formatting_title_2_green(id)

with selected_id_probability:
    displaying_id_score(prediction_id,0.5)

st.markdown("<hr/>",unsafe_allow_html=True)

#------------- Quelques KPIs clés sur le client -------------

formatting_title_2('Apprenez à mieux connaitre vos clients')

def display_numerical_kpi(titre,df,id,col):
    '''
    Affiche un KPI par colonne et par client sélectionné, pour les colonnes numériques
    '''
    st.markdown(titre)
    kpi = round(df.at[df[df['applicant_loan_id']==int(id)].index[0],col],0)
    st.markdown(f"<h1 style='text-align: center; color: red;'>{kpi}</h1>", unsafe_allow_html = True)

kpi_1, kpi_2, kpi_3,kpi_4 = st.columns(4)

def display_sex_kpi(titre,df,id,col):
    '''
    Affiche le sexe du client sélectionné
    '''
    st.markdown(titre)
    kpi_value = df.at[df[df['applicant_loan_id']==int(id)].index[0],col]
    if kpi_value == 0 :
        kpi = "Femme"
    elif kpi_value == 1:
        kpi = "Homme"
    else:
        kpi = "Autre"
    st.markdown(f"<h1 style='text-align: center; color: red;'>{kpi}</h1>", unsafe_allow_html = True)

with kpi_1:
    display_sex_kpi("**Sexe**",data,selected_id,'applicant_gender')

with kpi_2:
    display_numerical_kpi("**Age**",data,selected_id,'applicant_age')

def display_status_kpi(titre,df,id,col):
    '''
    Affiche le sexe du client sélectionné
    '''
    st.markdown(titre)
    kpi_value = df.at[df[df['applicant_loan_id']==int(id)].index[0],col]
    if kpi_value == 0 :
        kpi = "Marié(e) civilement"
    elif kpi_value == 1:
        kpi = "Marié(e)"
    elif kpi_value == 2:
        kpi = "Séparé(e)"  
    elif kpi_value == 3:
        kpi = "Célibataire"    
    elif kpi_value == 4:
        kpi = "Statut inconnu"   
    else:
        kpi = "Veuf/Veuve"
    st.markdown(f"<h1 style='text-align: center; color: red;'>{kpi}</h1>", unsafe_allow_html = True)

with kpi_3:
    display_status_kpi("**Statut marital**",data,selected_id,'applicant_family_status')

with kpi_4:
    display_numerical_kpi("**Revenus**",data,selected_id,'applicant_total_income')

st.markdown("<hr/>",unsafe_allow_html = True)

#------------- Quelques analyses au sujet du client  -------------

def chart_type_1(col1,col2,group,color):
    data = predicted_data[['group',col1,col2]]
    custom_chart = alt.Chart(data).mark_line().encode(
        x = col1,
        y = col2,
        color = alt.Color('group',scale = alt.Scale(domain = [group],range = [color]),legend = None)).properties(width = 500,height = 250)
    return (st.altair_chart(custom_chart))

chart_1,chart_2,chart_3 = st.columns(3)

with chart_1:
    formatting_title_4('Groupe de clients très peu susceptibles de ne pas rembourser leur crédit et dont le dossier peut être réexaminé')
    chart_type_1('applicant_age','total_credit_amount','refuse','#C70039')

with chart_2:
    formatting_title_4('Groupe de clients très peu susceptibles de ne pas rembourser leur crédit et dont le dossier peut être réexaminé')
    chart_type_1('applicant_age','total_credit_amount','monitor','#900C3F')

with chart_3:
    formatting_title_4('Groupe de clients très peu susceptibles de ne pas rembourser leur crédit et dont le dossier peut être réexaminé')
    chart_type_1('applicant_age','total_credit_amount','restudy','#900C3F')



#------------- Possibilité pour le conseiller de rédiger un rapport -------------

formatting_title_2('Rédigez un rapport pour le client dont vous venez de regarder le dossier')

def loading_excel_data(path):
    '''
    Retourne le dataframe à partir du chemin qu'on lui désigne
    '''
    df = pd.read_excel(path)
    return df

report_data = loading_excel_data(r"C:\Users\pauline_castoriadis\Documents\implement_scoring_model\data\reports.xlsx")
 
form = st.form(key="annotation")

with form: # Formulaire à remplur
    cols = st.columns((1, 1))
    author = cols[0].text_input("Nom du conseiller :")
    report_type = cols[1].selectbox(
        "Objet du rapport :", ["Dossier à ré-étudier", "Rendez-vous à fixer", "Documents manquants","Autre"], index=2
    )
    comment = st.text_area("Commentaire :")
    cols = st.columns(2)
    date = cols[0].date_input("Date de rapport :")
    report_severity = cols[1].slider("Criticité :", 1, 5, 2)
    submitted = st.form_submit_button(label = "Soumettre")

if submitted:
    report = {'Client':selected_id,'Conseiller': author, 'Objet': report_type, 'Urgence': report_severity, 'Commentaire': comment,'date': str(date)}
    report_data = report_data.append(report, ignore_index = True)
    st.success("Merci, votre rapport a bien été enregistré")
    st.balloons()

expander = st.expander("Voir tous les rapports produits")
with expander:
    st.write(report_data)


#------------- Logo de fin (droits d'auteurs) -------------

logo = Image.open(r'C:\Users\pauline_castoriadis\Documents\implement_scoring_model\images\logo.jpg')

copyright_1,copyright_2 = st.columns(2)

with copyright_1:
    st.write('eee')

with copyright_2:
    st.image(logo,width=100)