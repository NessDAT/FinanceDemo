import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import pandas as pd
from pycaret.classification import *
import plotly.express as px

st.set_page_config(page_title='Finance', layout='wide')

# Stylesheet link for bootstap icons
st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.3/font/bootstrap-icons.css">',
    unsafe_allow_html=True)

# Creating the sidebar menu-------------------------------------------------------------------------------------------------------------
with st.sidebar:
    selected = option_menu("Finance", ["About", "Loan Defaulting","Churn Prediction", "Fraud Detection", "Stock Price Prediction", 
                        "Asset Valuation"],
                         icons=['journal-bookmark-fill','columns-gap', 'person-circle', 'exclamation-diamond', 'graph-up-arrow','clock'],
                         menu_icon="building", default_index=0,
                         styles={"nav-link": {"--hover-color": "#eee"}}
                        )
    st.markdown("###")
    

    st.markdown("<a href='https://vanessaattafynn-demo-demo-site-at0bqf.streamlitapp.com/'>\
            <button class='b1' style='background-color:#F35106;color:white; border:None;border-radius:10px;\
            padding:15px;min-height:15px;min-width: 80px;' type='button'>\
            Go Home  <i class='bi bi-box-arrow-up-right'></i></button></a>",unsafe_allow_html=True)

#Get feature importance---------------------------------------------------------------------------------------------------------------
def get_final_column_names(pycaret_pipeline, sample_df):
    for (name, method) in pycaret_pipeline.named_steps.items():
        if method != 'passthrough' and name != 'trained_model':
            print(f'Running {name}')
            sample_df = method.transform(sample_df)
    return sample_df.columns.tolist()

def get_feature_importances_df(pycaret_pipeline, sample_df, n = 10):
    
    final_cols = get_final_column_names(pycaret_pipeline, sample_df)
    
    try:
        variables = pycaret_pipeline["trained_model"].feature_importances_
        
    except:
        variables = np.mean([
                        tree.feature_importances_ for tree in pycaret_pipeline["trained_model"].estimators_
                        ], axis=0)
    
    coef_df = pd.DataFrame({"Variable": final_cols, "Value": variables})
    sorted_df = (
        coef_df.sort_values(by="Value", ascending=False)
        .head(n)
        .sort_values(by="Value", ascending=True).reset_index(drop=True)
    )
    return sorted_df




#-----------------------------------------------------------------------------------------------------------------------
# THIS IS THE ABOUT SECTION
#------------------------------------------------------------------------------------------------------------------------
if selected == "About":
    #st.image('images/about-header2.png')
    col1, col2= st.columns(2)

    with col1:
        st.subheader("Introduction")
        st.markdown("<p style='text-align:justify;font-size:20px;font-family:helvetica;'> \
                    In today’s era of digitization, staying updated on technological advancements is a \
        necessity for businesses to both outsmart the competition and achieve desired business growth.\
        The recent years have seen a rapid acceleration in the pace of disruptive technologies such as \
        AI and ML in Finance due to improved software and hardware. The finance sector, specifically, has\
         seen a steep rise in the use cases of machine learning applications to advance better outcomes \
         for both consumers and businesses.</p>",
                    unsafe_allow_html=True)

        st.markdown("##")
        st.markdown("##")

    with col2:
        st.markdown("#")
        st.image("data/money.png")

    st.markdown("###")
    st.subheader("How Does Machine Learning In Finance Work?")
    st.markdown("<p style='text-align:justify;font-size:20px;font-family:helvetica;'>Machine Learning works \
        by extracting meaningful insights from raw sets of data \
        and provides accurate results. This information is then used to solve complex and data-rich \
        problems that are critical to the banking & finance sector. Further, machine learning algorithms \
        are equipped to learn from data, processes, and techniques used to find different insights.</p>",
        unsafe_allow_html=True)

    

    st.markdown("")

    st.subheader("Use Cases Being Tackled In This Platform")

    st.markdown("<ul>\
            <li style='font-size:20px;font-family:helvetica;'>Loan Default Prediction</li>\
            <li style='font-size:20px;font-family:helvetica;'>Churn Prediction</li>\
            <li style='font-size:20px;font-family:helvetica;'>Fraud Detection</li>\
            <li style='font-size:20px;font-family:helvetica;'>Stock Price Prediction</li>\
            <li style='font-size:20px;font-family:helvetica;'>Asset Valuation</li>\
        </ul>\
        </p>",
        unsafe_allow_html=True)



#-----------------------------------------------------------------------------------------------------------------------
# THIS IS THE LOAN DEFAULTING SECTION
#------------------------------------------------------------------------------------------------------------------------

if selected == "Loan Defaulting":
    choose = option_menu(None,["Individual Prediction","Batch Prediction"],
        icons=['person-rolodex', 'grid-3x3-gap-fill'],default_index=0,orientation="horizontal")

# INDIVIDUAL LOAN DEFAULTING
#------------------------------------------------------------------------------------------------------------------------

    if choose == "Individual Prediction":
        ID = st.text_input('Client ID','5000001',max_chars=10)
        def expander_obj():
            with st.expander('Input Client Data',expanded=True):
                #form = st.form(key='my_form', clear_on_submit=True)
                tenure = st.number_input(label='Client Tenure', min_value=0, format="%d")
                loan_amt = st.number_input(label='Loan Amount', min_value=0, format="%d")
                EMI = st.number_input(label='Estimated Monthly Installment(EMI)', min_value=0, format="%d")
                rate = st.number_input(label='Interest Rate', min_value=0, format="%d")
                numLoan = st.number_input(label='Number of Loans taken', min_value=0, format="%d")
                noEMI = st.number_input(label='Number of time EMI has been paid in advance', min_value=0, format="%d")
                max_Sanc = st.number_input(label='Maximum amount sanctioned on live loans', min_value=0, format="%d")
                newLoan = st.number_input(label='Number of new loans in the last 3months', min_value=0, format="%d")
                personalLoan = st.number_input(label='Time since last personal loan was taken (months)', min_value=0, format="%d")
                due_30days = st.number_input(label='Number of times 30 days past due in last 6months', min_value=0, format="%d")
                due_60days = st.number_input(label='Number of times 60 days past due in last 6months', min_value=0, format="%d")
                due_90days = st.number_input(label='Number of times 90 days past due in last 3months', min_value=0, format="%d")
                age = st.number_input(label='Age', min_value=0, format="%d")
                empType = st.selectbox('Employment Type', ['Salaried','Self-employed','Pensioner','Student','Unemployed'])
                gender = st.selectbox('Gender', ['Male','Female'])

                st.markdown('##')

            submit_button = st.button('Submit')

            if submit_button:
                st.markdown('##')
                st.markdown('##')
                data = {'Tenure':[tenure],'Loan_amt':[loan_amt], 'EMI':[EMI], 'InterestRate':[rate],
                'noAdvanceEMI':[noEMI], 'Num_of_Loans':[numLoan],'maxSanctions_Live':[max_Sanc], 'newLoans':[newLoan],
                'timeSinceLastPersonalLoan':[personalLoan], 'pastDue_30':[due_30days],'pastDue_60':[due_60days],
                'pastDue_90':[due_90days],'Age':[age],'Emp_type':[empType],'Gender':[gender]}
                data = pd.DataFrame.from_dict(data)

                loaded_model2 = load_model('Loan_default/models/Default_model2')
                pred_data = predict_model(loaded_model2, data=data)
                pred_data.rename(columns={"Label":"PredictedDefaulter","Score":"PredictionConfidence"},inplace=True)

                default_value = pred_data.at[0,'PredictedDefaulter']
                default_prob = round((pred_data.at[0, 'PredictionConfidence']*100), 2)
                default_prob_str = default_prob.astype(str)

                col1,col2,col3 = st.columns([1,0.7,3])
                with col1:
                    st.markdown("<p style='font-size:1.8em;font-weight:bold;color:#F35106'>Client ID :</p>",unsafe_allow_html=True)
                    st.markdown("<p style='font-size:1.8em;font-weight:bold;color:#F35106'>Churn : </p>",unsafe_allow_html=True)
                    st.markdown("<p style='font-size:1.8em;font-weight:bold;color:#F35106'>Probability : </p>",unsafe_allow_html=True)
                with col2:
                    st.subheader(ID)
                    if default_value == 0:
                        st.subheader("NO")
                    else:
                        st.subheader("YES")

                    st.subheader(default_prob_str + " %")

                

                st.write(pred_data)

        expander_obj()


# BATCH LOAN DEFAULTING
#------------------------------------------------------------------------------------------------------------------------


    if choose == "Batch Prediction":
        st.markdown("##")
        uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
        st.markdown("##")
        df = {}
        pred_df = {}
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            with st.expander('Preview Uploaded Data',expanded=True):
                st.write(df)

            if st.button('Submit'):
                st.markdown("____")
                st.subheader("Loan Default Prediction")
                loaded_model = load_model("Loan_default/models/default_model")
                pred_df = predict_model(loaded_model, data=df)
                pred_df.rename(columns={"Label":"PredictedDefaulter","Score":"PredictionConfidence"},inplace=True)
                pred_df['Target'] = pred_df["PredictedDefaulter"]
                pred_df['Target'].replace(0,'Non-Defaulter',inplace=True)
                pred_df['Target'].replace(1,'Defaulter',inplace=True)
                st.write(pred_df)

# ADDING VISUALIZATION
#------------------------------------------------------------------------------------------------------------------------

                st.markdown("##")
                st.markdown("_____")
                st.subheader("Visualize Results")
                st.markdown("##")
                st.markdown("##")

                col1,col2 = st.columns(2)
                with col1:
                    st.markdown("**Defaulter Vs Not Defaulter Pie Chart**")
                    fig1 = px.pie(pred_df, names='Target',hole=0.5,color_discrete_sequence=px.colors.sequential.RdBu)
                    fig1.update_layout(width=400,height=500,margin=dict(l=1,r=1,b=1,t=1))
                    st.write(fig1)

                with col2:
                    st.markdown("**Features that Contributed the Most to Prediction**")
                    feature_imps_df = get_feature_importances_df(loaded_model, pred_df, n = 10)
                    feature_imps_df
       














#-----------------------------------------------------------------------------------------------------------------------
# THIS IS THE CUSTOMER CHURN PREDICTION SECTION
#------------------------------------------------------------------------------------------------------------------------


if selected == "Churn Prediction":
    choose = option_menu(None,["Individual Prediction","Batch Prediction"],
        icons=['person-rolodex', 'grid-3x3-gap-fill'],default_index=0,orientation="horizontal")

# INDIVIDUAL CHURN PREDICTION
#------------------------------------------------------------------------------------------------------------------------
    if choose == "Individual Prediction":
        ID = st.text_input('Client ID','5000001',max_chars=10)
        def expander_obj():
            with st.expander('Input Client Data',expanded=True):
                #form = st.form(key='my_form', clear_on_submit=True)
                DOB = st.date_input(label='Date of Birth')
                gender = st.selectbox('Gender', ['Male','Female'])
                geography = st.selectbox('Region', ['Greater Accra','Western Region','Central Region'])
                estSalary = st.number_input(label='Estimted Salary(GHC)', min_value=1000, format="%d")
                st.markdown("_____")
                tenure = st.number_input(label='Client Tenure', min_value=1, format="%d")
                creditscore = st.number_input(label='Estimated Credit Score', min_value=350, format="%d")
                balance = st.number_input(label='Account Balance', min_value=0, format="%d")
                numOfPdts = st.number_input(label='Number of Products', min_value=0, format="%d")
                isActive = st.radio('Is An Active Member', ('Yes','No'))
                hasCrCard = st.radio('Has Credit Card', ('Yes','No'))

                if isActive == "Yes":
                    isActive = 1
                else:
                    isActive = 0

                if hasCrCard == "Yes":
                    hasCrCard = 1
                else:
                    hasCrCard = 0

                st.markdown('##')

            submit_button = st.button('Submit')

            if submit_button:
                st.markdown('##')
                st.markdown('##')
                data = {'Gender':[gender], 'Geography':[geography], 'EstimatedSalary':[estSalary],
                'Tenure':[tenure], 'CreditScore':[creditscore], 'Balance':[balance], 'NumOfProducts':[numOfPdts],
                'IsActiveMember':[isActive],'HasCrCard':[hasCrCard]}
                data = pd.DataFrame.from_dict(data)

                loaded_model3 = load_model('Churn/models/churn_model1')
                pred_data = predict_model(loaded_model3, data=data)
                pred_data.rename(columns={"Label":"PredictedChurn","Score":"PredictionConfidence"},inplace=True)

                churn_value = pred_data.at[0,'PredictedChurn']
                churn_prob = round((pred_data.at[0, 'PredictionConfidence']*100), 2)
                churn_prob_str = churn_prob.astype(str)

                col1,col2,col3 = st.columns([1,0.7,3])
                with col1:
                    st.markdown("<p style='font-size:1.8em;font-weight:bold;color:#F35106'>Client ID :</p>",unsafe_allow_html=True)
                    st.markdown("<p style='font-size:1.8em;font-weight:bold;color:#F35106'>Churn : </p>",unsafe_allow_html=True)
                    st.markdown("<p style='font-size:1.8em;font-weight:bold;color:#F35106'>Probability : </p>",unsafe_allow_html=True)
                with col2:
                    st.subheader(ID)
                    if churn_value == 0:
                        st.subheader("NO")
                    else:
                        st.subheader("YES")

                    st.subheader(churn_prob_str + " %")

                

                st.write(pred_data)

        expander_obj()



# BATCH CHURN PREDICTION
#------------------------------------------------------------------------------------------------------------------------

    if choose == "Batch Prediction":
        st.markdown("##")
        uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
        st.markdown("##")
        df = {}
        pred_df = {}
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            with st.expander('Preview Uploaded Data',expanded=True):
                st.write(df)

            if st.button('Submit'):
                st.markdown("____")
                st.subheader("E-commerce Churn Prediction")
                loaded_model = load_model("Churn/models/finance_churn_model2")
                pred_df = predict_model(loaded_model, data=df)
                pred_df.rename(columns={"Label":"PredictedChurn","Score":"PredictionConfidence"},inplace=True)
                pred_df['Target'] = pred_df["PredictedChurn"]
                pred_df['Target'].replace(0,'Not Churn',inplace=True)
                pred_df['Target'].replace(1,'Churn',inplace=True)
                st.write(pred_df)


# ADDING VISUALIZTION
#------------------------------------------------------------------------------------------------------------------------

                st.markdown("##")
                st.markdown("_____")
                st.subheader("Visualize Results")
                st.markdown("##")
                st.markdown("##")

                col1,col2 = st.columns(2)
                with col1:
                    st.markdown("**Chart of Churn vs Not Churn Clients**")

                    fig1 = px.pie(pred_df, names='Target',hole=0.5,color_discrete_sequence=px.colors.sequential.Jet)
                    fig1.update_layout(width=400,height=500,margin=dict(l=1,r=1,b=1,t=1))
                    st.write(fig1)

                with col2:
                    st.markdown("**Features that Contributed the Most to Prediction**")
                    feature_imps_df = get_feature_importances_df(loaded_model, pred_df, n = 10)
                    feature_imps_df

                with st.expander('See More Info on Churn Clients'):
                    st.markdown("more info")


#-----------------------------------------------------------------------------------------------------------------------
# THIS IS THE FRAUD DETECTION SECTION
#------------------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------------------------
# THIS IS THE STOCK PRICE PREDICTION SECTION
#------------------------------------------------------------------------------------------------------------------------





#-----------------------------------------------------------------------------------------------------------------------
# THIS IS THE ASSET VALUATION SECTION
#------------------------------------------------------------------------------------------------------------------------






































