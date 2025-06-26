import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import time 
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler


#Cargar modelos
model_stacking_all = joblib.load("stack_model_best_all_features_ok.pkl")
model_stacking_top = joblib.load("stack_model_best_top_features.pkl")
model_random_forest = joblib.load("random_forest_model.pkl")
model_svm = joblib.load("svm_model.pkl")
#Cargar Pipeline
pipeline = joblib.load('pipeline.pkl')

# Título principal
st.title("Telco Customer Churn")

# Subtítulo
st.subheader("Focused customer retention programs")

# Sección: About Dataset
st.markdown("### About Dataset")

# Contexto
st.markdown("""
- **Context**  
    Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs.
""")

# Contenido
st.markdown("""
- **Content**  
    Each row represents a customer, each column contains customer’s attributes described on the column Metadata.
""")

# Descripción de lo que incluye el dataset
st.markdown("""
**The data set includes information about:**

1. Customers who left within the last month – the column is called `Churn`
2. Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
3. Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
4. Demographic info about customers – gender, age range, and if they have partners and dependents
""")

# Enlace al dataset
st.markdown("[🔗 Link to Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)")

# Imagen
st.image("variables.png", caption="Variables in the dataset", use_container_width=True)

# Cargar tus datos (ajústalo a tu ruta real)
df = pd.read_csv("df_imputado_Telcom_Churn.csv")

st.title("Exploratory Data Analysis")

st.subheader("Distribution of Churn")
# Crear la figura y graficar con Seaborn
fig, ax = plt.subplots()
sns.countplot(x='Churn', data=df, ax=ax)
ax.set_title('Distribución de Churn')

# Mostrar el gráfico en Streamlit
st.pyplot(fig)

st.subheader("Distribution of Numerical Variables")

# Selección de variables numéricas
numericas = df[['tenure', 'MonthlyCharges', 'TotalCharges']]

# Crear figura y ejes
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Títulos de las variables
variables = numericas.columns

# Graficar con seaborn
for i, col in enumerate(variables):
    sns.histplot(numericas[col], bins=30, kde=True, ax=axes[i], color='skyblue')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frecuency')

# Ajustar el layout
plt.tight_layout()

# Mostrar en Streamlit
st.pyplot(fig)

#Matriz de correlación
st.subheader("Correlation Matrix")
# Calcular la matriz de correlación
corr = numericas.corr()

# Crear figura
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
ax.set_title('Matriz de Correlación')

# Mostrar en Streamlit
st.pyplot(fig)

# Análisis de género y churn
st.subheader("Gender vs Churn")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x='gender', hue='Churn', data=df, ax=ax)
ax.set_title('Gender vs Churn')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()

st.pyplot(fig)

# Análisis de servicio de internet y churn
st.subheader("InternetService vs Churn")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x='InternetService', hue='Churn', data=df, ax=ax)
ax.set_title('InternetService vs Churn')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()

st.pyplot(fig)
##Modeling Section
st.title("Modeling")

st.markdown("For the classification task, three models were used: Stacking Classifier (best model), Random Forest and Support Vector Machine. The best model was trained using both all the variables (24) and only the important variables.")

st.subheader("Feature Importance")
st.markdown(
    "To see the most important variables we use a Random Forest model with the following parameters: "
    "`n_estimators=30`, `max_depth=30`, `min_samples_split=10`"
)
# Cargar los datos de importancia de variables
feat_imp_df = pd.read_csv("feature_importance_df.csv")
# Filtrar el top 10
top_10 = feat_imp_df.sort_values(by='Importance', ascending=False).head(10)

# Crear gráfico interactivo
fig = px.bar(
    top_10.sort_values('Importance'),
    x='Importance',
    y='Feature',
    orientation='h',
    color='Importance',
    color_continuous_scale='Teal',
    title='Top 10 Most Important Features from Random Forest',
    labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
    height=500
)

# Personalizar diseño
fig.update_layout(
    xaxis_title='Feature Importance Score',
    yaxis_title='',
    coloraxis_showscale=False
)

# Mostrar en Streamlit
st.plotly_chart(fig, use_container_width=True)

st.markdown("We select the most important characteristics. The selected variables have the highest values of importance "
    "according to the model. To be more specific, we take the characteristics that have an importance greater than 5%. "
    "We select the 5 variables, and analyzing them, we can see that they are variables that can affect the decision "
    "of whether a customer stays or not in the company.")


st.title("Model Evaluation")
st.subheader("Stacking Classifier")

# Cargar los datos
comparison_df_best_model = pd.read_csv("comparison_df_best_model.csv")

# Métricas y valores
metrics_names = ['Accuracy', 'AUC', 'F1-Score']
all_values = comparison_df_best_model['All Features'][:3].values
top_values = comparison_df_best_model['Top Features'][:3].values

# Crear gráfico interactivo con Plotly
fig = go.Figure()

# Barras para All Features
fig.add_trace(go.Bar(
    x=metrics_names,
    y=all_values,
    name='All Features',
    marker_color='dodgerblue',
    hovertemplate='Model: All Features<br>Metric: %{x}<br>Value: %{y:.4f}<extra></extra>'
))

# Barras para Top Features
fig.add_trace(go.Bar(
    x=metrics_names,
    y=top_values,
    name='Top Features',
    marker_color='orange',
    hovertemplate='Model: Top Features<br>Metric: %{x}<br>Value: %{y:.4f}<extra></extra>'
))

# Layout
fig.update_layout(
    title='Model Performance Comparison: All Features vs. Top Features',
    xaxis_title='Metric',
    yaxis_title='Value',
    barmode='group',
    yaxis=dict(range=[0, 1.05]),
)

# Mostrar en Streamlit
st.plotly_chart(fig, use_container_width=True)

st.subheader("Models with all features")

results = {
    'RandomForest': [0.7906316536550745, 0.6961184737399572, 0.556390977443609],
    'BestModelStacking': [0.8026969481902059, 0.7120152936009713, 0.5825825825825826],
    'SVM': [0.7892122072391767, 0.695152290165078, 0.5547226386806596]
}

metrics = ['Accuracy', 'AUC', 'F1-Score']
num_metrics = len(metrics)
num_models = len(results)
bar_width = 0.13
x = np.arange(num_metrics)

colors = ['orange', 'red', 'deeppink']

fig = go.Figure()

# Añadir barras para cada modelo
for model, color in zip(results.keys(), colors):
    fig.add_trace(go.Bar(
        x=metrics,
        y=results[model],
        name=model,
        marker_color=color,
        hovertemplate='Model: ' + model + '<br>Metric: %{x}<br>Value: %{y}<extra></extra>'
    ))

# Layout
fig.update_layout(
    title='Comparison of Different Ensemble Learning Models Across Various Metrics With All Features',
    xaxis_title='Metric',
    yaxis_title='Metric Value',
    yaxis=dict(range=[0.5, 0.85]),
    barmode='group'
)

# Mostrar en Streamlit
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# INICIALIZAR ESTADO
# -------------------------------
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'selected_feature_set' not in st.session_state:
    st.session_state.selected_feature_set = None

# -------------------------------
# FUNCIONES DE SELECCIÓN
# -------------------------------
def set_model(model):
    st.session_state.selected_model = model
    st.session_state.selected_feature_set = None

def set_feature_option(option):
    st.session_state.selected_feature_set = option

# -------------------------------
# FUNCIÓN PARA FORMULARIO "ALL FEATURES"
# -------------------------------
def all_features_form(model):
    st.subheader("🔍 Enter Customer Information")

    with st.form(key=f"form_all_{model}"):
        col1, col2 = st.columns(2)

        with col1:
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, step=1)
            MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, step=1.0)
            TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, step=10.0)

            gender = st.selectbox("Gender", ['Female', 'Male'])
            SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
            Partner = st.selectbox("Partner", ['Yes', 'No'])
            Dependents = st.selectbox("Dependents", ['Yes', 'No'])

            Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])

        with col2:
            PhoneService = st.selectbox("Phone Service", ['Yes', 'No'])
            MultipleLines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
            InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])

            OnlineSecurity = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
            OnlineBackup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
            DeviceProtection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
            TechSupport = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
            StreamingTV = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
            StreamingMovies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])

            PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
            PaymentMethod = st.selectbox("Payment Method", [
                'Electronic check',
                'Mailed check',
                'Bank transfer (automatic)',
                'Credit card (automatic)'
            ])

        submit = st.form_submit_button("🔎 Predict")

    if submit:
        input_dict = {
            'tenure': [tenure],
            'MonthlyCharges': [MonthlyCharges],
            'TotalCharges': [TotalCharges],
            'gender': [gender],
            'SeniorCitizen': [SeniorCitizen],
            'Partner': [Partner],
            'Dependents': [Dependents],
            'PhoneService': [PhoneService],
            'MultipleLines': [MultipleLines],
            'InternetService': [InternetService],
            'OnlineSecurity': [OnlineSecurity],
            'OnlineBackup': [OnlineBackup],
            'DeviceProtection': [DeviceProtection],
            'TechSupport': [TechSupport],
            'StreamingTV': [StreamingTV],
            'StreamingMovies': [StreamingMovies],
            'PaperlessBilling': [PaperlessBilling],
            'Contract': [Contract],
            'PaymentMethod': [PaymentMethod],
        }

        df_input = pd.DataFrame(input_dict)

        mapeos = {
            'gender': {'Female': 0, 'Male': 1},
            'Partner': {'Yes': 1, 'No': 0},
            'Dependents': {'Yes': 1, 'No': 0},
            'PhoneService': {'Yes': 1, 'No': 0},
            'MultipleLines': {'Yes': 1, 'No': 0, 'No phone service': 0},
            'InternetService': {'DSL': 1, 'Fiber optic': 1, 'No': 0},
            'OnlineSecurity': {'Yes': 1, 'No': 0, 'No internet service': 0},
            'OnlineBackup': {'Yes': 1, 'No': 0, 'No internet service': 0},
            'DeviceProtection': {'Yes': 1, 'No': 0, 'No internet service': 0},
            'TechSupport': {'Yes': 1, 'No': 0, 'No internet service': 0},
            'StreamingTV': {'Yes': 1, 'No': 0, 'No internet service': 0},
            'StreamingMovies': {'Yes': 1, 'No': 0, 'No internet service': 0},
            'PaperlessBilling': {'Yes': 1, 'No': 0}
        }

        for col, mapping in mapeos.items():
            df_input[col] = df_input[col].map(mapping)

        #Medir tiempo de predicción
        start_time = time.time()
        X_input_transformed = pipeline.transform(df_input)

        model_map = {
            "Stacking": model_stacking_all,
            "Random Forest": model_random_forest,
            "SVM": model_svm
        }

        pred = model_map[model].predict(X_input_transformed )[0]
        prob = model_map[model].predict_proba(X_input_transformed )[0][1]
        end_time = time.time()
        elapsed_time = end_time - start_time

        st.markdown("---")
        if pred == 0:
            st.success("📊 Prediction: **Customer Will Stay**")
        else:
            st.error("📊 Prediction: **Customer Will Leave**")
        st.info(f"🔢 Probability of Churn: **{prob:.2%}**")
        st.caption(f"⏱️ Prediction time: {elapsed_time:.4f} seconds")

# -------------------------------
# TÍTULO E INTERFAZ PRINCIPAL
# -------------------------------
st.markdown("<h2 style='text-align: center; color: #4CAF50;'>It's your time to try one</h2>", unsafe_allow_html=True)
st.write("Choose a machine learning model to explore its performance:")

# Botones de modelos
col1, col2, col3 = st.columns(3)
with col1:
    st.button("Stacking Classifier", on_click=set_model, args=("Stacking",), use_container_width=True)
with col2:
    st.button("Random Forest", on_click=set_model, args=("Random Forest",), use_container_width=True)
with col3:
    st.button("Support Vector Machine", on_click=set_model, args=("SVM",), use_container_width=True)

# -------------------------------
# OPCIONES PARA STACKING
# -------------------------------
if st.session_state.selected_model == "Stacking":
    st.markdown("---")
    st.subheader("You've selected: Stacking Classifier")
    st.write("Now choose the feature strategy you'd like to apply:")

    col_a, col_b = st.columns(2)
    with col_a:
        st.button("All Features", on_click=set_feature_option, args=("all",), use_container_width=True)
    with col_b:
        st.button("Top Features", on_click=set_feature_option, args=("top",), use_container_width=True)

# -------------------------------
# LLAMAR A LA FUNCIÓN SEGÚN SELECCIÓN DEL USUARIO
# -------------------------------
if st.session_state.selected_model == "Stacking" and st.session_state.selected_feature_set == "all":
    st.success("✅ You selected: Stacking Classifier with All Features")
    all_features_form("Stacking")

elif st.session_state.selected_model == "Random Forest":
    st.success("✅ You selected: Random Forest")
    all_features_form("Random Forest")

elif st.session_state.selected_model == "SVM":
    st.success("✅ You selected: Support Vector Machine")
    all_features_form("SVM")
# -------------------------------
# FORMULARIO PARA "TOP FEATURES"
# -------------------------------
if st.session_state.selected_model == "Stacking" and st.session_state.selected_feature_set == "top":
    st.success("✅ You selected: Stacking Classifier with Top Features")
    st.subheader("🔍 Enter Only the Top 5 Features")

    with st.form(key="stacking_form_top"):
        col1, col2 = st.columns(2)

        with col1:
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, step=1)
            MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, step=1.0)
            TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, step=10.0)

        with col2:
            contract_month = st.selectbox("Contract: Month-to-month", [0, 1])
            payment_electronic = st.selectbox("Payment Method: Electronic check", [0, 1])

        submit_top = st.form_submit_button("🔎 Predict")

    if submit_top:
        # Crear DataFrame
        df_top = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [MonthlyCharges],
            'TotalCharges': [TotalCharges],
            'Contract_Month-to-month': [contract_month],
            'PaymentMethod_Electronic check': [payment_electronic]
        })

        # Escalar variables numéricas
        scaler = RobustScaler()
        df_top[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
            df_top[['tenure', 'MonthlyCharges', 'TotalCharges']]
        )

        # Asegurar el orden correcto
        ordered_columns = [
            'MonthlyCharges', 'TotalCharges', 'tenure',
            'Contract_Month-to-month', 'PaymentMethod_Electronic check'
        ]
        df_top = df_top[ordered_columns]
        #Medir tiempo de predicción
        start_time = time.time()
        # Predicción
        pred = model_stacking_top.predict(df_top)[0]
        prob = model_stacking_top.predict_proba(df_top)[0][1]
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.markdown("---")
        if pred == 0:
            st.success("📊 Prediction: **Customer Will Stay**")
        else:
            st.error("📊 Prediction: **Customer Will Leave**")
        st.info(f"🔢 Probability of Churn: **{prob:.2%}**")
        st.caption(f"⏱️ Prediction time: {elapsed_time:.4f} seconds")

# Estilo adicional opcional
st.markdown("""
<style>
div.stButton > button {
    height: 3em;
    font-size: 1.1em;
    background-color: #f0f0f5;
    border-radius: 8px;
    color: black;
}
div.stButton > button:hover {
    background-color: #d0f0c0;
}
</style>
""", unsafe_allow_html=True)
