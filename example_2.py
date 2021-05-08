import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


def load_data():
    '''
    Загрузка данных
    '''
    data = pd.read_csv('data/occupancy_datatraining.txt', sep=",")
    return data

def preprocess_data(data_in):
    '''
    Масштабирование признаков, функция возвращает X и y для кросс-валидации
    '''
    data_out = data_in.copy()
    # Числовые колонки для масштабирования
    scale_cols = ['Temperature', 'Humidity', 'Light', 'CO2']
    new_cols = []
    sc1 = MinMaxScaler()
    sc1_data = sc1.fit_transform(data_out[scale_cols])
    for i in range(len(scale_cols)):
        col = scale_cols[i]
        new_col_name = col + '_scaled'
        new_cols.append(new_col_name)
        data_out[new_col_name] = sc1_data[:,i]
    return data_out[new_cols], data_out['Occupancy']

data = load_data()
data_X, data_y = preprocess_data(data)
# Чтобы в тесте получилось низкое качество используем только 0,5% данных для обучения
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, train_size=0.005, random_state=1)

# Модели
models_list = ['LogR', 'KNN_5', 'SVC', 'Tree', 'RF', 'GB']
clas_models = {'LogR': LogisticRegression(), 
               'KNN_5':KNeighborsClassifier(n_neighbors=5),
               'SVC':SVC(probability=True),
               'Tree':DecisionTreeClassifier(),
               'RF':RandomForestClassifier(),
               'GB':GradientBoostingClassifier()}

roc_auc_dict = {}
for model_name, model in clas_models.items():
    model.fit(X_train, y_train)
    # Предсказание вероятности класса "1" для roc auc
    Y_pred_proba_temp = model.predict_proba(X_test)
    Y_pred_proba = Y_pred_proba_temp[:,1]
    roc_auc = roc_auc_score(y_test.values, Y_pred_proba)
    roc_auc_dict[model_name] = roc_auc

fig = px.bar(x=roc_auc_dict.keys(), y=roc_auc_dict.values())

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(children=[
    html.H1(children='Модели машинного обучения'),
    dcc.Dropdown(
            id='model-name',
            options=[{'label': x, 'value': x} for x in models_list],
            value='LogR',
            clearable=False
        ),
    dcc.Graph(id='roc-graph'),
    dcc.Graph(
        id='result-graph',
        figure=fig
    )
])

@app.callback(
    Output('roc-graph', 'figure'), 
    [Input('model-name', 'value')])
def show_roc(name):
    '''
    Построение графика ROC-кривой
    '''
    model = clas_models[name]
    # Предсказание вероятности класса "1" для roc auc
    Y_pred_proba_temp = model.predict_proba(X_test)
    Y_pred_proba = Y_pred_proba_temp[:,1]
    fpr, tpr, thresholds = roc_curve(y_test.values, Y_pred_proba)
    score = auc(fpr, tpr)
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={score:.4f})',
        labels=dict(
            x='False Positive Rate', 
            y='True Positive Rate'))
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)