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
from sklearn.neighbors import KNeighborsClassifier


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1(children='Метод ближайших соседей'),
    dcc.Slider(
        id='cv-slider',
        min=1,
        max=10,
        step=1,
        value=5,
        marks={i:str(i) for i in range(1,11)}
    ),
    dcc.Slider(
        id='knn-slider',
        min=1,
        max=50,
        step=1,
        value=5,
        marks={i:str(i) for i in range(1,51)} 
    ),
    dcc.Graph(id='scores-graphic'),
    html.Div(id='scores-output')
])

def load_data():
    '''
    Загрузка данных
    '''
    data = pd.read_csv('data/occupancy_datatraining.txt', sep=",", nrows=500)
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
data_len = data.shape[0]

@app.callback(
    Output(component_id='scores-graphic', component_property='figure'),
    Output(component_id='scores-output', component_property='children'),
    Input(component_id='knn-slider', component_property='value'),
    Input(component_id='cv-slider', component_property='value')
)
def update_knn_slider(knn_slider, cv_slider):
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=knn_slider), 
        data_X, data_y, scoring='accuracy', cv=cv_slider)
    #Формирование графика
    fig = px.bar(x=list(range(1, len(scores)+1)), y=scores)
    #Вывод среднего значения
    mean_text = 'Усредненное значение accuracy по всем фолдам - {}'.format(np.mean(scores))
    return fig, mean_text

if __name__ == '__main__':
    app.run_server(debug=True)