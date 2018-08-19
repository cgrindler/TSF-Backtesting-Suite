import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import io
import datetime
import base64
import json
import copy
from flask_caching import Cache
import numpy as np
import time
import os
import redis as rd


#toDo make config here

url = "127.0.0.1"
port = "6379"  
redisConn = rd.from_url(url)
mainFont = {'color': '#696969', 'font-family': 'Product Sans'}
dynamicVariables = ['SAR','AR','SMA','MA','Season']


#TBD
def getArimaModels():
    return ["A","B","C"]

#TBD
def getNetworks():
    return ["NN1","NN2","NN3"]

app = dash.Dash()
app.scripts.config.serve_locally = True
app.config['suppress_callback_exceptions']=True

def checkRedisKeys():
    for item in dynamicVariables:
        if not redisConn.exists(item):
            redisConn.set(item,0)

#toDo: tread css locally

app.css.append_css({
    "external_url": "https://fonts.googleapis.com/css?family=Product+Sans"
})

app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})


HeaderFrame = html.Div([
        
        html.Img(
             #toDO: tread images locally
             src="http://1000logos.net/wp-content/uploads/2016/10/ABB-logo.png",
             style={
                 "height": "80",
                 "width" : "195",
                 "float": "left",
                 "position": "relative"
             },
         ),

        html.H4("Forecasting Toolbox",
            style={
                'font-weight': 'bolder',
                'font-family': 'Product Sans',
                'color': "#696969",
                }),
        html.H5("Time Series Analysis with Tensorflow and Statsmodels", 
            style=mainFont),],

            className="Head"
        )


app.layout = html.Div([

        HeaderFrame,

        dcc.Tabs(
            tabs=[
                {'label': 'Datenverarbeitung', 'value': 1},
                {'label': 'Modellauswahl', 'value': 2},
                {'label': 'Prognoseeinstellungen', 'value': 3},
            ],
            value=1,
            id='maintabs'
        ),
        html.Br(),
        html.Div(id='divmaintab'),
        html.Div(id='hiddenSAR',style={'display':'none'}),
        html.Div(id='hiddenAR',style={'display':'none'}),
        html.Div(id='hiddenSMA',style={'display':'none'}),
        html.Div(id='hiddenMA',style={'display':'none'}),
        html.Div(id='hiddenSeason',style={'display':'none'})
])  

#

@app.callback(Output('divmaintab', 'children'),
              [Input('maintabs', 'value')])
def call_tab_layout(tab_value):
    if tab_value == 1:
        return html.Div([
    
            dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
            ),
            html.Div(id='output-data-upload'),
            #html.Div(dt.DataTable(rows=[{}]), style={'display': 'none'}),
        ])
    elif tab_value == 2:
        return html.Div(
    [
        dcc.Tabs(
            tabs=[
                {'label': 'Box-Jenkins Modelle', 'value': 1},
                {'label': 'Neuronale Netzwerke', 'value': 2},
            ],
            value=1,
            id='modeltabs'
        ),
        html.Br(),
        html.Div(id="divmodeltab")
    ]
)
    elif tab_value == 3:
        return html.Div(
    [
        html.P("TBD", style={'color': '#696969', 'font-family': 'Product Sans'}), 
    ]
)

#
@app.callback(Output('hiddenSAR','children') , [Input('SAR','value')])
def changeSAR(SARvalue):
    redisConn.delete("SAR")
    redisConn.append("SAR",SARvalue)

@app.callback(Output('hiddenAR','children') , [Input('AR','value')])
def changeAR(ARvalue):
    redisConn.delete("AR")
    redisConn.append("AR",ARvalue)

@app.callback(Output('hiddenSMA','children') , [Input('SMA','value')])
def changeSMA(SMAvalue):
    redisConn.delete("SMA")
    redisConn.append("SMA",SMAvalue)

@app.callback(Output('hiddenMA','children') , [Input('MA','value')])
def changeMA(MAvalue):
    redisConn.delete("MA")
    redisConn.append("MA",MAvalue)

@app.callback(Output('hiddenSeason','children') , [Input('Season','value')])
def changeSeason(Seasonvalue):
    redisConn.delete("Season")
    redisConn.append("Season",Seasonvalue)  


@app.callback(dash.dependencies.Output('divmodeltab', 'children'),
              [dash.dependencies.Input('modeltabs', 'value')])
def call_tab_layout(tab_value):
    if tab_value == 2:
        return html.Div([
                html.Div(
                    [
                        html.H6('Neural Networks:',
                        style={
                            'color': '#696969',
                            'font-family': 'Product Sans'}),                      
                        html.P("einstellbare Modellparamter:", style={'color': '#696969', 'font-family': 'Product Sans'}),     
                        html.Div([
                            dcc.Input(placeholder="Hidden Layers",type = "number", style={'color': '#696969', 'font-family': 'Product Sans'}),
                            html.Br(),
                            dcc.Input(placeholder="Number of Inputs",type = "number", style={'color': '#696969', 'font-family': 'Product Sans'}),
                            dcc.Input(placeholder="Number of Inputs",type = "number", style={'display': 'none'}),
                        ]),
                        dcc.RadioItems(
                            id='deepItems',
                            style={'font-family': 'Product Sans'},
                            options=[
                                {'label': "LSTM", 'value': 'lstm'},
                                {'label': 'RNN', 'value': 'rnn'},
                                {'label': 'Feed-Forward', 'value': 'ff'}
                            ],
                            value='active',
                            labelStyle={'display': 'inline-block'}
                        ),
                        html.P("gespeicherte Modelle", style={'color': '#696969', 'font-family': 'Product Sans'}),
                        dcc.Dropdown(
                            id='deepDrop',
                            options=[{'label': 'Modell2', 'value': '2'}]
                        ),
                        html.Div([html.Div(id="output-data-upload")]),
                    ],
                    className='six columns'
                ),
            ], className='row')
    else:      

        checkRedisKeys()

        return html.Div(
                    [
                        html.H6('Box-Jenkins Method:', style=mainFont),
                        html.P("einstellbare Modellparamter:", style=mainFont),     
                        html.Div([
                            html.Div([
                            #toDo: check if it is the first view and set a placeholder
                            dcc.Input(placeholder = "AR", id = "AR", value=float(redisConn.get('AR')), type = "number", style=mainFont),
                            dcc.Input(id = "MA", value=float(redisConn.get('MA')), type = "number", style=mainFont),
                            ]),
                            html.Div([
                            dcc.Input(id = "SAR", value=float(redisConn.get('SAR')),type = "number", style=mainFont),
                            dcc.Input(id = "SMA", value=float(redisConn.get('SMA')),type = "number", style=mainFont),
                            dcc.Input(id = "Season", value=float(redisConn.get('Season')), type="number", style=mainFont)
                            ]),
                        ]),  
                        html.Br(),      
                        html.P("gespeicherte Modelle:", style=mainFont),               
                        dcc.Dropdown(
                            id='boxDrop',
                            options=[{'label': 'Modell 1', 'value': 'one'}],
                        ),
                    ],
                    className='six columns'
                )
                
                
#

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # Use the DataTable prototype component:
        # github.com/plotly/dash-table-experiments
        dt.DataTable(rows=df.to_dict('records')),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

#

@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')])
def update_output(contents):
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(
        io.StringIO(decoded.decode('utf-8')))
    redisConn.delete("tempFrame")
    redisConn.set("tempFrame", df.to_msgpack(compress='zlib'))
    return html.Div([
        dt.DataTable(
            rows=df.to_dict('records'),

            # optional - sets the order of columns
            columns=sorted(df.columns),

            row_selectable=True,
            filterable=True,
            sortable=True,
            selected_row_indices=[],
        )
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
    redisConn.client_kill(url)