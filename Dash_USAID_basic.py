# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import json
import collections
import pandas as pd

import plotly.graph_objects as go # or plotly.express as px

# FIGURE AND IMAGE
fig = go.Figure() # or any Plotly Express function e.g. px.bar(...)
# Add trace
fig.add_trace(
    go.Scatter(
        x=[0.6, 0.7, 0.85, 0, 1],
        y=[0.5, 0.5, 0.3,  0, 1],
        text = ["Psychological Violence", "Physical Violence", "Deportation", "", ""],
        hovertemplate = "%{text}",
        opacity=0.0,
        name="",
    )
)
# Add images
fig.add_layout_image(
        dict(
            source="/assets/SD_Model_2020-02-26.png",
            xref="x",
            yref="y",
            x=0,
            y=1,
            sizex=1,
            sizey=1,
            # sizing="stretch",
            opacity=1.0,
            layer="above",
            )
)
# Set templates
fig.update_layout(
    width=800,
    height=550,
    margin=dict(l=20, r=20, t=20, b=20),
    template="plotly_white", # white background
    clickmode='event+select',
    xaxis = go.XAxis(
        title = '',
        showticklabels=False,
        fixedrange= True,
    ),
    yaxis = go.YAxis(
        title = '',
        showticklabels=False,
        fixedrange= True,
    ),
)


# SET UP APP
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
	'backgroundColor': '#d9eef2',
    'text': '#1f77b4'  # dark blue
}

df_children_deportation = pd.read_excel('DEPORTED_CHILDREN.xlsx', header=5)
df_phiv = pd.read_excel('PHYSICAL_VIOLENCE.xlsx')
df_psyv = pd.read_excel('PSYCHOLOGICAL_VIOLENCE.xlsx')

# DATA FIGURE
def deported_children_monthly_plot():
    counter = collections.Counter(df_children_deportation['Month'])
    months = []
    for key in df_children_deportation['Month']:
        if key not in months:
            months.append(key)
    return {
        'data': [{
            'x': months,
            'y': [counter[month] for month in months],
            'type': 'bar',
            'name': 'Deported children 2018',
        }]
    }

def physical_violence_plot():
    return {
        'data': [{
            'x': df_phiv.year.to_list(),
            'y': df_phiv.physical_violence.to_list(),
            'type': 'bar',
            'name': 'Physical Violence', 
        }]
    }

def psychological_violence_plot():
    return {
        'data': [{
            'x': df_psyv.year.to_list(),
            'y': df_psyv.psychological_violence.to_list(),
            'type': 'bar',
            'name': 'Psychological Violence', 
        }]
    }

# def deported_children_life_stage_plot():
#     counter = collections.Counter(df_children['Life stage'])
#     return {
#         'data': [{
#             'x': [key for key in counter.keys()],
#             'y': [counter[key] for key in counter.keys()],
#             'type': 'bar',
#             'name': 'Deported children 2018 life stage',
#         }]
#     }

# def deported_children_reasons_for_migration_plot():
#     counter = collections.Counter(df_children['Reasons for migration'])
#     return {
#         'data': [{
#             'x': [key for key in counter.keys()],
#             'y': [counter[key] for key in counter.keys()],
#             'type': 'bar',
#             'name': 'Deported reason of children 2018',
#         }]
#     }

def plot_figure(pointIndex=-1):
    if pointIndex==2:
        title = 'Monthly Deported Children in 2018'
        fig = deported_children_monthly_plot()
    elif pointIndex==1:
        title = 'Physical violence from 2014 to 2017'
        fig = physical_violence_plot()
    elif pointIndex==0:
        title = 'Psychlogical violence from 2014 to 2017'
        fig = psychological_violence_plot()
    # elif pointIndex==5:
    #     title = 'Deported Children Reasons for Migration in 2018'
    #     fig = deported_children_reasons_for_migration_plot()
    else:
        title = 'Somethine Else in 2018'
        return ({'visibility':'hidden'}, deported_children_monthly_plot(), '')
    return ({'visibility':'visible'}, fig, title)


    # Sex Birth date  Interview age   Age group   Life stage  Reasons for migration

# LAYOUT OF THE APP
app.layout = html.Div(style={'backgroundColor':colors['backgroundColor']}, children=[
    html.Div("",style={'padding-top':'5px','backgroundColor':colors['backgroundColor']}),
    html.Div([
        html.Div(
            html.H2(
                'SD Data and Model Visualization, Version Alpha 1.0',
                style={
                    'textAlign': 'left',
                    'color': colors['text'],
                },
            ),
            className="eight columns",
        ),
        html.Div(
            html.Button(
                'Reset',
                id='reset-button',
                n_clicks=0,
                style = {
                    # 'width': '30%',
                    'float': 'right',
                    'margin-top':'30px',
                },
            ),
            className="four columns",
            hidden = True,
        ),
    ], className="row pretty_container"),	

    html.Div([
        html.Div(
            dcc.Graph(
                figure=fig,
                id="SD-model-image",
                config={
                    'displayModeBar': False
                },
            ),
            className="eight columns",
        ),
        html.Div(
            [
                html.H5(children='',
                    id='figure-h'),
                dcc.Graph(
                    id="figure-graph",
                    config={
                    'displayModeBar': False
                    },
                )
            ],
            id="figure-div",
            className="four columns",
            style = {
                'visibility':"hidden", # visible or hidden
            },
        ),
    ], className="row pretty_container"),

    html.Div([
        html.Div(
            html.Img(
                src=app.get_asset_url('Map_of_the_SD_Model.png'),
                style = {
                    'width': '100%',
                    'height': '100%',
                },
            ),
            className="twelve columns",
        ),
        html.Div(
            className="zero columns",
        ),
    ], className="row pretty_container"),

    html.Div([
        html.Div(
            html.H3(
                'SD Team in El Salvador',
                style={
                    'textAlign': 'center',
                    'color': colors['text'],
                },
            ),
            className="six columns",
        ),
        html.Div(
            html.H3(
                'SD Team in Honduras',
                style={
                    'textAlign': 'center',
                    'color': colors['text'],
                },
            ),
            className="six columns",
        ),
    ], className="row pretty_container"),   


    html.Div([
        html.Div(
            html.Img(
                src=app.get_asset_url('El_Salvador_picture.png'),
                style = {
                    'width': '100%',
                    'height': '100%',
                },
            ),
            className="six columns",
        ),
        html.Div(
            html.Img(
                src=app.get_asset_url('Honduras_picture.png'),
                style = {
                    'width': '100%',
                    'height': '100%',
                },
            ),
            className="six columns",
        ),
    ], className="row pretty_container"),
])

# CALLBACKS
@app.callback([
    Output('figure-div', 'style'),
    Output('figure-graph', 'figure'),
    Output('figure-h', 'children'),
    ],
    [Input('SD-model-image', 'clickData')])
def display_click_data(clickData):
    if clickData is None:
        pointIndex = -1
    else:
        pointIndex = clickData["points"][0]["pointIndex"] # needed later

    return plot_figure(pointIndex)

if __name__ == '__main__':
    app.run_server(debug=True)
