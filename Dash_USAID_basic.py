# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.graph_objects as go # or plotly.express as px

# FIGURE AND IMAGE
fig = go.Figure() # or any Plotly Express function e.g. px.bar(...)
# Add trace
fig.add_trace(
    go.Scatter(x=[0, 1, 1], y=[0, 0, 1])
)
# Add images
fig.add_layout_image(
        dict(
            source="/assets/SD Model 2020-02-26.png",
            xref="x",
            yref="y",
            x=0,
            y=1,
            sizex=1,
            sizey=1,
            sizing="stretch",
            opacity=1.0,
            layer="below")
)
# Set templates
fig.update_layout(template="plotly_white") # white background

# SET UP APP
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
	'backgroundColor': '#d9eef2',
    'text': '#1f77b4'  # dark blue
}

app.layout = html.Div(style={'backgroundColor':colors['backgroundColor']}, children=[
    html.Div("",style={'padding-top':'5px','backgroundColor':colors['backgroundColor']}),
    html.Div([
        html.Div(
            html.H2(
                'SD Model',
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
        ),
    ], className="row pretty_container"),	

    html.Div([
        html.Div(
            dcc.Graph(figure=fig
            ),
            className="ten columns",
        ),
    ], className="row pretty_container"),

    html.Div([
        html.Div(
            html.Img(
                src=app.get_asset_url('Map of the SD Model.png'),
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

])

if __name__ == '__main__':
    app.run_server(debug=True)
