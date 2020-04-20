# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

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
            html.Img(
                src=app.get_asset_url('SD Model 2020-02-26.png'),
                style = {
                    'width': '100%',
                    'height': '100%',
                },
            ),
            className="nine columns",
        ),
        html.Div(
            className="three columns",
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
