import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc

import numpy as np
# from scipy.integrate import odeint
import plotly.graph_objects as go

# APP SETUP
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP,external_stylesheets[0]])
app.config.suppress_callback_exceptions = True

colors = {
    'background': '#111111',
    'text': '#1f77b4'  # dark blue
}

# GM = Gang_Membership
# IN = Incarceration
# LE = Law_Enforcement
# MD = Migration_Displacement
# PHV= Physical_Violence
# PG = Positive_Gang_Perception
# PSV= Psychological_Violence
# SD = School_Dropouts
# SV = Sexual_Violence
# SA = Substance_Abuse
# TM = Teenager_Mothers
# UN = Unemployment

# Factors
Access_to_Abortion = [ 0.2 ]
Access_to_Contraception = [ 0.2 ]
Bad_Governance = [ 0.2 ]
Bully = [ 0.2 ]
Deportation = [ 0.2 ]
Economy = [ 0.2 ]
Economy_Opportunities = [ 0.2 ]
Exposure_to_Violent_Media = [ 0.2 ]
Extortion = [ 0.2 ]
Family_Breakdown = [ 0.2 ]
Family_Cohesion = [ 0.2 ]
Gang_Affiliation = [ 0.2 ]
Gang_Cohesion = [ 0.2 ]
Gang_Control = [ 0.2 ]
Interventions = [ 0.2 ]
Impunity_Governance = [ 0.2 ]
Machismo = [ 0.2 ]
Mental_Health = [ 0.2 ]
Neighborhood_Stigma = [ 0.2 ]
School_Quality = [ 0.2 ]
Territorial_Fights = [ 0.2 ]
Victimizer = [ 0.2 ]
Youth_Empowerment = [ 0.2 ]

F_label = [
    'Access_to_Abortion', 'Access_to_Contraception', 'Bad_Governance', 'Bully', 'Deportation', 'Economy',
    'Economy_Opportunities', 'Exposure_to_Violent_Media', 'Extortion', 'Family_Breakdown', 'Family_Cohesion',
    'Gang_Affiliation', 'Gang_Cohesion', 'Gang_Control', 'Interventions', 'Impunity_Governance', 'Machismo', 'Mental_Health',
    'Neighborhood_Stigma', 'School_Quality', 'Territorial_Fights', 'Victimizer', 'Youth_Empowerment',
]

F_0 = [
    Access_to_Abortion, Access_to_Contraception, Bad_Governance, Bully, Deportation, Economy,
    Economy_Opportunities, Exposure_to_Violent_Media, Extortion, Family_Breakdown, Family_Cohesion, 
    Gang_Affiliation, Gang_Cohesion, Gang_Control, Interventions, Impunity_Governance, Machismo, Mental_Health, 
    Neighborhood_Stigma, School_Quality, Territorial_Fights, Victimizer, Youth_Empowerment
]

F_change = np.zeros(len(F_0)) # chaged values based on sliders
for i in range(len(F_0)):
    F_change[i] = F_0[i][0] # don't copy the object, just the value

# stocks
S_label = [
    'Gang_Membership', 'Incarceration', 'Law_Enforcement', 'Migration_Displacement',
    'Physical_Violence', 'Positive_Gang_Perception', 'Psychological_Violence',
    'School_Dropouts', 'Sexual_Violence', 'Substance_Abuse', 'Teenager_Mothers', 'Unemployment'
]

# Initial values
S_GM_0 = 0.2
S_IN_0 = 0.4
S_LE_0 = 0.4
S_MD_0 = 0.4
S_PHV_0 = 0.4
S_PG_0 = 0.4
S_PSV_0 = 0.4
S_SD_0 = 0.4
S_SV_0 = 0.4
S_SA_0 = 0.4
S_TM_0 = 0.4
S_UN_0 = 0.4

# Initialize all stocks
S_GM, S_IN, S_LE, S_MD, S_PHV, S_PG, S_PSV, S_SD, S_SV, S_SA, S_TM, S_UN = [[0] for i in range(len(S_label))]

# Bundle initial conditions for ODE solver
S_0 = [ S_GM_0, S_IN_0, S_LE_0, S_MD_0, S_PHV_0, S_PG_0, S_PSV_0, S_SD_0, S_SV_0, S_SA_0, S_TM_0, S_UN_0 ]
y_0 = S_0

Stocks = {
    'S_GM':{
        'flows_in':{
            'S_PG':{
                'variables_plus': [Exposure_to_Violent_Media],
                'variables_minus': [S_LE, Family_Cohesion, Mental_Health],
            },
            'S_MD':{
                'variables_plus': [Deportation],
                'variables_minus': [],
            },
        },
    },
    'S_IN':{
        'flows_in':{
            'S_LE':{
                'variables_plus': [],
                'variables_minus': [],
            },
        },
    },
    'S_LE':{
        'flows_in':{
            'S_GM':{
                'variables_plus': [Gang_Control],
                'variables_minus': [Impunity_Governance],            
            },
            'S_PV':{
                'variables_plus': [],
                'variables_minus': [Impunity_Governance],
            },
        },
    },
    'S_MD':{
        'flows_in':{
            'S_PV':{
                'variables_plus': [Economy],
                'variables_minus': [],
            },
            'S_UN':{
                'variables_plus': [Machismo],
                'variables_minus': [Economy_Opportunities],
            },
        }    
    },
    'S_PG':{
        'flows_in':{
            'S_IN':{
                'variables_plus': [],
                'variables_minus': [],
            },
            'S_UN':{
                'variables_plus': [Exposure_to_Violent_Media],
                'variables_minus': [],
            },
            'S_PSV':{
                'variables_plus': [],
                'variables_minus': [],
            },
            'S_SV':{
                'variables_plus': [Victimizer, S_GM],
                'variables_minus': [Gang_Control],
            },
            'S_PV':{
                'variables_plus': [Family_Breakdown, Exposure_to_Violent_Media],
                'variables_minus': [Mental_Health],
            },
        },
    },
    'S_PV':{
        'flows_in':{
            'S_LE':{
                'variables_plus': [Exposure_to_Violent_Media, Bad_Governance, Neighborhood_Stigma],
                'variables_minus': [Youth_Empowerment],
            },
            'S_GM':{
                'variables_plus': [Territorial_Fights],
                'variables_minus': [S_LE],
            },
        },
    },
    'S_PSV':{
        'flows_in':{
            'S_SV':{
                'variables_plus': [],
                'variables_minus': [],
            },
            'S_PV':{
                'variables_plus': [S_SA],
                'variables_minus': [],
            },
            'S_MD':{
                'variables_plus': [],
                'variables_minus': [Family_Cohesion],
            },
            'S_GM':{
                'variables_plus': [S_SA, Gang_Cohesion],
                'variables_minus': [],
            },
        },
    },
    'S_SD':{
        'flows_in':{
            'S_TM':{
                'variables_plus': [Access_to_Abortion, S_UN],
                'variables_minus': [Interventions],
            },
            'S_PSV':{
                'variables_plus': [Bully],
                'variables_minus': [Interventions, School_Quality],
            },
            'S_PV':{
                'variables_plus': [],
                'variables_minus': [],
            },
        },
    },
    'S_SV':{
        'flows_in':{
            'S_SA':{
                'variables_plus': [Gang_Affiliation, S_SD],
                'variables_minus': [Economy_Opportunities, S_LE, Mental_Health],
            },
            'S_GM':{
                'variables_plus': [],
                'variables_minus': [],
            },
            'S_PSV':{
                'variables_plus': [S_SA],
                'variables_minus': [],
            },
            'S_UN':{
                'variables_plus': [Exposure_to_Violent_Media, Family_Breakdown],
                'variables_minus': [Youth_Empowerment, Gang_Control],
            },
        },
    },
    'S_SA':{
        'flows_in':{
            'S_GM':{
                'variables_plus': [],
                'variables_minus': [],
            },
        },
    },
    'S_TM':{
        'flows_in':{
            'S_SV':{
                'variables_plus': [],
                'variables_minus': [Access_to_Contraception],
            },
        },
    },
    'S_UN':{
        'flows_in':{
            'S_SD':{
                'variables_plus': [Neighborhood_Stigma, Extortion],
                'variables_minus': [Economy_Opportunities],
            },
        },
    },
}

i = 0
for stock in Stocks: # S_GM, S_IN, ...
    Stock = Stocks[stock]
    Stock['rate'] = 0.03
    Stock['slope'] = 0.4
    Stock['lower'] = 0.3
    Stock['index'] = i
    i += 1

for stock in Stocks: # S_GM, S_IN, ...
    Stock_flows_in = Stocks[stock]['flows_in']
    for flow in Stock_flows_in: # S_PG, S_MD, ...
        Flow = Stock_flows_in[flow]
        Flow['beta'] = 20
        Flow['index'] = Stocks[flow]['index']

def logistic(x):
    return 1 / (1 + np.exp(-x))

# S = stocks, X = covariates, Z = transformed variable between 0 and 1
# S_limit(Z) = upper limit for stocks (at equilibrium) as a function of Z
# S_lower = S_Limit(0)
# b = positive and negative coefficients representing flows into and out of stocks
# Governing variable: Z = logistic(X*b)
# Limiting curve: S_limit(Z) = S_slope*Z + S_lower
# Rates:  D_S = a*(S_limit - S)
# Note: the flows do not satisfy mass conversation, but this is okay because the stocks have different units
#   (for instance, the flow into unemployment is many times larger than the flow out of school dropout)

def f(y, t, parameters): # 12 variables
    S_GM[0], S_IN[0], S_LE[0], S_MD[0], S_PHV[0], S_PG[0], S_PSV[0], S_SD[0], S_SV[0], S_SA[0], S_TM[0], S_UN[0] = y

    if t>parameters['t_change']:
        for i in range(len(F_0)):
            F_0[i][0] = F_change[i]

    D_y = np.zeros(len(y))

    Xb = np.zeros(len(y))
    for stock in Stocks: # S_GM, S_IN, ...
        Stock = Stocks[stock]
        Stock_flows_in = Stock['flows_in']
        i = Stock['index']
        for flow in Stock_flows_in: # S_PG, S_MD, ...
            Flow = Stock_flows_in[flow]            
            v_plus = Flow['variables_plus'] # X1, X2, ...
            v_minus = Flow['variables_minus']
            beta = Flow['beta']
            j = Flow['index']
            X_flow_in_j = 1
            X_flow_in_j += sum(X[0] for X in v_plus) - sum(X[0] for X in v_minus)
            Xb_flow_in_j =  y[j]*beta*X_flow_in_j
            Xb[i] += Xb_flow_in_j
            Xb[j] += -Xb_flow_in_j

    for stock in Stocks:
        Stock = Stocks[stock]
        rate = Stock['rate']
        slope = Stock['slope']
        lower = Stock['lower']
        i = Stock['index']
        Z = logistic(Xb[i])
        limit = slope*Z + lower
        D_y[i] = rate*(limit-y[i])

    return D_y

# Bundle parameters for ODE solver
parameters = {
    't_change':20.0,
}

# Make time array for solution
t_stop = 100.
t_increment = 0.3
t = np.arange(0., t_stop, t_increment)

# Dashboard
def slider_markers(start=0, end=1, step=0.1, red=None):
    marks = {str(round(num,2)): {'label' : str(round(num,2)), 'style': {'font-size': 14}} for num in np.arange(start, end, step)}
    if red is not None:
        marks[str(red)] = {'label' : str(red), 'style': {'font-size': 14, 'color': '#f50'}}
    return marks

def make_slider(i,slider_label,slider_type,default_value):
    return html.Div(children=[
        html.Label(slider_label,
            style={},
       ),
        dcc.Slider(
            id={'type':slider_type,'index':i},
            min=0,
            max=1,
            value=default_value,
            marks=slider_markers(0, 1, 0.2, default_value),
            step=None
        ),
    ])

def many_sliders():
    return dbc.Row([
        dbc.Col([
            html.Div([
                make_slider(i,F_label[i],'F_slider',F_0[i][0]) for i in range(k*4,min((k+1)*4,len(F_0)))
            ])
        ],width=2) for k in range(0,6)
    ],style={'width':'100%'})

F_sliders = many_sliders()

app.layout = html.Div(style={'backgroundColor':'#f6fbfc'}, children=[
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='plot_stocks',config={'displayModeBar': False})
        ],width=8),
        dbc.Col([
            dbc.Col(html.H5('Initial Stock Values'),width=12),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        make_slider(i,S_label[i],'S_slider',S_0[i]) for i in range(0,6)
                    ]),
                ],width=6),
                dbc.Col([
                    html.Div([
                        make_slider(i,S_label[i],'S_slider',S_0[i]) for i in range(6,len(S_0))
                    ]),
                ],width=6),
            ]),
        ],width=4),
    ],className="pretty_container"),

    dbc.Row([
        dbc.Col(html.H5('Factors'),width=12),
        F_sliders,
        ],className="pretty_container"
    ),
])

def Euler(f, y_0, t, parameters):
    n_iter = 10
    h = (t[1] - t[0])/n_iter
    n = len(t)
    m = len(y_0)
    y = np.zeros([n,m])
    y[0,:] = y_0
    for i in range(n-1):
        y_intermediate = y[i,:]
        for j in range(n_iter):
            y_intermediate = y_intermediate + np.multiply(h,f(y_intermediate,t[i],parameters))
        y[i+1,:] = y_intermediate
    return y

@app.callback(
    dash.dependencies.Output('plot_stocks', 'figure'),
    [Input({'type':'S_slider','index':ALL}, 'value'),
    Input({'type':'F_slider','index':ALL}, 'value'),]
)
def update_graph(S_values,F_values):
    for i in range(len(S_values)):
        y_0[i] = S_values[i] 
    for i in range(len(F_values)):
        F_change[i] = F_values[i] 
    # Call the ODE solver
    y_t = odeint(f, y_0, t, args=(parameters,))
    # y_t = Euler(f, y_0, t, parameters)
    return {
        'data':[{
            'x': t,
            'y': y_t[:,k],
            'name': S_label[k]
        } for k in range(len(S_label))],
        'layout': {
            'title':  'Stocks'
        },
    }

# CAN LEAVE IN FOR PYTHONEVERYWHERE
if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run_server(debug=True,dev_tools_ui=False)
