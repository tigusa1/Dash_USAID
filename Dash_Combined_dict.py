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

# Factors
Factors =  {
    'Access_to_Abortion':{
        'location':{
            'x':0.85,
            'y':0.3,
        },
        'text':'Access to Abortion',  
        'value': 0.2,     
    },
    'Access_to_Contraception':{
        'location':{
            'x':0.75,
            'y':0.3,
        },
        'text':'Access to Contraception',   
        'value': 0.2,    
    },
    'Bad_Governance':{
        'location':{
            'x':0.65,
            'y':0.3,
        },
        'text':'Bad Governance',  
        'value': 0.2,     
    },
    'Bully':{
        'location':{
            'x':0.55,
            'y':0.3,
        },
        'text':'Bully',   
        'value': 0.2,       
    },
    'Deportation':{
        'location':{
            'x':0.45,
            'y':0.3,
        },
        'text':'Deportation',   
        'value': 0.2,    
    },
    'Economy':{
        'location':{
            'x':0.35,
            'y':0.3,
        },
        'text':'Economy',   
        'value': 0.2,    
    },
    'Economy_Opportunities':{
        'location':{
            'x':0.25,
            'y':0.3,
        },
        'text':'Economy Opportunities', 
        'value': 0.2,      
    },
    'Exposure_to_Violent_Media':{
        'location':{
            'x':0.15,
            'y':0.3,
        },
        'text':'Exposure to Violent Media',  
        'value': 0.2,     
    },
    'Extortion':{
        'location':{
            'x':0.85,
            'y':0.4,
        },
        'text':'Extortion',   
        'value': 0.2,    
    },
    'Family_Breakdown':{
        'location':{
            'x':0.75,
            'y':0.4,
        },
        'text':'Family Breakdown',  
        'value': 0.2,     
    },
    'Family_Cohesion':{
        'location':{
            'x':0.65,
            'y':0.4,
        },
        'text':'Family Cohesion', 
        'value': 0.2,         
    },
    'Gang_Affiliation':{
        'location':{
            'x':0.55,
            'y':0.4,
        },
        'text':'Gang Affiliation',   
        'value': 0.2,    
    },
    'Gang_Cohesion':{
        'location':{
            'x':0.45,
            'y':0.4,
        },
        'text':'Gang Cohesion',   
        'value': 0.2,    
    },
    'Gang_Control':{
        'location':{
            'x':0.35,
            'y':0.4,
        },
        'text':'Gang Control', 
        'value': 0.2,      
    },
    'Interventions':{
        'location':{
            'x':0.25,
            'y':0.4,
        },
        'text':'Interventions',  
        'value': 0.2,     
    },
    'Impunity_Governance':{
        'location':{
            'x':0.85,
            'y':0.3,
        },
        'text':'Impunity Governance',   
        'value': 0.2,    
    },
    'Machismo':{
        'location':{
            'x':0.85,
            'y':0.90,
        },
        'text':'Machismo',  
        'value': 0.2,     
    },
    'Mental_Health':{
        'location':{
            'x':0.85,
            'y':0.3,
        },
        'text':'Mental Health', 
        'value': 0.2,         
    },
    'Neighborhood_Stigma':{
        'location':{
            'x':0.85,
            'y':0.3,
        },
        'text':'Neighborhood Stigma',   
        'value': 0.2,    
    },
    'School_Quality':{
        'location':{
            'x':0.85,
            'y':0.3,
        },
        'text':'School Quality',   
        'value': 0.2,    
    },
    'Territorial_Fights':{
        'location':{
            'x':0.85,
            'y':0.3,
        },
        'text':'Territorial Fights', 
        'value': 0.2,      
    },
    'Victimizer':{
        'location':{
            'x':0.85,
            'y':0.3,
        },
        'text':'Victimizer',  
        'value': 0.2,     
    },
    'Youth_Empowerment':{
        'location':{
            'x':0.85,
            'y':0.3,
        },
        'text':'Youth Empowerment',   
        'value': 0.2,    
    },
}

Stocks = {
    'Gang_Membership':{
        'flows_in':{
            'Positive_Gang_Perception':{
                'variables_plus': ['Exposure_to_Violent_Media'],
                'variables_minus': ['Law_Enforcement', 'Family_Cohesion', 'Mental_Health'],
            },
            'Migration_Displacement':{
                'variables_plus': ['Deportation'],
                'variables_minus': [],
            },
        },
        'value':0.2,
        'text':'Gang Membership',
        'location':{
            'x':0.1,
            'y':0.1,
        },
    },
    'Incarceration':{
        'flows_in':{
            'Law_Enforcement':{
                'variables_plus': [],
                'variables_minus': [],
            },
        },
        'value':0.4,
        'text':'Incarceration',
        'location':{
            'x':0.2,
            'y':0.2,
        },
    },
    'Law_Enforcement':{
        'flows_in':{
            'Gang_Membership':{
                'variables_plus': ['Gang_Control'],
                'variables_minus': ['Impunity_Governance'],            
            },
            'Physical_Violence':{
                'variables_plus': [],
                'variables_minus': ['Impunity_Governance'],
            },
        },
        'value':0.4,
        'text':'Law Enforcement',
        'location':{
            'x':0.3,
            'y':0.3,
        },
    },
    'Migration_Displacement':{
        'flows_in':{
            'Physical_Violence':{
                'variables_plus': ['Economy'],
                'variables_minus': [],
            },
            'Unemployment':{
                'variables_plus': ['Machismo'],
                'variables_minus': ['Economy_Opportunities'],
            },
        },
        'value':0.4, 
        'text':'Migration Displacement',
        'location':{
            'x':0.4,
            'y':0.3,
        },

    },
    'Positive_Gang_Perception':{
        'flows_in':{
            'Incarceration':{
                'variables_plus': [],
                'variables_minus': [],
            },
            'Unemployment':{
                'variables_plus': ['Exposure_to_Violent_Media'],
                'variables_minus': [],
            },
            'Psychological_Violence':{
                'variables_plus': [],
                'variables_minus': [],
            },
            'Sexual_Violence':{
                'variables_plus': ['Victimizer', 'Gang_Membership'],
                'variables_minus': ['Gang_Control'],
            },
            'Physical_Violence':{
                'variables_plus': ['Family_Breakdown', 'Exposure_to_Violent_Media'],
                'variables_minus': ['Mental_Health'],
            },
        },
        'value':0.4,
        'text':'Positive Gang Perception',
        'location':{
            'x':0.5,
            'y':0.3,
        },
    },
    'Physical_Violence':{
        'flows_in':{
            'Law_Enforcement':{
                'variables_plus': ['Exposure_to_Violent_Media', 'Bad_Governance', 'Neighborhood_Stigma'],
                'variables_minus': ['Youth_Empowerment'],
            },
            'Gang_Membership':{
                'variables_plus': ['Territorial_Fights'],
                'variables_minus': ['Law_Enforcement'],
            },
        },
        'value':0.4,
        'text':'Physical Violence',
        'location':{
            'x':0.6,
            'y':0.3,
        },
    },
    'Psychological_Violence':{
        'flows_in':{
            'Sexual_Violence':{
                'variables_plus': [],
                'variables_minus': [],
            },
            'Physical_Violence':{
                'variables_plus': ['Substance_Abuse'],
                'variables_minus': [],
            },
            'Migration_Displacement':{
                'variables_plus': [],
                'variables_minus': ['Family_Cohesion'],
            },
            'Gang_Membership':{
                'variables_plus': ['Substance_Abuse', 'Gang_Cohesion'],
                'variables_minus': [],
            },
        },
        'value':0.4,
        'text':'Psychological Violence',
        'location':{
            'x':0.75,
            'y':0.3,
        },
    },
    'School_Dropouts':{
        'flows_in':{
            'Teenager_Mothers':{
                'variables_plus': ['Access_to_Abortion', 'Unemployment'],
                'variables_minus': ['Interventions'],
            },
            'Psychological_Violence':{
                'variables_plus': ['Bully'],
                'variables_minus': ['Interventions', 'School_Quality'],
            },
            'Physical_Violence':{
                'variables_plus': [],
                'variables_minus': [],
            },
        },
        'value':0.4,
        'text':'School Dropouts',
        'location':{
            'x':0.95,
            'y':0.3,
        },
    },
    'Sexual_Violence':{
        'flows_in':{
            'Substance_Abuse':{
                'variables_plus': ['Gang_Affiliation', 'School_Dropouts'],
                'variables_minus': ['Economy_Opportunities', 'Law_Enforcement', 'Mental_Health'],
            },
            'Gang_Membership':{
                'variables_plus': [],
                'variables_minus': [],
            },
            'Psychological_Violence':{
                'variables_plus': ['Substance_Abuse'],
                'variables_minus': [],
            },
            'Unemployment':{
                'variables_plus': ['Exposure_to_Violent_Media', 'Family_Breakdown'],
                'variables_minus': ['Youth_Empowerment', 'Gang_Control'],
            },
        },
        'value':0.4,
        'text':'Sexual Violence',
        'location':{
            'x':0.85,
            'y':0.36,
        },
    },
    'Substance_Abuse':{
        'flows_in':{
            'Gang_Membership':{
                'variables_plus': [],
                'variables_minus': [],
            },
        },
        'value':0.4,
        'text':'Substance Abuse',
        'location':{
            'x':0.85,
            'y':0.4,
        },
    },
    'Teenager_Mothers':{
        'flows_in':{
            'Sexual_Violence':{
                'variables_plus': [],
                'variables_minus': ['Access_to_Contraception'],
            },
        },
        'value':0.4,
        'text':'Teenager Mothers',
        'location':{
            'x':0.85,
            'y':0.5,
        },
    },
    'Unemployment':{
        'flows_in':{
            'School_Dropouts':{
                'variables_plus': ['Neighborhood_Stigma', 'Extortion'],
                'variables_minus': ['Economy_Opportunities'],
            },
        },
        'value':0.4,
        'text':'Unemployment',
        'location':{
            'x':0.85,
            'y':0.6,
        },
    },
}

# Bundle parameters for ODE solver
parameters = {
    't_change':20.0,
    'beta':20.0
}

# FIGURE AND IMAGE
fig = go.Figure() # or any Plotly Express function e.g. px.bar(...)
# Add diagram
fig.add_layout_image(
    dict(
        source="/assets/SD_Model_2020-02-26.png",
        xref="x",
        yref="y",
        x=-0.06,
        y=1.065,
        sizex=1.13,
        sizey=1.13,
        opacity=1.0,
        layer="below",
    )
)
# Plot sensitivities
stocks_x = []
stocks_y = []
for s in Stocks:
    stocks_x.append(Stocks[s]['location']['x'])
    stocks_y.append(Stocks[s]['location']['y'])

sens = np.zeros((len(Factors),len(Stocks)))

fig.add_trace(
    go.Scatter(
        x=stocks_x,
        y=stocks_y,
        mode = 'markers',
        marker = {
            'symbol':'square',
            'size':20,
            'colorscale':'Viridis',
            'color':sens[0],
        },
        hovertemplate = "x: %{x:.1f}<br> color: %{sens[0]:.3f}",
        name = "",
    )
)
# Add stocks
fig.add_trace(
    go.Scatter(
        x=[Stocks[stock_id]['location']['x'] for stock_id in sorted(Stocks.keys())] + [0, 1],
        y=[Stocks[stock_id]['location']['y'] for stock_id in sorted(Stocks.keys())] + [0, 1],
        text = [Stocks[stock_id]['text'] for stock_id in sorted(Stocks.keys())] + ["", ""],
        hovertemplate = "%{text}",
        opacity=0.0,
        name="",
    )
)
# Add factors
fig.add_trace(
    go.Scatter(
        x=[Factors[factor_id]['location']['x'] for factor_id in sorted(Factors.keys())] + [0, 1],
        y=[Factors[factor_id]['location']['y'] for factor_id in sorted(Factors.keys())] + [0, 1],
        text = [Factors[factor_id]['text'] for factor_id in sorted(Factors.keys())] + ["", ""],
        hovertemplate = "%{text}",
        opacity=0.0,
        name="",
    )
)
# Set templates
fig.update_layout(
    width=800*1.35,
    height=550*1.4,
    autosize=False,
    margin=dict(l=0, r=0, t=0, b=0),
    template="plotly_white", # white background
    showlegend = False,
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

F_change = np.zeros(len(Factors)) # chaged values based on sliders
F_original = np.zeros(len(Factors)) # chaged values based on sliders
F_label = []
for i, factor_id in enumerate(sorted(Factors.keys())):
    Factors[factor_id]['index'] = i
    F_change[i] = Factors[factor_id]['value'] # don't copy the object, just the value
    F_original[i] = Factors[factor_id]['value']
    F_label.append(Factors[factor_id]['text'])

# Bundle initial conditions for ODE solver
F_0 = [Factors[factor_id]['value'] for factor_id in sorted(Factors.keys())]
S_0 = [Stocks[stock_id]['value'] for stock_id in sorted(Stocks.keys())]
y_0 = S_0

S_label = []
for i, stock_id in enumerate(sorted(Stocks.keys())): # S_GM, S_IN, ...
    Stock = Stocks[stock_id]
    Stock['rate'] = 0.03
    Stock['slope'] = 0.4
    Stock['lower'] = 0.3
    Stock['index'] = i
    S_label.append(Stock['text'])

for i, stock_id in enumerate(sorted(Stocks.keys())): # S_GM, S_IN, ...
    Stock_flows_in = Stocks[stock_id]['flows_in']
    for flow_id in Stock_flows_in: # S_PG, S_MD, ...
        Flow = Stock_flows_in[flow_id]
        Flow['beta'] = 20
        Flow['index'] = Stocks[flow_id]['index']

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

def calc_y(S_values,F_values,P_values):
    for i in range(len(S_values)):
        y_0[i] = S_values[i] 
    for i in range(len(F_values)):
        F_change[i] = F_values[i] 
    parameters['t_change'] = P_values[0]
    parameters['beta'] = P_values[1]
    y_t = Euler(f,y_0,t,parameters) # y_t = odeint(f, y_0, t, args=(parameters,))
    return y_t

def f(y, t, parameters): # 12 variables
    if t>parameters['t_change']:
        for i in range(len(Factors)):
            F_0[i] = F_change[i]
    else:
        for i in range(len(Factors)):
            F_0[i] = F_original[i]
    D_y = np.zeros(len(y))
    Xb = np.zeros(len(y))
    for stock_id in sorted(Stocks.keys()): # S_GM, S_IN, ...
        Stock = Stocks[stock_id]
        Stock_flows_in = Stock['flows_in']
        i = Stock['index']
        for flow_id in sorted(Stock_flows_in.keys()): # S_PG, S_MD, ...
            Flow = Stock_flows_in[flow_id]  
            v_plus = []
            v_minus = []
            for vid in Flow['variables_plus']:
                if vid in Stocks:
                    v_plus.append(y_0[Stocks[vid]['index']])
                    continue
                else:
                    v_plus.append(F_0[Factors[vid]['index']])
                    continue
            for vid in Flow['variables_minus']:
                if vid in Stocks:
                    v_minus.append(y_0[Stocks[vid]['index']])
                    continue
                else:
                    v_minus.append(F_0[Factors[vid]['index']])
                    continue
            beta = parameters['beta']
            j = Flow['index']
            X_flow_in_j = 1
            X_flow_in_j += sum(X for X in v_plus) - sum(X for X in v_minus)
            Xb_flow_in_j =  y[j]*beta*X_flow_in_j
            Xb[i] += Xb_flow_in_j
            Xb[j] += -Xb_flow_in_j
    for stock_id in sorted(Stocks.keys()):
        Stock = Stocks[stock_id]
        rate = Stock['rate']
        slope = Stock['slope']
        lower = Stock['lower']
        i = Stock['index']
        Z = logistic(Xb[i])
        limit = slope*Z + lower
        D_y[i] = rate*(limit-y[i])
    return D_y

def Euler(f, y_0, t, parameters):
    n_iter = 3 # 10 was used initially, took a long time
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

# Make time array for solution
t_stop = 100.
t_increment = 0.3
t = np.arange(0., t_stop, t_increment)

# Calculate sensitivities
S_values_global = np.array(S_0)
F_values_global = np.array(F_0)

# Use global: https://www.pythoncircle.com/post/680/solving-python-error-unboundlocalerror-local-variable-x-referenced-before-assignment/
def calc_sensitivity(i):
    delta = 0.01
    global F_change # need to use global here
    F_change_original = np.array(F_change)
    F_change[i] += delta
    y_t_delta= Euler(f,S_values_global,t,parameters)
    F_change = np.array(F_change_original)
    y_t      = Euler(f,S_values_global,t,parameters)
    Dy = (y_t_delta[-1,:] - y_t[-1,:])/delta
    return Dy

def calc_sensitivities():
    for i in range(len(F_0)):
        sens[i] = calc_sensitivity(i)

# calc_sensitivities()

# Dashboard
def slider_markers(start=0, end=1, step=0.1, red=None):
    nums = [int(num) if isinstance(num, int) or num%0.01 == 0 or num >= 1 else round(num,2) for num in np.arange(start, end+0.0000001, step)] 
    marks = {num: {'label' : str(num), 'style': {'font-size': 14}} for num in nums}
    if red is not None:
        if isinstance(red, int) or red%0.01 == 0 or red >= 1:
            marks[int(red)] = {'label' : str(int(red)), 'style': {'font-size': 14, 'color': '#f50'}}
        else:
            marks[red] = {'label' : str(red), 'style': {'font-size': 14, 'color': '#f50'}}
    return marks

def make_slider(i,slider_label,slider_type,default_value,min_value=0,max_value=1):
    return html.Div(children=[
        html.Label(slider_label,
            style={},
       ),
        dcc.Slider(
            id={'type':slider_type,'index':i},
            min=min_value,
            max=max_value,
            value=default_value,
            marks=slider_markers(min_value, max_value, (max_value-min_value)/5, default_value),
            step=(max_value-min_value)/100,
        ),
    ])

def many_sliders():
    return dbc.Row([
        dbc.Col([
            html.Div([
                make_slider(i,F_label[i],'F_slider',F_0[i]) for i in range(k*4,min((k+1)*4,len(F_0)))
            ])
        ],width=2) for k in range(0,6)
    ],style={'width':'100%'})

F_sliders = many_sliders()

app.layout = html.Div(style={'backgroundColor':'#f6fbfc'}, children=[
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                figure=fig,
                id="SD-model-image",
                config={
                    'displayModeBar': False
                },
            ),
            width=8,
        ),
        dbc.Col([
            dcc.Graph(id='plot_stocks',config={'displayModeBar': False}),
            dbc.Row([
                dbc.Col([
                    html.Div(id='dynamic-slider'),
                    html.Button("Show All Parameters",id={'type':'modal-open','index':0},style ={'float':'right','margin-top':'0px'}),
                    ], width = 12)
            ]),
        ],width=4),
    ]),

    dbc.Modal([
        dbc.ModalHeader([
            "",
            dbc.Button("Close", id={'type':'modal-close','index':0}, className="ml-auto", style={'position':'absolute','right': 20}),
        ]),
        dbc.ModalBody([
            dbc.Row([
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
                ],width=12),
            ],className="pretty_container"),
            dbc.Row([
                dbc.Col(html.H5('Run parameters'),width=3),
                dbc.Col([
                    html.Div([
                        make_slider(0,'t_change','P_slider',parameters['t_change'],0,100)
                    ]),
                ],width=3),
                dbc.Col([
                    html.Div([
                        make_slider(1,'beta','P_slider',parameters['beta'],0,100)
                    ]),
                ],width=3),
            ],className="pretty_container"),
            dbc.Row([
                dbc.Col(html.H5('Factors'),width=12),
                F_sliders,
                ],className="pretty_container"
            ),
        ]),
    ], id={'type':"modal",'index':0}, size='xl'),
])

@app.callback(
    dash.dependencies.Output('plot_stocks', 'figure'),
    [Input({'type':'S_slider','index':ALL}, 'value'),
    Input({'type':'F_slider','index':ALL}, 'value'),
    Input({'type':'P_slider','index':ALL}, 'value'),]
)
def update_graph(S_values,F_values,P_values):
    S_values_global = np.array(S_values)
    F_values_global = np.array(F_values)
    y_t = calc_y(S_values,F_values,P_values)
    return {
        'data':[{
            'x': t,
            'y': y_t[:,k],
            'name': S_label[k]
        } for k in range(len(S_label))],
        'layout': {
            'title':'Stocks over time (using artificial data)',
            'xaxis':{'title':'Time (months)'},
            'yaxis':{'title':'Stocks (normalized units)'}
        },
    }

@app.callback(
    [Output({'type':'S_slider','index':i}, 'value') for i in range(len(S_label))]
   +[Output({'type':'F_slider','index':i}, 'value') for i in range(len(F_label))],
    [Input('SD-model-image', 'clickData'),
     Input({'type':'N_slider','index':0}, 'value')],
    [State({'type':'S_slider','index':ALL}, 'value'),
     State({'type':'F_slider','index':ALL}, 'value'),]
)
def update_slider_value(clickData,dynamic_slider_value,S_values,F_values):
    if clickData is None:
        return S_values + F_values
    text = clickData["points"][0]['text']
    for i, label in enumerate(S_label):
        if label == text:
            S_values[i] = dynamic_slider_value
    for i, label in enumerate(F_label):
        if label == text:
            F_values[i] = dynamic_slider_value
    return S_values + F_values

# CALLBACKS
@app.callback(Output('dynamic-slider', 'children'),
    [Input('SD-model-image', 'clickData')])
def update_dynamic_slider(clickData):
    if clickData is None:
        return make_slider(0,'choose a factor','N_slider',0)
    pointIndex = clickData["points"][0]["pointIndex"] # needed later
    text = clickData["points"][0]['text']
    return make_slider(0, text, 'N_slider', 0)

@app.callback(
    Output({'type':"modal",'index':MATCH}, "is_open"),
    [Input({'type':"modal-open",'index':MATCH}, "n_clicks"),
     Input({'type':"modal-close",'index':MATCH}, "n_clicks")],
    [State({'type':"modal",'index':MATCH}, "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# CAN LEAVE IN FOR PYTHONEVERYWHERE
if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run_server(debug=True,dev_tools_ui=False)
