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
background_figure_sizex = 1.13
background_figure_sizey = 1.13

colors = {
    'background': '#111111',
    'text': '#1f77b4'  # dark blue
}

# Factors
Factors =  {
    'Access_to_Abortion':{
        'location':{
            'x':0.585,
            'y':0.754,
        },
        'text':'Access to Abortion',  
        'value': 0.2,     
    },
    'Access_to_Contraception':{
        'location':{
            'x':0.55,
            'y':0.6,
        },
        'text':'Access to Contraception',   
        'value': 0.2,    
    },
    'Bad_Governance':{
        'location':{
            'x':0.88,
            'y':0.18,
        },
        'text':'Bad Governance',  
        'value': 0.2,     
    },
    'Bully':{
        'location':{
            'x':0.72,
            'y':0.71,
        },
        'text':'Bully',   
        'value': 0.2,       
    },
    'Deportation':{
        'location':{
            'x':0.9,
            'y':0.31,
        },
        'text':'Deportation',   
        'value': 0.2,    
    },
    'Economy':{
        'location':{
            'x':0.92,
            'y':0.69,
        },
        'text':'Economy',   
        'value': 0.2,    
    },
    'Economy_Opportunities':{
        'location':{
            'x':0.88,
            'y':0.86,
        },
        'text':'Economy Opportunities', 
        'value': 0.2,      
    },
    'Exposure_to_Violent_Media':{
        'location':{
            'x':0.04,
            'y':0.9,
        },
        'text':'Exposure to Violent Media',  
        'value': 0.2,     
    },
    'Extortion':{
        'location':{
            'x':0.67,
            'y':0.92,
        },
        'text':'Extortion',   
        'value': 0.2,    
    },
    'Family_Breakdown':{
        'location':{
            'x':0.345,
            'y':0.8,
        },
        'text':'Family Breakdown',  
        'value': 0.2,     
    },
    'Family_Cohesion':{
        'location':{
            'x':0.53,
            'y':0.195,
        },
        'text':'Family Cohesion', 
        'value': 0.2,         
    },
    'Gang_Affiliation':{
        'location':{
            'x':0.43,
            'y':0.41,
        },
        'text':'Gang Affiliation',   
        'value': 0.2,    
    },
    'Gang_Cohesion':{
        'location':{
            'x':0.63,
            'y':0.34,
        },
        'text':'Gang Cohesion',   
        'value': 0.2,    
    },
    'Gang_Control':{
        'location':{
            'x':0.34,
            'y':0.585,
        },
        'text':'Gang Control', 
        'value': 0.2,      
    },
    'Interventions':{
        'location':{
            'x':0.59,
            'y':0.795,
        },
        'text':'Interventions',  
        'value': 0.2,     
    },
    'Impunity_Governance':{
        'location':{
            'x':0.56,
            'y':0.04,
        },
        'text':'Impunity Governance',   
        'value': 0.2,    
    },
    'Machismo':{
        'location':{
            'x':0.93,
            'y':0.9,
        },
        'text':'Machismo',  
        'value': 0.2,     
    },
    'Mental_Health':{
        'location':{
            'x':0.26,
            'y':0.53,
        },
        'text':'Mental Health', 
        'value': 0.2,         
    },
    'Neighborhood_Stigma':{
        'location':{
            'x':0.875,
            'y':0.62,
        },
        'text':'Neighborhood Stigma',   
        'value': 0.2,    
    },
    'School_Quality':{
        'location':{
            'x':0.72,
            'y':0.745,
        },
        'text':'School Quality',   
        'value': 0.2,    
    },
    'Territorial_Fights':{
        'location':{
            'x':0.7,
            'y':0.36,
        },
        'text':'Territorial Fights', 
        'value': 0.2,      
    },
    'Victimizer':{
        'location':{
            'x':0.38,
            'y':0.55,
        },
        'text':'Victimizer',  
        'value': 0.2,     
    },
    'Youth_Empowerment':{
        'location':{
            'x':0.72,
            'y':0.79,
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
            'x':0.19,
            'y':0.35,
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
            'x':0.09,
            'y':0.125,
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
            'x':0.57,
            'y':0.125,
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
            'x':0.96,
            'y':0.485,
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
            'x':0.185,
            'y':0.69,
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
            'x':0.78,
            'y':0.48,
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
            'x':0.62,
            'y':0.48,
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
            'x':0.66,
            'y':0.865,
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
            'x':0.44,
            'y':0.475,
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
            'x':0.53,
            'y':0.345,
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
            'x':0.515,
            'y':0.715,
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
            'x':0.34,
            'y':0.96,
        },
    },
}

# Bundle parameters for ODE solver
parameters = {
    't_change':20.0,
    'beta':20.0
}

# Plot sensitivities
F_change = np.zeros(len(Factors)) # chaged values based on sliders
F_original = np.zeros(len(Factors)) # chaged values based on sliders
F_label = []
factors_x = []
factors_y = []
factors_text = []
for idx, f in enumerate(Factors.keys()):
    Factor = Factors[f]
    Factor['location']['x'] = Factor['location']['x'] * background_figure_sizex
    Factor['location']['y'] = (Factor['location']['y'] - 1) * background_figure_sizey
    print(Factor['location'])
    Factor['index'] = idx
    F_change[idx] = Factor['value'] # don't copy the object, just the value
    F_original[idx] = Factor['value']
    F_label.append(Factor['text'])
    factors_x.append(Factor['location']['x'])
    factors_y.append(Factor['location']['y'])
    factors_text.append(Factor['text'])

# Bundle initial conditions for ODE solver
F_0 = [Factors[factor_id]['value'] for factor_id in Factors.keys()]
S_0 = [Stocks[stock_id]['value'] for stock_id in Stocks.keys()]
y_0 = S_0

S_label = []
stocks_x = []
stocks_y = []
stocks_text = []
for idx, s in enumerate(Stocks.keys()): # S_GM, S_IN, ...
    Stock = Stocks[s]
    Stock['location']['x'] = Stock['location']['x'] * background_figure_sizex
    Stock['location']['y'] = (Stock['location']['y'] - 1) * background_figure_sizey
    Stock['rate'] = 0.03
    Stock['slope'] = 0.4
    Stock['lower'] = 0.3
    Stock['index'] = idx
    S_label.append(Stock['text'])
    stocks_x.append(Stock['location']['x'])
    stocks_y.append(Stock['location']['y'])
    stocks_text.append(Stock['text'])

for idx, stock_id in enumerate(Stocks.keys()): # S_GM, S_IN, ...
    Stock_flows_in = Stocks[stock_id]['flows_in']
    for flow_id in Stock_flows_in: # S_PG, S_MD, ...
        Flow = Stock_flows_in[flow_id]
        Flow['beta'] = 20
        Flow['index'] = Stocks[flow_id]['index']

sens = np.zeros((len(Factors),len(Stocks)))

def SD_fig(sensitivities=None):
    # FIGURE AND IMAGE
    fig = go.Figure() # or any Plotly Express function e.g. px.bar(...)
    # Add diagram
    fig.add_layout_image(
        dict(
            source="/assets/SD_Model_2020-02-26.png",
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=background_figure_sizex,
            sizey=background_figure_sizey,
            opacity=1.0,
            layer="below",
        )
    )
    if sensitivities is None:
        sensitivities_visible = False
    else:
        sensitivities_visible = True
    fig.add_trace(
        go.Scatter(
            x=stocks_x,
            y=stocks_y,
            mode = 'markers',
            marker = {
                'symbol':'square',
                'size':20,
                'colorscale':'Viridis',
                'color':sensitivities,
            },
            text=sensitivities,
            hovertemplate = 'sensitivity: %{text:.2f}',
            name = "",
            visible = sensitivities_visible,
        )
    )
    # Add stocks
    fig.add_trace(
        go.Scatter(
            x=[Stocks[stock_id]['location']['x'] for stock_id in Stocks.keys()],
            y=[Stocks[stock_id]['location']['y'] for stock_id in Stocks.keys()],
            text = [Stocks[stock_id]['text'] for stock_id in Stocks.keys()],
            # x=[Stocks[stock_id]['location']['x'] for stock_id in sorted(Stocks.keys())] + [0, 1],
            # y=[Stocks[stock_id]['location']['y'] for stock_id in sorted(Stocks.keys())] + [0, 1],
            # text = [Stocks[stock_id]['text'] for stock_id in sorted(Stocks.keys())] + ["", ""],
            hovertemplate = "%{text}",
            opacity=0.0,
            name="",
            visible = not sensitivities_visible,
        )
    )
    # Add factors
    fig.add_trace(
        go.Scatter(
            x=[Factors[factor_id]['location']['x'] for factor_id in Factors.keys()] + [0, 1],
            y=[Factors[factor_id]['location']['y'] for factor_id in Factors.keys()] + [0, 1],
            text = [Factors[factor_id]['text'] for factor_id in Factors.keys()] + ["", ""],
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
        xaxis = dict(
            title = '',
            showticklabels=False,
            fixedrange= True,
            tickvals = [v for v in np.arange(0, 1.01, 0.1)],
            range  = [0, background_figure_sizex],
        ),
        yaxis = dict(
            title = '',
            showticklabels=False,
            fixedrange= True,
            tickvals = [v for v in np.arange(0, 1.01, 0.1)],
            range  = [-background_figure_sizey, 0],
        ),
    )
    return fig

fig = SD_fig()

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
        y_0[i] = S_values[i] # make a copy, order of index i is not important
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
    for stock_id in Stocks.keys(): # S_GM, S_IN, ...
        Stock = Stocks[stock_id]
        Stock_flows_in = Stock['flows_in']
        idx = Stock['index']
        for flow_id in Stock_flows_in.keys(): # S_PG, S_MD, ...
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
            jdx = Flow['index']
            X_flow_in_j = 1
            X_flow_in_j += sum(X for X in v_plus) - sum(X for X in v_minus)
            Xb_flow_in_j =  y[jdx]*beta*X_flow_in_j
            Xb[idx] += Xb_flow_in_j
            Xb[jdx] += -Xb_flow_in_j
    for stock_id in Stocks.keys():
        Stock = Stocks[stock_id]
        rate = Stock['rate']
        slope = Stock['slope']
        lower = Stock['lower']
        idx = Stock['index']
        Z = logistic(Xb[idx])
        limit = slope*Z + lower
        D_y[idx] = rate*(limit-y[idx])
    return D_y

def Euler(f, y_0, t, parameters):
    n_iter = 3 # 10 was used initially, took a long time
    h = (t[1] - t[0])/n_iter
    n = len(t)
    m = len(y_0)
    y = np.zeros([n,m])
    y[0,:] = y_0
    for time_step in range(n-1):
        y_intermediate = y[time_step,:]
        for i_iter in range(n_iter):
            y_intermediate = y_intermediate + np.multiply(h,f(y_intermediate,t[time_step],parameters))
        y[time_step+1,:] = y_intermediate
    return y

# Make time array for solution
t_stop = 100.
t_increment = 0.3
t = np.arange(0., t_stop, t_increment)

# Calculate sensitivities
S_values_global = np.array(S_0)
F_values_global = np.array(F_0)

# Use global: https://www.pythoncircle.com/post/680/solving-python-error-unboundlocalerror-local-variable-x-referenced-before-assignment/
def calc_sensitivity(idx):
    delta = 0.01
    global F_change # need to use global here
    F_change_original = np.array(F_change)
    F_change[idx] += delta
    y_t_delta= Euler(f,S_values_global,t,parameters)
    F_change = np.array(F_change_original)
    y_t      = Euler(f,S_values_global,t,parameters)
    Dy = (y_t_delta[-1,:] - y_t[-1,:])/delta
    return Dy

# def calc_sensitivities():
#     for idx in range(len(F_0)):
#         sens[idx] = calc_sensitivity(idx)

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

def make_slider(idx,slider_label,slider_type,default_value,value,min_value=0,max_value=1):
    return html.Div(children=[
        html.Label(slider_label,
            style={},
       ),
        dcc.Slider(
            id={'type':slider_type,'index':idx},
            min=min_value,
            max=max_value,
            value=value,
            marks=slider_markers(min_value, max_value, (max_value-min_value)/5, default_value),
            step=(max_value-min_value)/100,
        ),
    ])

def many_sliders():
    return dbc.Row([
        dbc.Col([
            html.Div([
                make_slider(idx,F_label[idx],'F_slider',F_0[idx],F_0[idx]) for idx in range(k*4,min((k+1)*4,len(F_0)))
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
                    html.Button("Show All Parameters",id={'type':'modal-open','index':0},
                        style ={'float':'right','margin-top':'0px'}),
                    html.Div("",id='sensitivity-label'),
                    ], width = 12)
            ]),
            dcc.RadioItems(
                id='radio-items',
                options=[
                    {'label': 'Change factors or initial value of stocks', 'value': 'stocks'},
                    {'label': 'Sensitivities with respect to factors', 'value': 'sensitivities'},
                ],
                value='stocks',
                labelStyle={'display': 'inline-block'}
            ),
            html.Div("stocks",id='save-previous-radio',style={'display': 'none'}),
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
                                make_slider(idx,S_label[idx],'S_slider',S_0[idx],S_0[idx]) for idx in range(0,6)
                            ]),
                        ],width=6),
                        dbc.Col([
                            html.Div([
                                make_slider(idx,S_label[idx],'S_slider',S_0[idx],S_0[idx]) for idx in range(6,len(S_0))
                            ]),
                        ],width=6),
                    ]),
                ],width=12),
            ],className="pretty_container"),
            dbc.Row([
                dbc.Col(html.H5('Run parameters'),width=3),
                dbc.Col([
                    html.Div([
                        make_slider(0,'t_change','P_slider',parameters['t_change'],parameters['t_change'],0,100)
                    ]),
                ],width=3),
                dbc.Col([
                    html.Div([
                        make_slider(1,'beta','P_slider',parameters['beta'],parameters['beta'],0,100)
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

def get_S_F(text):
    for s in Stocks:
        Stock = Stocks[s]
        if Stock['text'] == text:
            return (Stock['index'],True,False,text,Stock['value'])
    for f in Factors:
        Factor = Factors[f]
        if Factor['text'] == text:
            return (Factor['index'],False,True,text,Factor['value'])
    print('ERROR IN get_S_F(): label not found')
    return (-1,False,False,'not found')

# CALLBACKS
@app.callback(
    [Output('dynamic-slider', 'children'),
     Output('SD-model-image','figure'),
     Output('sensitivity-label','children'),
     Output('save-previous-radio','children')],
    [Input('SD-model-image', 'clickData'),
     Input('radio-items','value'),],
    [State('dynamic-slider','children'),
     State('sensitivity-label','children'),
     State('save-previous-radio','children'),
     State({'type':'S_slider','index':ALL}, 'value'),
     State({'type':'F_slider','index':ALL}, 'value'),] # use save-previous-radio to check when it is changed
 )
def update_dynamic_slider(clickData,radio,dynamic_slider,prior_label,previous_radio,S_values,F_values):
    if clickData is not None:  # always do this, even if clickData is remnant of previous interaction
        pointIndex = clickData["points"][0]["pointIndex"] # needed later
        text = clickData["points"][0]['text']        
        idx,is_S,is_F,label,default_value = get_S_F(text)
        if is_S:
            saved_value = S_values[idx]
        elif is_F:
            saved_value = F_values[idx]
    if radio=='stocks':
        fig = SD_fig()
        if clickData is None or previous_radio=='sensitivities':
            return ('Select a factor or stock on the diagram.',fig,"",radio)
        return (make_slider(0, text, 'N_slider', default_value, saved_value),fig,"",radio)
    elif radio=='sensitivities':
        if previous_radio=='sensitivities':
            sens = calc_sensitivity(idx)
            sens /= max(sens)
            fig = SD_fig(sensitivities=sens)
            return ('Select a factor',fig,label,radio)
        else:
            fig = SD_fig()
            return ('Select a factor',fig,"",radio)

@app.callback(
    [Output({'type':'S_slider','index':Stocks[s]['index']}, 'value') for s in Stocks]
   +[Output({'type':'F_slider','index':Factors[f]['index']}, 'value') for f in Factors], # ALL doesn't work
    [Input('SD-model-image', 'clickData'),
     Input({'type':'N_slider','index':0}, 'value')],
    [State({'type':'S_slider','index':ALL}, 'value'),
     State({'type':'F_slider','index':ALL}, 'value'),
     State('radio-items','value'),]
)
def update_slider_value(clickData,dynamic_slider_value,S_values,F_values,radio):
    if radio=='stocks':
        if clickData is None:
            return S_values + F_values
        text = clickData["points"][0]['text']
        idx,is_S,is_F,label,default_value = get_S_F(text)
        if is_S:
            S_values[idx] = dynamic_slider_value
        elif is_F:
            F_values[idx] = dynamic_slider_value
        return S_values + F_values

@app.callback(
    Output('plot_stocks', 'figure'),
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
            'name': S_label[k],
            'hovertemplate':'%{y:.2f}'
        } for k in range(len(S_label))],
        'layout': {
            'title':'Stocks over time (using artificial data)',
            'xaxis':{'title':'Time (months)'},
            'yaxis':{'title':'Stocks (normalized units)'}
        }
    }

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
