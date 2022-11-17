import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc # conda install -c conda-forge dash-bootstrap-components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random

from Mother import Mother

random.seed(10)
# INITIALIZE DATA FOR CLASS MOTHER
# read file of data with columns 1-4 [Mother]: wealth, education, age, number of children
# simulate for health, gestational age, predisposition for ANC
df = pd.read_excel("SDR_Mother.xlsx")
wealth      = df['Wealth'].to_numpy()
education   = df['Education'].to_numpy()
age         = df['Age'].to_numpy()
no_children = df['No_children'].to_numpy()

n_repeat = 1
for i in range(n_repeat):
    wealth = wealth.repeat(n_repeat)
    education = education.repeat(n_repeat)
    age = age.repeat(n_repeat)
    no_children = no_children.repeat(n_repeat)
    wealth = wealth.repeat(n_repeat)

no_mothers  = len(age)

# quality   = 0.7

proximity = 0.8

# CLASS MOTHER
# class Mother_simple():
#     def __init__(self, urban=True, wealth=False):
#         self.urban = urban # True or False
#         self.wealth = wealth
#         self.proportion_healthy = 0.4
#         self.proportion_d_home = 0.3
#         rand = np.random.random(1)[0]
#         if rand < self.proportion_d_home:
#             self.prefer_d_home = True
#         else:
#             self.prefer_d_home = False
#         self.QoC_threshold = 0.3
#         self.QoC_home = 0.1
#         self.QoC_healthy_mother_threshold = 0.03
#         self.QoC_healthy_baby_threshold = 0.04
#
#     def health_BL(self):
#         if np.random.random(1)[0] < self.proportion_healthy:
#             health_status_at_BL = 1;
#         else:
#             health_status_at_BL = 0;
#         return health_status_at_BL
#
#     def decide(self, QoC_2, QoC_4):
#         if self.proportion_d_home and QoC_2 < self.QoC_threshold and QoC_4 < self.QoC_threshold:
#             self.decision = 0
#         elif QoC_2 < QoC_4:
#             self.decision = 2
#         else:
#             self.decision = 1
#         return self.decision # e.g., 0=home delivery, 1=2/3 delivery, 2=4/5 delivery
#
#     def health_outcome(self, QoC_2, QoC_4):
#         QoCs = (self.QoC_home, QoC_2, QoC_4)
#         QoC = QoCs[self.decision]
#         QoC_received = QoC * np.random.random(1)[0]
#         if self.QoC_healthy_mother_threshold < QoC_received:
#             health_outcome_of_mother = 0
#         else:
#             health_outcome_of_mother = 1
#
#         if self.QoC_healthy_baby_threshold < QoC_received:
#             health_outcome_of_baby = 0
#         else:
#             health_outcome_of_baby = 1
#
#         return health_outcome_of_baby, health_outcome_of_mother
#
# GENERATE MOTHERS
# mothers = []
# num_mothers = 100
# proportion_wealthy = 0.1
# proportion_urban = 0.3
# for i in range(num_mothers):
#     mothers.append(Mother(urban=(np.random.random(1)[0] < proportion_urban),
#                           wealth=(np.random.random(1)[0] < proportion_urban)))

# APP SETUP

# Make time array for solution
nt = 48 # number of months

# RR  Resources for RMNCH
# R2  Healthcare financing (Resources for 2/3 facilities)
# R4  Healthcare financing (Resources for 4/5 facilities)
# RS  SDR transition financing
# PDS Policy: Development support

# P_P, P_A, P_D
# Names, initial values
S_info = [
    ['P_P',   0.2, 'Political_Goodwill'         ],
    ['P_A',   0.2, 'SDR_Adoption_Other_Factors' ],
    ['P_M',   0.2, 'Advocacy/Media'             ],
    ['P_I',   0.2, 'Stakeholder_Involvement'    ],
    ['P_SP',  0.2, 'Support_for_Policy'         ],

    ['P_DP',  0.2, 'Development_Support'        ],
    ['P_RR',  0.2, 'Resources_RMNCH'            ],
    ['L4_HF', 0.2, 'H_Financing_4/5'            ],
    ['L2_HF', 0.2, 'H_Financing_2/3'            ],
    ['S_TF', 0.2, 'SDR_Transition_Financing'    ],

    ['L4_HR', 0.2, 'HR_Readiness_4/5'           ],
    ['L4_S',  0.2, 'Supplies/Infrastructure_4/5'],
    ['L2_HR', 0.2, 'HR_Readiness_2/3'           ],
    ['L2_S',  0.2, 'Supplies/Infrastructure_2/3'],
    ['S_T',   0.2, 'SDR_HR_Training'            ],
    ['S_FR',  0.2, 'SDR_Facility_Repurposing'   ],

    ['L4_DC', 0.4, 'Delivery_Capacity_4/5'      ],
    ['L2_DC', 0.4, 'Delivery_Capacity_2/3'      ],
    ['P_D',   0.2, 'Performance_Data'           ],
    ['L4_Q', 0.5, 'Quality_4/5'],
    ['L2_Q', 0.2, 'Quality_2/3'],
]
S_names, S_0, S_label, S_idx_names = [],[],[],[]

for i in range(len(S_info)):
    name = S_info[i][0]
    S_names.append(name)
    S_idx_names.append([i, name])
    S_0.append(    S_info[i][1])
    S_label.append(S_info[i][2])
    globals()[name + '_0'] = S_0[i]
    globals()[name] = np.zeros(nt)

# FACTORS
# Names, initial values
F_info = [
    ['Funding_MNCH',          0.2, 'Funding_MNCH'         ],
    ['Prioritization_MNCH',   0.2, 'Prioritization_MNCH'  ],
    ['Delayed_disbursement',  0.2, 'Delayed_disbursement' ],
    ['Adherence_budget',      0.2, 'Adherence_budget'     ],
    ['Employee_incentives',   0.2, 'Employee_incentives'  ],
    ['Lack_promotion',        0.2, 'Lack_promotion'       ],
    ['Lack_action_depletion', 0.2, 'Lack_action_depletion'],
    ['Visibility',            0.2, 'Visibility'           ],
]
#   ['BL_Capacity_2',         0.2, 'BL_Capacity_L2/3 (100s)'],
#   ['BL_Capacity_4',         0.2, 'BL_Capacity_L4/5 (100s)'],

F_names, F_0, F_label, F_idx_names = [],[],[],[]
for i in range(len(F_info)):
    name = F_info[i][0]
    F_names.append(name)
    F_idx_names.append([i, name])
    F_0.append(    F_info[i][1])
    F_label.append(F_info[i][2])
    globals()[name] = F_0[i]
    globals()[name] = np.zeros(nt)

BL_Capacity_factor = 10

F_change   = np.zeros(len(F_0)) # changed values based on sliders
F_original = np.zeros(len(F_0)) # original values
for i in range(len(F_0)):
    F_change[i]   = F_0[i] # don't copy the object, just the value
    F_original[i] = F_0[i]

y_0 = S_0

def calc_y(S_values, F_values, P_values):
    # P_values = parameter values
    for i in range(len(F_values)):
        F_change[i] = F_values[i]
    parameters['t_change'] = P_values[0]
    parameters['beta'] = P_values[1]

    beta  = parameters['beta'] / 10
    y_t   = np.zeros((nt,len(S_values)))
    t_all = np.zeros(nt)
    anc_t, health_t, gest_age_t, deliveries, facilities = {0:[0]}, {0:[0]}, {0:[0]}, {0:[0]}, {0:[0]}
    # anc_t, health_t, gest_age_t, deliveries, facilities = [[]]*nt, [[]]*nt, [[]]*nt, [[]]*nt, [[]]*nt # NG
    num_deliver_home, num_deliver_2, num_deliver_4, num_deliver_total = \
        np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
    pos_HO, neg_HO = np.zeros([nt,3]), np.zeros([nt,3])

    for i in range(len(S_values)):
        y_t[0,i] = S_values[i]

    for idx,name in S_idx_names:
        globals()[name][0] = S_values[idx]

    mothers = []

    for mother in range(0, no_mothers):
        mothers.append(Mother(wealth[mother], education[mother], age[mother], no_children[mother], nt))

    for t in range(0,nt-1):
        if t > parameters['t_change']:
            for idx, name in F_idx_names:
                globals()[name] = F_change[idx]
        else:
            for idx, name in F_idx_names:
                globals()[name] = F_original[idx]

        t_all[t+1] = t_all[t] + 1 # increment by month
        gest_age, health, anc, delivery, facility = [], [], [], [], []
        for idx,name in S_idx_names:
            d_name = 'd' + name
            globals()[d_name + '_in'] = 0.0
            globals()[d_name + '_out'] = 0.0

        if t == 0:
            L2_demand = 0
            L4_demand = 0
        else:
            L2_demand = logistic(num_deliver_2[t] / (L2_DC[t] * BL_Capacity_factor))
            L4_demand = logistic(num_deliver_4[t] / (L4_DC[t] * BL_Capacity_factor))

        neg_HO_t = sum(neg_HO[0:t+1,:])
        if neg_HO_t[0] == 0:
            L2_4_health_outcomes = 0
        else:
            L2_4_health_outcomes = logistic([
                neg_HO_t[1] / neg_HO_t[0],
                neg_HO_t[2] / neg_HO_t[0] ])

        P_P_target    = 0.8
        P_A_target    = (P_M[t] * logistic([Visibility,1]) + P_I[t]) / 2
        P_D_target    = 0.7
        P_DP_target   = 0.7
        P_M_target    = 0.7
        P_I_target    = 0.6
        P_SP_target   = (P_P[t] + P_A[t] + P_D[t] * logistic([Visibility,1])) / 3
        dP_SP_in      = (P_P[t] + P_A[t] + P_D[t])
        dP_A_in       = (P_M[t] + P_I[t])

        P_RR_target   = 1.0 * logistic([Funding_MNCH, Prioritization_MNCH, -Delayed_disbursement,3])
        dP_RR_in      = P_DP[t] + P_SP[t]

        L2_HF_target  = 0.9 * P_RR[t] * logistic([Adherence_budget,2])
        L4_HF_target  = 0.8 * P_RR[t] * logistic([Adherence_budget,2])
        S_TF_target   = 0.8 * P_RR[t] * logistic([Adherence_budget,2])
        dL2_HF_in     = P_RR[t]  # coefficients of these three dStock_in terms add up to 1
        dL4_HF_in     = P_RR[t]
        dS_TF_in      = P_RR[t]
        # dP_RR_out = dL2_HF_in + dL4_HF_in + dS_TF_in

        L2_target_0   = 0.9 * L2_HF[t]
        L2_HR_target  = L2_target_0 * logistic([Employee_incentives, -Lack_promotion,3])
        L2_S_target   = L2_target_0 * logistic([Lack_action_depletion, -L2_demand,2])
        dL2_HR_in     = L2_HF[t]
        dL2_S_in      = L2_HF[t]
        # dL2_HF_out = dL2_HR_in + dL2_S_in
        L4_target_0   = 0.9 * L4_HF[t]
        L4_HR_target  = L4_target_0 * logistic([Employee_incentives, -Lack_promotion,3])
        L4_S_target   = L4_target_0 * logistic([Lack_action_depletion, -L4_demand,2])
        dL4_HR_in     = L4_HF[t]
        dL4_S_in      = L4_HF[t]
        # dL4_HF_out = dL4_HR_in + dL4_S_in
        S_FR_target  = 0.7 * S_TF[t] * logistic([Employee_incentives, -Lack_promotion,2])
        S_T_target   = 0.9 * S_TF[t] * logistic([Employee_incentives, -Lack_promotion,2])
        dS_FR_in     = S_TF[t]
        dS_T_in      = S_TF[t]
        # dS_TF_out  = dS_FR_in + dS_T_in

        L2_DC_target  = 0.1
        L4_DC_target  = 0.9
        dL2_DC_in     =  0.1 * S_FR[t] # target < stock so need to reverse sign here
        dL4_DC_in     =  0.1 * S_FR[t]

        L2_Q_target  = (L2_HR_target + L2_S_target) / 2 / L2_target_0 * logistic([-9*L2_demand,5])
        L4_Q_target  = (L4_HR_target + L4_S_target) / 2 / L4_target_0 * logistic([-9*L4_demand,5])
        dL2_Q_in     = (L2_HR[t] + L2_S[t])
        dL4_Q_in     = (L4_HR[t] + L4_S[t])

        # Stock[t + 1] = Stock[t] * (1 + beta * (dStock_in - dStock_out)) * (Stock_target - Stock[t])
        y_t_list = []
        for idx,name in S_idx_names:
            d_name = 'd' + name
            y_t_list.append(eval(name + '[t]'))
            globals()[name][t+1] = \
                 eval(name + '[t] * (1 + beta * (' + d_name + '_in - ' + d_name + '_out)' \
                                             '* (' + name + '_target - ' + name + '[t]))')

        y_t[t+1,:] = np.array(y_t_list)
        # quality = (L2_Q[t+1] + L4_Q[t+1]) / 2
        l2_quality = L2_Q[t+1]
        l4_quality = L4_Q[t+1]
        P_D[t+1]   = L2_4_health_outcomes

        # neg_home   = neg_H0[t+1,2]

        fac_t  = np.array(facilities[t])
        fac_t1 = sum(fac_t == 1)
        # fac_t2 = sum(fac_t == 2)
        for mother in mothers:
            fac = np.array(facility)
            fac1 = sum(fac == 1)
            den = (L2_DC[t]*BL_Capacity_factor)
            mother.increase_age(l4_quality, l2_quality, proximity, L2_4_health_outcomes,
                1 - (sum(fac == 1) - fac_t1) / (L2_DC[t]*BL_Capacity_factor),
                None) # don't need the last argument
                # 1 - (sum(fac == 2) - fac_t2) / (L4_DC[t] * BL_Capacity_factor))
            # mother.increase_age(quality, proximity)
            gest_age.append(mother._gest_age) # done
            health.append(float(mother._health)) # done
            anc.append(mother._anc) # done
            delivery.append(mother._delivery)
            facility.append(mother._facility)

        gest_age_t[t+1] = gest_age
        health_t[t+1]   = health
        anc_t[t+1]      = anc
        deliveries[t+1] = delivery # dictionary with integer keys, can access using [idx] for idx>0
        facilities[t+1] = facility

        fac_t1 = np.array(facilities[t+1])
        fac_t  = np.array(facilities[t])
        num_deliver_home[t+1] = sum(fac_t1 == 0) - sum(fac_t == 0)
        num_deliver_2[t+1]    = sum(fac_t1 == 1) - sum(fac_t == 1)
        num_deliver_4[t+1]    = sum(fac_t1 == 2) - sum(fac_t == 2)
        num_deliver_total[t+1]= num_deliver_home[t+1] + num_deliver_2[t+1] + num_deliver_4[t+1]

        del_t1 = np.array(deliveries[t+1])
        del_t  = np.array(deliveries[t])
        for k in range(3):
            pos_HO[t+1,2-k] = sum((del_t1 ==  1) & (fac_t1 == k)) - sum((del_t ==  1) & (fac_t == k))
            neg_HO[t+1,2-k] = sum((del_t1 == -1) & (fac_t1 == k)) - sum((del_t == -1) & (fac_t == k))

    return t_all, y_t, [num_deliver_4,num_deliver_2,num_deliver_home,num_deliver_total], [pos_HO,neg_HO]

# FOR OTHER PLOTTING METHODS
# gest_age_t = pd.DataFrame.from_dict(gest_age_t)
# health_t = pd.DataFrame.from_dict(health_t)
# anc_t = pd.DataFrame.from_dict(anc_t)
# deliveries = pd.DataFrame.from_dict(deliveries)
# facilities = pd.DataFrame.from_dict(facilities)

def logistic(x):
    return 1 / (1 + np.exp(-np.mean(x)))

def f(y, t, parameters): # 12 variables
    S_RR[0], S_R2[0], S_R4[0], S_RS[0], S_PDS[0] = y

    if t>parameters['t_change']:
        for i in range(len(F_0)):
            F_0[i][0] = F_change[i]
    else:
        for i in range(len(F_0)):
            F_0[i][0] = F_original[i]

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
            beta = parameters['beta']
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

# Bundle parameters for ODE solver
parameters = {
    't_change':20.0,
    'beta':20.0
}

# DASHBOARD
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP,external_stylesheets[0]])
app.config.suppress_callback_exceptions = True

colors = {
    'background': '#111111',
    'text': '#1f77b4'  # dark blue
}

def slider_markers(start=0, end=1, step=0.1, red=None):
    nums = [int(num) if isinstance(num, int) or num%0.01 == 0 or num >= 1 else round(num,2) for num in np.arange(start, end+0.0000001, step)] 
    marks = {num: {'label' : str(num), 'style': {'fontSize': 14}} for num in nums}
    if red is not None:
        if isinstance(red, int) or red%0.01 == 0 or red >= 1:
            marks[int(red)] = {'label' : str(int(red)), 'style': {'fontSize': 14, 'color': '#f50'}}
        else:
            marks[red] = {'label' : str(red), 'style': {'fontSize': 14, 'color': '#f50'}}
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
            step=None
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
        dbc.Col([
            dcc.Graph(id='plot_1a',config={'displayModeBar': False})
        ],width=3),
        dbc.Col([
            dcc.Graph(id='plot_1b',config={'displayModeBar': False})
        ],width=3),
        dbc.Col([
            dcc.Graph(id='plot_1c',config={'displayModeBar': False})
        ],width=3),
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
                        make_slider(i,S_label[i],'S_slider',S_0[i]) for i in range(6,12)
                        # make_slider(i, S_label[i], 'S_slider', S_0[i]) for i in range(6, len(S_0))
            ]),
                ],width=6),
            ]),
        ],width=3),
    ],className="pretty_container"),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='plot_2a', config={'displayModeBar': False})
        ], width=3),
        dbc.Col([
            dcc.Graph(id='plot_2b', config={'displayModeBar': False})
        ], width=3),
        dbc.Col([
            dcc.Graph(id='plot_2c', config={'displayModeBar': False})
        ], width=3),
        dbc.Col([
            # dbc.Col(html.H5('Initial Stock Values'), width=12),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        make_slider(i, S_label[i], 'S_slider', S_0[i]) for i in range(12, 18)
                    ]),
                ], width=6),
                dbc.Col([
                    html.Div([
                        make_slider(i, S_label[i], 'S_slider', S_0[i]) for i in range(18, len(S_0))
                    ]),
                ], width=6),
            ]),
        ], width=3),
    ], className="pretty_container"),

    # dbc.Row([
    #     dbc.Col(html.H5('Sensitivity'),width=3),
    #     dbc.Col([
    #         html.Button(
    #             F_label[11],
    #             id={'type':'sensitivity_variable','index':11},
    #             n_clicks=0,
    #         ),
    #     ],width=3),
    #     dbc.Col([
    #         html.Div('',id={'type':'sensitivity_output','index':11}),
    #     ], width=3),
    # ],  className="pretty_container"),

    dbc.Row([
        dbc.Col(html.H5('Factors'),width=12),
        F_sliders,
        ],className="pretty_container"
    ),

    dbc.Row([
        dbc.Col(html.H5('Run parameters'), width=3),
        dbc.Col([
            html.Div([
                make_slider(0, 't_change', 'P_slider', parameters['t_change'], 0, 100)
            ]),
        ], width=3),
        dbc.Col([
            html.Div([
                make_slider(1, 'beta', 'P_slider', parameters['beta'], 0, 100)
            ]),
        ], width=3),
    ], className="pretty_container"),
])

# S_values_global = np.array(S_0) # used for sensitivities
# F_values_global = np.array(F_0)

def calc_y_old(S_values,F_values,P_values):
    for i in range(len(S_values)):
        y_0[i] = S_values[i] 
    for i in range(len(F_values)):
        F_change[i] = F_values[i] 
    parameters['t_change'] = P_values[0]
    parameters['beta'] = P_values[1]
    # Call the ODE solver
    # y_t = odeint(f, y_0, t, args=(parameters,))
    # y_t = Euler(f,y_0,t,parameters)
    # return y_t

@app.callback(
    dash.dependencies.Output('plot_1a', 'figure'),
    dash.dependencies.Output('plot_1b', 'figure'),
    dash.dependencies.Output('plot_1c', 'figure'),
    dash.dependencies.Output('plot_2a', 'figure'),
    dash.dependencies.Output('plot_2b', 'figure'),
    dash.dependencies.Output('plot_2c', 'figure'),
    [Input({'type':'S_slider','index':ALL}, 'value'),
     Input({'type':'F_slider','index':ALL}, 'value'),
     Input({'type':'P_slider','index':ALL}, 'value'),]
)
def update_graph(S_values,F_values,P_values):
    # S_values_global = np.array(S_values) # used for sensitivities
    # F_values_global = np.array(F_values)
    t_all, y_t, num_d, pos_neg_HO = calc_y(S_values,F_values,P_values)
    # num_deliver_home, num_deliver_2, num_deliver_4, num_deliver_total = num_d[0], num_d[1], num_d[2], num_d[3]
    k_range_1A  = range(0,6)
    k_range_1B  = range(6,11)
    k_range_1C  = range(11,16)
    k_range_2A  = range(16,len(S_label))

    fig_1A = {
        'data':[{
            'x': t_all,
            'y': y_t[:,k],
            'name': S_label[k]
        } for k in k_range_1A],
        'layout': {
            'title':  'POLICY',
            'xaxis':{'title':'Time (months)'},
            'yaxis':{'range':[0,.5], 'title':'Stocks (normalized units)'}
        }
    }

    fig_1B = {
        'data':[{
            'x': t_all,
            'y': y_t[:,k],
            'name': S_label[k]
        } for k in k_range_1B],
        'layout': {
            'title':  'FINANCE',
            'xaxis':{'title':'Time (months)'},
            'yaxis':{'range':[0,0.8], 'title':'Stocks (normalized units)'}
        }
    }

    fig_1C = {
        'data':[{
            'x': t_all,
            'y': y_t[:,k],
            'name': S_label[k]
        } for k in k_range_1C],
        'layout': {
            'title':  'FACILITIES',
            'xaxis':{'title':'Time (months)'},
            'yaxis':{'range':[0,0.5], 'title':'Stocks (normalized units)'}
        }
    }

    fig_2A = {
        'data':[{
            'x': t_all,
            'y': y_t[:,k],
            'name': S_label[k]
        } for k in k_range_2A],
        'layout': {
            'title':  'QUALITY',
            'xaxis':{'title':'Time (months)'},
            'yaxis':{'range':[0,1], 'title':'Stocks (normalized units)'}
        }
    }

    num_deliveries_labels = ['Level 4/5','Level 2/3','Home','Total'] # total is unused
    fig_2B = {
        'data':[{
            'x': t_all,
            'y': num_d[k],
            'name': num_deliveries_labels[k]
        } for k in range(3)], # don't need total, so just the first three
        'layout': {
            'title':  'Deliveries over time',
            'xaxis':{'title':'Time (months)'},
            'yaxis':{'range':[0,40], 'title':'Deliveries'}
        }
    }

    HO_labels = ['Home delivery','L2', 'L4']
    fig_2C = {
        'data':[{
            'x': t_all,
            'y': pos_neg_HO[1][:,k],
            'name': HO_labels[k]
        } for k in range(3)],
        'layout': {
            'title':  'Negative birth outcomes over time',
            'xaxis':{'title':'Time (months)'},
            'yaxis':{'range':[0,10], 'title':'Number of dyads'}
        }
    }

    return fig_1A, fig_1B, fig_1C, fig_2A, fig_2B, fig_2C

# SENSITIVITY (NOT NEEDED)
# delta = 0.1
# @app.callback(
#     Output({'type':'sensitivity_output','index':MATCH},'children'),
#   [ Input({'type':'sensitivity_variable','index':MATCH},'n_clicks') ],
#   [ State({'type':'sensitivity_variable','index':MATCH},'id') ]
# )
# def calc_sensitivity(n,id_match):
#     i = id_match['index']
#     F_values_delta = F_values_global
#     F_values_delta[i] = F_values_global[i] + 0.01
#     P_values = np.array([ parameters['t_change'], parameters['beta']])
#     y_t      = calc_y(S_values_global,F_values_global,P_values)
#     y_t_delta= calc_y(S_values_global,F_values_delta, P_values)
#     Dy = (y_t_delta[-1] - y_t[-1])/delta
#     return Dy[0]

# CAN LEAVE IN FOR PYTHONEVERYWHERE
if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run_server(debug=True,dev_tools_ui=False)
