import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc # conda install -c conda-forge dash-bootstrap-components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pcolors
import random

from Mother import Mother_simplified, get_prob_logit_health
from Dash_ODE_Methods import logistic, many_sliders, make_slider

# CHANGES:
#   Added probs
#   Predisp_L2_L4: 1.0
#   Plot colors, dash
#   Don't use 2-k in HO array
#   Calculate HO array at end of loop, use t instead of t+1 for quality before Mothers loop
#   Use -9 for Mother self._gest_age, include logit_initial for delivery logit values
#   Reorder S_info HR_Readiness, increase HR_Readiness_4/5 and L4_HF_target_0
#   P_D[0] = logistic(logit_)
#   beta
#   time window

random.seed(10)
# INITIALIZE DATA FOR CLASS MOTHER
# read file of data with columns 1-4 [Mother]: wealth, education, age, number of children
# simulate for health, gestational age, predisposition for ANC
df = pd.read_excel("SDR_Mother.xlsx")
wealth      = df['Wealth'].to_numpy()
education   = df['Education'].to_numpy()
age         = df['Age'].to_numpy()
no_children = df['No_children'].to_numpy()

n_repeat = 1 # n_repeat greater than 1 makes the code run slow, it is better to repeat in the Excel file
for i in range(n_repeat):
    wealth = wealth.repeat(n_repeat)
    education = education.repeat(n_repeat)
    age = age.repeat(n_repeat)
    no_children = no_children.repeat(n_repeat)
    wealth = wealth.repeat(n_repeat)

no_mothers  = len(age)

# quality   = 0.7

proximity = 0.8

def set_variables(X_info, name_append = '', nt=0):
    X_names, X_0, X_label, X_idx_names = [], [], [], []
    for i in range(len(X_info)):
        name = X_info[i][0]            # name of the variable is set, e.g. name = 'P_P_target'
        X_names.append(name)           # C_names = ['P_P_target',...]
        X_idx_names.append([i, name])  # C_idx_names = [[0,'P_P_target'],[1,P_D_target],...]
        X_0.append(X_info[i][1])       # C_0 = [0.8, 0.7, ...] hard-coded initial value used in sliders
        X_label.append(X_info[i][2])   # C_label = ['Political_Goodwill_target',...] for sliders
        globals()[name + name_append] = X_0[i]  # P_P_target = C_0[0] = 0.8 for model and sliders
        if nt > 0:                         # S_P_P_0 = X_0[0] for stocks
            globals()[name] = np.zeros(nt) # S_P_P = np.zeros(nt) for stocks

    return X_names, X_0, X_label, X_idx_names


# APP SETUP

# Make time array for solution
nt = 12*5 # number of months
# RR  Resources for RMNCH
# R2  Healthcare financing (Resources for 2/3 facilities)
# R4  Healthcare financing (Resources for 4/5 facilities)
# RS  SDR transition financing
# PDS Policy: Development support

# Names, initial values
S_info = [
    ['P_RR',  0.2, 'Resources_RMNCH'            ],
    ['L4_HF', 0.2, 'HC_Financing_4/5'           ], #
    ['L2_HF', 0.2, 'HC_Financing_2/3'           ], #
    ['S_FR',  0.2, 'SDR_Facility_Repurposing'   ],
    ['L4_HR', 0.3, 'HR_Readiness_4/5'           ],
    ['L2_HR', 0.2, 'HR_Readiness_2/3'           ],
    ['L4_DC', 0.4, 'Delivery_Capacity_4/5'      ],
    ['L2_DC', 0.9, 'Delivery_Capacity_2/3'      ],
    ['P_D',   0.6, 'Performance_Data'           ],
    ['L4_Q',  0.5, 'Quality_4/5'                ],
    ['L2_Q',  0.4, 'Quality_2/3'                ], #
]

S_names, S_0, S_label, S_idx_names = set_variables(S_info, nt=nt)

# FACTORS
FP_info = [
    ['Funding_MNCH',          0.2, 'Funding_MNCH'         ],
    ['Adherence_budget',      0.2, 'Adherence_budget'     ],
    ['Employee_incentives',   0.2, 'Employee_incentives'  ],
    ['Strong_referrals',      0.2, 'Strong_referrals'     ],
    ['Training_incentives',   0.2, 'Training_incentives'  ],
]

FP_combination_info = [
    ['INCREASE_ALL_P', 0.0, 'INCREASE_ALL_P'],
    ['INCREASE_ALL_P', 0.0, 'INCREASE_SOME_P'],
]

FP_names, FP_0, FP_label, FP_idx_names = set_variables(FP_info)
FP_combination_names, FP_combination_0, FP_combination_label, FP_combination_idx_names = \
    set_variables(FP_combination_info)

B_info = [
    ['BL_Capacity_factor',        20, 'BL_Capacity_Factor'                ],
    ['Initial_Negative_Predisp',   2, 'Initial_Negative_Predisp'],
    ['Health_outcomes__Predisp', 3.2, 'Health outcome -> Predisp hospital'], #
    ['L4_Q__Predisp',            0.5, 'L4/5 quality -> Predisp hospital'  ],
    ['Predisp_L2_L4',            2.0, 'Predisp L2'], #
    ['Health_const_0',           0.8, 'Initial Health'],
    ['Health_slope_0',           0.2, 'Variability Initial Health'],#
    ['Q_Health_multiplier',             4., 'Q Health Effect'], #
    ['Q_Health_L4_constant',            1.5, 'L4 Health Effect'],
    ['Q_Health_L4_L2_difference',        1., 'Difference L4-L2 Health Effect'],
    ['Q_Health_L4_referral_difference', 0.5, 'L2 -> L4 Referral Health Effect'],
    ['Q_Health_Home_negative',          3.0, 'Home Delivery Health Effect'], #
    ['Time_delay_awareness',           24.0, 'Awareness Delay (months)'], #
]

B_names, B_0, B_label, B_idx_names = set_variables(B_info)

# MODEL PARAMETER INFORMATION FOR SLIDERS
C_info = [
    ['P_D_target',     0.7, 'Data Performance target'],
    ['L4_HF_target_0', 0.8, 'L4/5 HC Financing target'],
    ['L2_HF_target_0', 0.6, 'L2/3 HC Financing target'],
    ['L4_target_0',    0.9, 'L4 target'],
    ['L2_target_0',    0.9, 'L2 target'],
    ['S_FR_target_0',  0.7, 'SDR FR target'],
    ['S_T_target_0',   0.9, 'SDR T target'],
    ['dL4_DC_in_0',    0.2, 'L4 Rate of Capacity Increase'],
    ['dL2_DC_in_0',    0.2, 'L2 Rate of Capacity Decrease'],
    ['P_RR_target_0',  1.0, 'Resources RMNCH target'],
    ['L4_DC_target',   0.9, 'L4 Delivery Capacity Target'],
    ['L2_DC_target',   0.1, 'L2 Delivery Capacity Target'],
]

# SET UP OTHER INTERMEDIATE PYTHON VARIABLES FOR THE MODEL AND SLIDERS
C_names, C_0, C_label, C_idx_names = set_variables(C_info)

def set_F_change(F_0):
    F_original = np.zeros(len(F_0)) # original values
    F_change   = np.zeros(len(F_0)) # changed values based on sliders
    for i in range(len(F_0)):
        F_original[i] = F_0[i] # get the hard-coded value, originally from F_info
        F_change[i]   = F_0[i] # initialize (don't copy the object, just the value)
    return F_original, F_change

FP_original, FP_change = set_F_change(FP_0)

# Parameters for ODE solver
parameters = {
    't_change' :  0.0,
    'beta' : 1.0
}

y_0 = S_0

def get_factors_0():
    return FP_0

def calc_y(S_values, FP_values, B_values, C_values, P_values): # values from the sliders
    # P_values = parameter values
    for i in range(len(FP_values)): # for each F-slider
        FP_change[i] = FP_values[i] # F-parameter that is collected from the slider

    parameters['t_change'] = P_values[0] # slider value for time when the parameters change
    parameters['beta']     = P_values[1] # slider value for beta

    beta  = parameters['beta']
    y_t   = np.zeros((nt,len(S_values)))
    t_all = np.zeros(nt)
    anc_t, health_t, gest_age_t, deliveries, facilities = {0:[0]}, {0:[0]}, {0:[0]}, {0:[0]}, {0:[0]}
    # anc_t, health_t, gest_age_t, deliveries, facilities = [[]]*nt, [[]]*nt, [[]]*nt, [[]]*nt, [[]]*nt # NG
    num_deliver = np.zeros((nt,4))
    pos_HO, neg_HO, L2_D_Capacity, L4_D_Capacity = np.zeros([nt,4]), np.zeros([nt,4]), np.ones(nt), np.ones(nt)
    probs = np.ones((nt, 10))

    for i in range(len(S_values)):
        y_t[0,i] = S_values[i]

    for idx,name in S_idx_names:
        globals()[name][0] = S_values[idx]

    B = {} # Use a dictionary so that we only need to pass B to the Mother class
    for idx,name in B_idx_names:
        B[name] = B_values[idx]         # B['Health_outcomes__Predisp'] = 2.4
        globals()[name] = B_values[idx] # Health_outcomes__Predisp = 2.4

    for idx,name in C_idx_names:
        globals()[name] = C_values[idx] # P_P_target = 0.8

    mothers = []

    for mother in range(0, no_mothers):
        mothers.append(Mother_simplified(nt, B))

    # OTHER MISCELLANEOUS FACTORS
    L4_D_Capacity_Multiplier = 2

    # INITIAL PROBABILITIES
    L2_4_health_outcomes_0 = 0  # initialize with low opinion of health outcomes
    prob_l4_0, prob_l2_0, logit_health_l4_0, logit_health_l2_0, logit_health_l4_l2_0, logit_health_l0_0, _ = \
        get_prob_logit_health(B, L4_Q[0], L2_Q[0], L2_4_health_outcomes_0, B['Health_const_0'])
    P_D[0] = logistic(logit_health_l0_0) # B['Health_const_0'] - B['Q_Health_Home_negative']

    # LOOP OVER EVERY TIME VALUE
    for t in range(0,nt-1):
        if t > parameters['t_change']: # IF TIME IS LARGER THAN THE t_change SLIDER VALUE
            for idx, name in FP_idx_names:
                globals()[name] = FP_change[idx] # then use the SLIDER value for the F-parameter, e.g., Visibility = 0.0
        else:                                   # otherwise
            for idx, name in FP_idx_names:
                globals()[name] = FP_original[idx] # use the HARD-CODED value for the F-parameter saved in F_info

        t_all[t+1] = t_all[t] + 1 # increment by month
        gest_age, health, anc, delivery, facility = [], [], [], [], []
        for idx,name in S_idx_names:
            d_name = 'd' + name
            globals()[d_name + '_in'] = 0.0
            globals()[d_name + '_out'] = 0.0

        # if t == 0:
        #     L2_demand = 0
        #     L4_demand = 0
        # else:
        L2_D_Capacity[t] = L2_DC[t] * BL_Capacity_factor
        L4_D_Capacity[t] = L4_DC[t] * BL_Capacity_factor * L4_D_Capacity_Multiplier
        L2_demand = logistic(num_deliver[t,1] / (L2_D_Capacity[t]))
        L4_demand = logistic(num_deliver[t,2] / (L4_D_Capacity[t]))

        P_RR_target   = P_RR_target_0 * logistic([Funding_MNCH, 3])
        dP_RR_in      = P_D[t]

        L2_HF_target  = L2_HF_target_0 * P_RR[t] * logistic([Adherence_budget, 2])
        L4_HF_target  = L4_HF_target_0 * P_RR[t] * logistic([Adherence_budget, 2])
        dL2_HF_in     = P_RR[t]  # coefficients of these three dStock_in terms add up to 1
        dL4_HF_in     = P_RR[t]
        # dP_RR_out = dL2_HF_in + dL4_HF_in + dS_TF_in
        L2_target_combined_0 = L2_target_0 * L2_HF[t] # combined targets of L2_HR and L2_S =0.9*target of L2_HF
        L2_HR_target  = L2_target_combined_0 * logistic([Employee_incentives, Strong_referrals, Training_incentives, 3])
        dL2_HR_in     = L2_HF[t]
        # dL2_HF_out = dL2_HR_in + dL2_S_in
        L4_target_combined_0 = L4_target_0 * L4_HF[t]
        # L4_target_combined_0 = L4_target_0
        L4_HR_target  = L4_target_combined_0 * logistic([Employee_incentives, Strong_referrals, Training_incentives, 3])
        dL4_HR_in     = L4_HF[t]
        # dL4_HF_out = dL4_HR_in + dL4_S_in
        S_FR_target  = S_FR_target_0 * P_RR[t] * logistic([Employee_incentives, Strong_referrals, Training_incentives, 3])
        S_T_target   = S_T_target_0 * P_RR[t] * logistic([Employee_incentives, Strong_referrals, Training_incentives, 3])
        dS_FR_in     = P_RR[t]
        dS_T_in      = P_RR[t]
        # dS_TF_out  = dS_FR_in + dS_T_in

        # L2_DC_target  = 0.1
        # L4_DC_target  = 0.9
        dL2_DC_in     =  dL2_DC_in_0 * S_FR[t] # target < stock so need to reverse sign here
        dL4_DC_in     =  dL4_DC_in_0 * S_FR[t]

        L2_Q_target  = L2_HR_target / L2_target_combined_0 * logistic([Strong_referrals, -9*L2_demand,5])
        L4_Q_target  = L4_HR_target / L4_target_combined_0 * logistic([Strong_referrals, -9/2*L4_demand,5])
        dL2_Q_in     = L2_HR[t]
        dL4_Q_in     = L4_HR[t]

        # Stock[t + 1] = Stock[t] * (1 + beta * (dStock_in - dStock_out)) * (Stock_target - Stock[t])
        y_t_list = []
        for idx,name in S_idx_names:
            d_name = 'd' + name
            y_t_list.append(eval(name + '[t]'))
            globals()[name][t+1] = \
                 eval(name + '[t] * (1 + beta * (' + d_name + '_in - ' + d_name + '_out)' \
                                             '* (' + name + '_target - ' + name + '[t]))')

        y_t[t+1,:] = np.array(y_t_list)
        # l2_quality = L2_Q[t+1]
        # l4_quality = L4_Q[t+1]
        # P_D[t+1]   = L2_4_health_outcomes
        l2_quality = L2_Q[t]
        l4_quality = L4_Q[t]
        L2_4_health_outcomes = P_D[t]
        # neg_home   = neg_H0[t+1,2]

        L2_deliveries = 0
        for mother in mothers:
            L2_net_capacity = 1 - (L2_deliveries + 0) / L2_D_Capacity[t] # add 1 to see if one more can be delivered
            mother.increase_age(l4_quality, l2_quality, L2_4_health_outcomes, L2_net_capacity)
            if mother.delivered:
                L2_deliveries += 1
                mother.delivered = False  # reset
            # mother.increase_age(quality, proximity)
            gest_age.append(mother._gest_age) # done
            health.append(float(mother._health)) # done
            delivery.append(mother._delivery)
            facility.append(mother._facility)

        gest_age_t[t+1] = gest_age
        health_t[t+1]   = health
        deliveries[t+1] = delivery # dictionary with integer keys, can access using [idx] for idx>0
        facilities[t+1] = facility

        fac_t1 = np.array(facilities[t+1])
        fac_t  = np.array(facilities[t])
        for k in range(3):
            num_deliver[t+1,k]  = sum(fac_t1 == k) - sum(fac_t == k)
        num_deliver[t+1,3]  = np.sum(num_deliver[t+1,:]) # initialized to zero

        del_t1 = np.array(deliveries[t+1])
        del_t  = np.array(deliveries[t])
        for k in range(3): # fac_t = 0 (home), 1 (L2), 2 (L4)
            pos_HO[t+1,k] = sum((del_t1 ==  1) & (fac_t1 == k)) - sum((del_t ==  1) & (fac_t == k))
            neg_HO[t+1,k] = sum((del_t1 == -1) & (fac_t1 == k)) - sum((del_t == -1) & (fac_t == k))
        pos_HO[t+1,3] = sum(pos_HO[t+1,:3]) # totals
        neg_HO[t+1,3] = sum(neg_HO[t+1,:3])

        if t==nt-2: # last value of t, need to add to these array for plotting
            L2_D_Capacity[t+1] = L2_DC[t+1] * BL_Capacity_factor  # need to fill in the last time value
            L4_D_Capacity[t+1] = L4_DC[t+1] * BL_Capacity_factor * L4_D_Capacity_Multiplier

        # HEALTH OUTCOMES ANALYSIS
        time_window  = int(Time_delay_awareness) # averaging window
        if t+2 < time_window: # time_window = 2, then we have enough if t = 0
            avg_deliveries_window =             np.sum(num_deliver[0:t+2,:], axis=0)      / (t+1) # take average number
#           avg_deliveries_window = np.maximum( np.sum(num_deliver[0:t+2,:], axis=0), 1 ) / (t+1) # take average number
            avg_deliveries_append = np.ones((time_window-t-2,4))*avg_deliveries_window
            num_deliver_window = np.append( avg_deliveries_append,       num_deliver[0:t+2,:], axis=0 )
            neg_HO_window      = np.append( avg_deliveries_append*(1-P_D[0]), neg_HO[0:t+2,:], axis=0 )
        else:
            num_deliver_window = num_deliver[(t+2-time_window):t+2,:]
            neg_HO_window           = neg_HO[(t+2-time_window):t+2,:]

        linearly_decreasing_weights = np.array(range(time_window)).reshape((time_window,1))

        neg_HO_t     = sum(neg_HO_window      * linearly_decreasing_weights) # cumulative negative health outcomes (HO)
        deliveries_t = sum(num_deliver_window * linearly_decreasing_weights)
        # neg_HO_t     = sum(neg_HO[0:t+1,:]) # cumulative negative health outcomes (HO)
        # deliveries_t = np.sum(num_deliver[0:t+1,:], axis=0)
        if np.prod(deliveries_t[:2]) == 0:
            L2_4_health_outcomes = P_D[0] # use home value
        else:
            L2_4_health_outcomes = 1 - sum(neg_HO_t[1:3]) / sum(deliveries_t[1:3]) # don't include neg_HO_t[3] = total neg HO
        L_health_outcomes = np.ones(3) * P_D[0]
        for k in range(3):
            if deliveries_t[k] > 0:
                L_health_outcomes[k] = 1 - neg_HO_t[k] / deliveries_t[k]

        P_D[t+1]   = L2_4_health_outcomes

        prob_l4, prob_l2, logit_health_l4, logit_health_l2, logit_health_l4_l2, logit_health_l0, _ = \
            get_prob_logit_health(B, l4_quality, l2_quality, L2_4_health_outcomes, B['Health_const_0'])

        probs[t+1,:] = np.append(
                            np.array([prob_l4, prob_l2*(1-prob_l4), 1-prob_l4-prob_l2*(1-prob_l4),
                                logistic(logit_health_l4), logistic(logit_health_l2),
                                logistic(logit_health_l4_l2), logistic(logit_health_l0)]),
                            [ L_health_outcomes[2], L_health_outcomes[1], L_health_outcomes[0] ]) # plot order

    return t_all, y_t, \
           [ num_deliver[:,2], num_deliver[:,1], num_deliver[:,0], num_deliver[:,3], L4_D_Capacity, L2_D_Capacity ],\
           [ pos_HO, neg_HO ], probs

# DASHBOARD
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP,external_stylesheets[0]])
app.config.suppress_callback_exceptions = True

colors = {
    'background': '#111111',
    'text': '#1f77b4'  # dark blue
}

FP_sliders = many_sliders(FP_label,'FP_slider',FP_0,np.zeros(len(FP_0)),
                          np.array(FP_0)*4, num_rows=3, num_cols=2, width=6) # add 0 to 1 slider for INCREASE ALL
FP_combination_sliders = many_sliders(FP_combination_label,'FP_combination_slider',FP_combination_0,
                                      np.zeros(len(FP_combination_0)),np.ones(len(FP_combination_0)),
                                      num_rows=1, num_cols=2, width=6)
B_sliders = many_sliders(B_label,'B_slider',B_0,np.zeros(len(B_0)),np.array(B_0)*4, num_rows=4, num_cols=4, width=3)
# many_sliders(labels, type used in Input() as an identifier of group of sliders, initial values, min, max, ...
C_sliders = many_sliders(C_label,'C_slider',C_0,np.zeros(len(C_0)),np.ones(len(C_0)), num_rows=3, num_cols=4, width=3)
fcolor = '#316395'  # font
fsize = 36  # font size

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
            dbc.Col(html.H5('Initial Stock Values', style={'fontSize':fsize, 'color':fcolor}),width=12),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        make_slider(i,S_label[i],'S_slider',S_0[i]) for i in range(0,6)
                    ]),
                ],width=6),
                dbc.Col([
                    html.Div([
                        make_slider(i,S_label[i],'S_slider',S_0[i]) for i in range(6,len(S_0))
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
            dbc.Col([html.H5('Positive Combination Factors', style={'fontSize':fsize, 'color':fcolor}),FP_combination_sliders],width=12),
            dbc.Col([html.H5('Positive Factors', style={'fontSize':fsize, 'color':fcolor}),FP_sliders],width=12),
        ], width=3)
    ],className="pretty_container"),

    dbc.Row([
        dbc.Col(html.H5('Beta coefficients', style={'fontSize':fsize, 'color':fcolor}),width=12),
        B_sliders,
        ],className="pretty_container"
    ),

    dbc.Row([
        dbc.Col(html.H5('Constant coefficients for the model', style={'fontSize':fsize, 'color':fcolor}),width=12),
        C_sliders,
        ],className="pretty_container"
    ),

    dbc.Row([
        dbc.Col(html.H5('Meta parameters', style={'fontSize':fsize, 'color':fcolor}), width=3),
        dbc.Col([
            html.Div([
                make_slider(0, 'Time when factor changes will take place', 'P_slider', parameters['t_change'], 0, nt)
            ]),
        ], width=3),
        dbc.Col([
            html.Div([
                make_slider(1, 'Common coefficient for rate of change', 'P_slider', parameters['beta'], 0, 2.0)
            ]),
        ], width=3),
    ], className="pretty_container"),
])

# S_values_global = np.array(S_0) # used for sensitivities
# F_values_global = np.array(F_0)

def update_colors(F_clicks, F_styles, F_combination_styles):
    idx = 0
    F_combination_styles[1]['color'] = '#000' # default color if no F_slider_buttons are selected
    for click in F_clicks:
        if click % 2 == 1:
            F_combination_styles[1]['color'] = '#f50'
            F_styles[idx]['color'] = '#f50'
        else:
            F_styles[idx]['color'] = '#000'
        idx += 1
    return F_styles, F_combination_styles

@app.callback(
    dash.dependencies.Output({'type': 'FP_slider_button', 'index': ALL}, 'style'),
    dash.dependencies.Output({'type': 'FP_combination_slider_button', 'index': ALL}, 'style'),
    [Input({'type': 'FP_slider_button', 'index': ALL}, 'n_clicks'),],
    [State({'type': 'FP_slider_button', 'index': ALL}, 'style'),
     State({'type': 'FP_combination_slider_button', 'index': ALL}, 'style'),],
)
def update_labels(FP_clicks, FP_styles, FP_combination_styles):
    return update_colors(FP_clicks, FP_styles, FP_combination_styles)

@app.callback(
    dash.dependencies.Output({'type': 'FP_slider', 'index': ALL}, 'value'),  # simple trial
    [Input({'type': 'FP_combination_slider', 'index': ALL}, 'value'),],
    [State({'type': 'FP_slider', 'index': ALL}, 'value'),
     State({'type': 'FP_slider', 'index': ALL}, 'max'),
     State({'type': 'FP_slider_button', 'index': ALL}, 'style'),]
)
def update_combination_slider(FP_combination_values, FP_values, FP_max, FP_style):
    FP_0 = get_factors_0()
    def update_F_values(F_values, F_0, F_max, F_style, F_combination_values):
        F_values = np.array(F_values)
        F_max    = np.array(F_max)
        F_values      = F_0 + (F_max - F_0) * F_combination_values[0]
        F_some_values = F_0 + (F_max - F_0) * F_combination_values[1]
        idx = 0
        for style in F_style:
            if (style['color'] != '#000'): # not black
                F_values[idx] = F_some_values[idx]
            idx += 1
        return list(F_values)

    return update_F_values(FP_values, FP_0, FP_max, FP_style, FP_combination_values)

@app.callback(
    dash.dependencies.Output('plot_1a', 'figure'), # component_id='plot_1a', component_property='figure'
    dash.dependencies.Output('plot_1b', 'figure'),
    dash.dependencies.Output('plot_1c', 'figure'),
    dash.dependencies.Output('plot_2a', 'figure'),
    dash.dependencies.Output('plot_2b', 'figure'),
    dash.dependencies.Output('plot_2c', 'figure'),
    # each row is passed to update_graph (update the dashboard) as a separate argument in the same
    [Input({'type':'S_slider','index':ALL}, 'value'), # get all S-slider values, pass as 1st argument to update_graph()
     Input({'type':'FP_slider','index':ALL}, 'value'),
     Input({'type':'B_slider','index':ALL}, 'value'),
     Input({'type':'C_slider','index':ALL}, 'value'),
     Input({'type':'P_slider','index':ALL}, 'value'),],
    [State({'type':'FP_slider','index':ALL}, 'max'),],
)
def update_graph(S_values,FP_values,B_values,C_values,P_values,FP_max): # each argument is one of Input(...)
    # S_values_global = np.array(S_values) # used for sensitivities
    # F_values_global = np.array(F_values)
    # SLIDER VALUES GETS PASSED TO THE MODEL TO COMPUTE THE MODEL RESULTS (e.g., y_t = stocks over time)
    t_all, y_t, num_d, pos_neg_HO, probs = calc_y(S_values,FP_values,B_values,C_values,P_values)
    # num_deliver_home, num_deliver_2, num_deliver_4, num_deliver_total, L2_D_Capacity, L4_D_Capacity = num_d

    k_range_1B  = range(0,4)
    k_range_1C  = range(4,8)
    k_range_2A  = range(8,len(S_label))
    def y_max_t(k_range, y, increments=5):
        if isinstance(y,list):
            max_y = 0
            for k in k_range:
                max_y = max(max_y, max(y[k]))
        else:
            max_y = np.amax(np.array(y)[:, k_range])

        return np.ceil(increments * max_y) / increments

    colors = pcolors.qualitative.D3
    fcolor = '#316395' # font
    fsize  = 24 # font size

    # PROBABILITIES
    labels     = ['Deliver at 4/5','Deliver at  2/3','Deliver at Home','Healthy at 4/5','Healthy at 2/3',
                  'Healthy at 2/3 -> 4/5','Health Home',
                  'Deliver at 4/5 (actual)','Deliver at 2/3 (actual)','Deliver at Home (actual)']
    line_color = [0, 1, 2, 0, 1, 4, 2, 0, 1, 2]
    line_dash  = ['none', 'none', 'none', 'dash', 'dash', 'dash', 'dash', 'dot', 'dot', 'dot' ]
    fig_1A = {
        'data':[{
            'x': t_all[1:],
            'y': probs[1:,k],
            'name': labels[k],
            'line' : {'color' : colors[line_color[k]], 'dash' : line_dash[k]},
        } for k in range(len(labels))], # don't need total, so just the first three
        'layout': {
            'title': {'text' : 'Probabilities over time', 'font' : {'color' : fcolor, 'size' : fsize}},
            'xaxis':{'range':[0,nt], 'title':'Time (months)'},
            'yaxis':{'title':'Probabilities'}
        }
    }

    # QUALITY
    line_color = [4, 0, 1, 3]
    line_dash  = ['dash', 'none', 'none', 'dash']
    fig_1B = {
        'data':[{
            'x': t_all,
            'y': y_t[:,k],
            'name': S_label[k],
            'line': {'color': colors[line_color[k]], 'dash': line_dash[k]},
        } for k in k_range_1B],
        'layout': {
            'title': {'text' : 'Resources', 'font' : {'color' : fcolor, 'size' : fsize}},
            'xaxis':{'title':'Time (months)'},
            'yaxis':{'range':[0,y_max_t(k_range_1B, y_t)], 'title':'Stocks (normalized units)'}
        }
    }

    # SERVICE READINESS
    line_color = [0, 1, 0, 1]
    line_dash  = ['none', 'none', 'dash', 'dash']
    fig_1C = {
        'data':[{
            'x': t_all,
            'y': y_t[:,k],
            'name': S_label[k],
            'line': {'color': colors[line_color[k-max(k_range_1B)-1]], 'dash': line_dash[k-max(k_range_1B)-1]},
        } for k in k_range_1C],
        'layout': {
            'title': {'text' : 'Service Readiness', 'font' : {'color' : fcolor, 'size' : fsize}},
            'xaxis':{'title':'Time (months)'},
            'yaxis':{'range':[0,y_max_t(k_range_1C, y_t)], 'title':'Stocks (normalized units)'},
        }
    }

    # QUALITY
    line_color = [4, 0, 1]
    line_dash  = ['dash', 'none', 'none']
    fig_2A = {
        'data':[{
            'x': t_all,
            'y': y_t[:,k],
            'name': S_label[k],
            'line': {'color': colors[line_color[k-max(k_range_1C)-1]], 'dash': line_dash[k-max(k_range_1C)-1]},
        } for k in k_range_2A],
        'layout': {
            'title': {'text' : 'Quality', 'font' : {'color' : fcolor, 'size' : fsize}},
            'xaxis':{'title':'Time (months)'},
            'yaxis':{'range':[0,1], 'title':'Stocks (normalized units)'}
        }
    }

    # DELIVERIES
    labels     = ['Level 4/5','Level 2/3','Home','Total','Capacity 4/5','Capacity 2/3'] # total is unused
    line_color = [0, 1, 2, 5, 0, 1]
    line_dash  = ['none', 'none', 'none', 'dash', 'dash', 'dash' ]
    fig_2B = {
        'data':[{
            'x': t_all[1:],
            'y': num_d[k][1:],
            'name': labels[k],
            'line': {'color': colors[line_color[k]], 'dash': line_dash[k]},
        } for k in [0,1,2,4,5]], # don't need total, so just the first three
        'layout': {
            'title': {'text' : 'Deliveries over time', 'font' : {'color' : fcolor, 'size' : fsize}},
            'xaxis': {'range':[0,nt], 'title':'Time (months)'},
            'yaxis': {'range':[0,y_max_t([0,1,2,4,5], num_d)], 'title':'Deliveries'}
        }
    }

    # QUALITY
    labels     = ['Home','L2', 'L4', 'Total']
    line_color = [2, 1, 0, 4]
    line_dash  = ['none', 'none', 'none', 'dash']
    fig_2C = {
        'data':[{
            'x': t_all[1:],
            'y': pos_neg_HO[1][1:,k],
            'name': labels[k],
            'line': {'color': colors[line_color[k]], 'dash': line_dash[k]},
        } for k in [2,1,0,3]],
        'layout': {
            'title': {'text' : 'Negative birth outcomes', 'font' : {'color' : fcolor, 'size' : fsize}},
            'xaxis':{'range':[0,nt], 'title':'Time (months)'},
            'yaxis':{'range':[0,y_max_t([2,1,0,3], pos_neg_HO[1])], 'title':'Number of dyads'}
        }
    }

    return fig_1A, fig_1B, fig_1C, fig_2A, fig_2B, fig_2C

# CAN LEAVE IN FOR PYTHONEVERYWHERE
if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run_server(debug=True,dev_tools_ui=False)
