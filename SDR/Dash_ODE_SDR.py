import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc # conda install -c conda-forge dash-bootstrap-components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random

from Mother import Mother
from Dash_ODE_Methods import logistic, many_sliders, make_slider

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
    ['S_TF',  0.2, 'SDR_Transition_Financing'   ],

    ['L4_HR', 0.2, 'HR_Readiness_4/5'           ],
    ['L4_S',  0.2, 'Supplies/Infrastructure_4/5'],
    ['L2_HR', 0.2, 'HR_Readiness_2/3'           ],
    ['L2_S',  0.2, 'Supplies/Infrastructure_2/3'],
    ['S_T',   0.2, 'SDR_HR_Training'            ],
    ['S_FR',  0.2, 'SDR_Facility_Repurposing'   ],

    ['L4_DC', 0.4, 'Delivery_Capacity_4/5'      ],
    ['L2_DC', 0.9, 'Delivery_Capacity_2/3'      ],
    ['P_D',   0.6, 'Performance_Data'           ],
    ['L4_Q',  0.5, 'Quality_4/5'                ],
    ['L2_Q',  0.1, 'Quality_2/3'                ],
]

S_names, S_0, S_label, S_idx_names = set_variables(S_info, nt=nt)

# FACTORS
# Names, initial values
FP_info = [
    ['Funding_MNCH',          0.2, 'Funding_MNCH'         ],
    ['Support_Linda_Mama',    0.2, 'Support_Linda_Mama'   ],
    ['Prioritization_MNCH',   0.2, 'Prioritization_MNCH'  ],
    ['Adherence_budget',      0.2, 'Adherence_budget'     ],
    ['Employee_incentives',   0.2, 'Employee_incentives'  ],
    ['Visibility',            0.2, 'Visibility'           ],
    ['Timely_promotions',     0.2, 'Timely_promotions'    ], 
    ['Action_depletion',      0.2, 'Action_depletion'     ], 
    ['Increase_awareness',    0.2, 'Increase_awareness'   ], 
    ['Strong_referrals',      0.2, 'Strong_referrals'     ], 
    ['Training_incentives',   0.2, 'Training_incentives'  ], 
    ['Pos_supply_chain',      0.2, 'Pos_supply_chain'     ], 
    ['Increase_awareness_address_myths', 0.2, 'Increase_awareness_address_myths'],
]

FP_combination_info = [
    ['INCREASE_ALL_P', 0.0, 'INCREASE_ALL_P'],
    ['INCREASE_ALL_P', 0.0, 'INCREASE_SOME_P'],
]

FN_combination_info = [
    ['DECREASE_ALL_N', 0.0, 'DECREASE_ALL_N'],
    ['DECREASE_ALL_N', 0.0, 'DECREASE_SOME_N'],
]

FN_info = [
    ['Delayed_disbursement',  0.2, 'Delayed_disbursement' ],
    ['Lack_promotion',        0.2, 'Lack_promotion'       ],
    ['Lack_action_depletion', 0.2, 'Lack_action_depletion'],
    ['Inadequate_financing',  0.2, 'Inadequate_financing' ],
    ['Lack_adherence_budget', 0.2, 'Lack_adherence_budget'], 
    ['Delay_hiring',          0.2, 'Delay_hiring'         ], 
    ['Frequent_transfer',     0.2, 'Frequent_transfer'    ], 
    ['Burn_out',              0.2, 'Burn_out'             ], 
    ['Poor_management',       0.2, 'Poor_management'      ],
    ['Neg_supply_chain',      0.2, 'Neg_supply_chain'     ], 
]

FP_names, FP_0, FP_label, FP_idx_names = set_variables(FP_info)
FP_combination_names, FP_combination_0, FP_combination_label, FP_combination_idx_names = \
    set_variables(FP_combination_info)
FN_names, FN_0, FN_label, FN_idx_names = set_variables(FN_info)
FN_combination_names, FN_combination_0, FN_combination_label, FN_combination_idx_names = \
    set_variables(FN_combination_info)

B_info = [
    ['L_Capacity_factor',        20, 'BL_Capacity_Factor'                ],
    ['Initial_Negative_Predisp', 2, 'Initial_Negative_Predisp'],
    ['Health_outcomes__Predisp', 2.4, 'Health outcome -> Predisp hospital'],
    ['L4_Q__Predisp',            0.5, 'L4/5 quality -> Predisp hospital'  ],
    ['Health_Predisp',           0.2, 'Health_Predisp -> Predisp hospital'],
    ['Predisp_ANC_const_0',      0.4, 'Predisp_ANC_const_0'],
    ['Predisp_ANC_slope_0',      0.2, 'Predisp_ANC_slope_0'],
    ['Predisp_L2_L4',             4., 'Predisp_L2_L4'],  # 1
    ['Wealth__Predisp',          0.2, 'Wealth__Predisp'],  # 0.2
    ['Education__Predisp',      0.02, 'Education__Predisp'],  # 0.02
    ['Age__Predisp',           0.001, 'Age__Predisp'],  # 0.001
    ['No_Children__Predisp',    0.05, 'No_Children__Predisp'],  # 0.05
    ['Proximity__Predisp',       0.1, 'Proximity__Predisp'],  # 0.1
    ['Health_const_0',           0.8, 'Health_const_0'],
    ['Health_slope_0',           0.2, 'Health_slope_0'],
    ['Q_Health_multiplier',             10., 'Q_Health_multiplier'],  # 6
    ['Q_Health_L4_constant',            1.5, 'Q_Health_L4_constant'],  # 1.5
    ['Q_Health_L4_L2_difference',        1., 'Q_Health_L4_L2_difference'],  # 1
    ['Q_Health_L4_referral_difference', 0.5, 'Q_Health_L4_referral_difference'],  # 0.5
    ['Q_Health_Home_negative',         10.0, 'Q_Health_Home_negative'],  # 0.5
]

B_names, B_0, B_label, B_idx_names = set_variables(B_info)

# MODEL PARAMETER INFORMATION FOR SLIDERS
C_info = [
    ['P_P_target', 0.8, 'Political_Goodwill_target'],
    ['P_D_target', 0.7, 'Data_Performance_target'],
    ['L2_HF_target_0', 0.8, 'L2/3_HC_Financing_target'],
    ['L2_target_0', 0.9, 'L2_target_0'],
    ['L4_target_0', 0.9, 'L4_target_0'],
    ['S_FR_target_0', 0.7, 'S_FR_target_0'],
    ['S_T_target_0', 0.9, 'S_T_target_0'],
    ['dL2_DC_in_0', 0.2, 'dL2_DC_in_0'],
    ['dL4_DC_in_0', 0.2, 'dL4_DC_in_0'],
    ['P_DP_target', 0.7, 'P_DP_target'],
    ['P_M_target', 0.7, 'P_M_target'],
    ['P_I_target', 0.6, 'P_I_target'],
    ['P_RR_target_0', 1.0, 'P_RR_target_0'],
    ['L4_HF_target_0', 0.8, 'L4_HF_target_0'],
    ['S_TF_target_0', 0.8, 'S_TF_target_0'],
    ['L2_DC_target', 0.1, 'L2_Delivery_Capacity_Target'],
    ['L4_DC_target', 0.9, 'L4_Delivery_Capacity_Target'],
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
FN_original, FN_change = set_F_change(FN_0)

y_0 = S_0

def get_factors_0():
    return FP_0, FN_0

def calc_y(S_values, FP_values, FN_values, B_values, C_values, P_values): # values from the sliders
    # P_values = parameter values
    for i in range(len(FP_values)): # for each F-slider
        FP_change[i] = FP_values[i] # F-parameter that is collected from the slider
    for i in range(len(FN_values)): # for each F-slider
        FN_change[i] = FN_values[i] # F-parameter that is collected from the slider

    parameters['t_change'] = P_values[0] # slider value for time when the parameters change
    parameters['beta']     = P_values[1] # slider value for beta

    beta  = parameters['beta'] / 10
    y_t   = np.zeros((nt,len(S_values)))
    t_all = np.zeros(nt)
    anc_t, health_t, gest_age_t, deliveries, facilities = {0:[0]}, {0:[0]}, {0:[0]}, {0:[0]}, {0:[0]}
    # anc_t, health_t, gest_age_t, deliveries, facilities = [[]]*nt, [[]]*nt, [[]]*nt, [[]]*nt, [[]]*nt # NG
    num_deliver_home, num_deliver_2, num_deliver_4, num_deliver_total = \
        np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
    pos_HO, neg_HO, L2_D_Capacity, L4_D_Capacity = np.zeros([nt,4]), np.zeros([nt,4]), np.ones(nt), np.ones(nt)

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
        mothers.append(Mother(wealth[mother], education[mother], age[mother], no_children[mother], nt, B))

    # OTHER MISCELLANEOUS FACTORS
    L4_D_Capacity_Multiplier = 2

    # LOOP OVER EVERY TIME VALUE
    for t in range(0,nt-1):
        if t > parameters['t_change']: # IF TIME IS LARGER THAN THE t_change SLIDER VALUE
            for idx, name in FP_idx_names:
                globals()[name] = FP_change[idx] # then use the SLIDER value for the F-parameter, e.g., Visibility = 0.0
            for idx, name in FN_idx_names:
                globals()[name] = FN_change[idx]  # then use the SLIDER value for the F-parameter, e.g., Visibility = 0.0
        else:                                   # otherwise
            for idx, name in FP_idx_names:
                globals()[name] = FP_original[idx] # use the HARD-CODED value for the F-parameter saved in F_info
            for idx, name in FN_idx_names:
                globals()[name] = FN_original[idx] # use the HARD-CODED value for the F-parameter saved in F_info

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
        L2_demand = logistic(num_deliver_2[t] / (L2_D_Capacity[t]))
        L4_demand = logistic(num_deliver_4[t] / (L4_D_Capacity[t]))

        neg_HO_t = sum(neg_HO[0:t+1,:])
        if neg_HO_t[0] == 0:
            L2_4_health_outcomes = 0
        else:
            L2_4_health_outcomes = logistic([
                neg_HO_t[1] / neg_HO_t[0],
                neg_HO_t[2] / neg_HO_t[0] ])

        P_A_target    = (P_M[t] * logistic([Visibility, Action_depletion, 1]) + P_I[t]) / 2
        P_SP_target   = (P_P[t] + P_A[t] + P_D[t] * logistic([Visibility, Action_depletion, 1])) / 3
        dP_SP_in      = (P_P[t] + P_A[t] + P_D[t])
        dP_A_in       = (P_M[t] + P_I[t])

        P_RR_target   = P_RR_target_0 * logistic([Funding_MNCH, Support_Linda_Mama, Prioritization_MNCH, -Delayed_disbursement, -Lack_adherence_budget, 3])
        dP_RR_in      = P_DP[t] + P_SP[t]

        L2_HF_target  = L2_HF_target_0 * P_RR[t] * logistic([Adherence_budget, -Lack_adherence_budget, -Inadequate_financing, -Delayed_disbursement, 2])
        L4_HF_target  = L4_HF_target_0 * P_RR[t] * logistic([Adherence_budget, -Lack_adherence_budget, -Inadequate_financing, -Delayed_disbursement, 2])
        S_TF_target   = S_TF_target_0 * P_RR[t] * logistic([Adherence_budget, -Lack_adherence_budget, -Inadequate_financing, -Delayed_disbursement, 2])
        dL2_HF_in     = P_RR[t]  # coefficients of these three dStock_in terms add up to 1
        dL4_HF_in     = P_RR[t]
        dS_TF_in      = P_RR[t]
        # dP_RR_out = dL2_HF_in + dL4_HF_in + dS_TF_in
        L2_target_combined_0 = L2_target_0 * L2_HF[t] # combined targets of L2_HR and L2_S =0.9*target of L2_HF
        # L2_target_combined_0 = L2_target_0 # combined targets of L2_HR and L2_S =0.9*target of L2_HF
        L2_HR_target  = L2_target_combined_0 * logistic([Employee_incentives, -Lack_promotion, Timely_promotions, -Delay_hiring, -Frequent_transfer, -Burn_out, -Poor_management, Strong_referrals, Training_incentives, 3])
        L2_S_target   = L2_target_combined_0 * logistic([-Lack_action_depletion, Pos_supply_chain, -Neg_supply_chain, -L2_demand,2])
        dL2_HR_in     = L2_HF[t]
        dL2_S_in      = L2_HF[t]
        # dL2_HF_out = dL2_HR_in + dL2_S_in
        L4_target_combined_0 = L4_target_0 * L4_HF[t]
        # L4_target_combined_0 = L4_target_0
        L4_HR_target  = L4_target_combined_0 * logistic([Employee_incentives, -Lack_promotion, Timely_promotions, -Delay_hiring, -Frequent_transfer, -Burn_out, -Poor_management, Strong_referrals, Training_incentives, 3])
        L4_S_target   = L4_target_combined_0 * logistic([-Lack_action_depletion, Pos_supply_chain, -Neg_supply_chain, -L4_demand,2])
        dL4_HR_in     = L4_HF[t]
        dL4_S_in      = L4_HF[t]
        # dL4_HF_out = dL4_HR_in + dL4_S_in
        S_FR_target  = S_FR_target_0 * S_TF[t] * logistic([Employee_incentives, -Lack_promotion, Timely_promotions, -Delay_hiring, -Frequent_transfer, -Burn_out, -Poor_management, Strong_referrals, Training_incentives, 3])
        S_T_target   = S_T_target_0 * S_TF[t] * logistic([Employee_incentives, -Lack_promotion, Timely_promotions, -Delay_hiring, -Frequent_transfer, -Burn_out, -Poor_management, Strong_referrals, Training_incentives, 3])
        dS_FR_in     = S_TF[t]
        dS_T_in      = S_TF[t]
        # dS_TF_out  = dS_FR_in + dS_T_in

        # L2_DC_target  = 0.1
        # L4_DC_target  = 0.9
        dL2_DC_in     =  dL2_DC_in_0 * S_FR[t] # target < stock so need to reverse sign here
        dL4_DC_in     =  dL4_DC_in_0 * S_FR[t]

        L2_Q_target  = (L2_HR_target + L2_S_target) / 2 / L2_target_combined_0 * logistic([Strong_referrals, Increase_awareness, -9*L2_demand,5])
        L4_Q_target  = (L4_HR_target + L4_S_target) / 2 / L4_target_combined_0 * logistic([Strong_referrals, Increase_awareness_address_myths, -9/2*L4_demand,5])
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

        L2_deliveries = 0
        for mother in mothers:
            L2_net_capacity = 1 - (L2_deliveries + 1) / L2_D_Capacity[t] # add 1 to see if one more can be delivered
            mother.increase_age(l4_quality, l2_quality, proximity, L2_4_health_outcomes,
                                L2_net_capacity, None)  # don't need the last argument
            if mother.delivered:
                L2_deliveries += 1
                mother.delivered = False  # reset
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
        pos_HO[t+1,3] = sum(pos_HO[t+1,:3]) # totals
        neg_HO[t+1,3] = sum(neg_HO[t+1,:3])

        if t==nt-2: # last value of t, need to add to these array for plotting
            L2_D_Capacity[t+1] = L2_DC[t+1] * BL_Capacity_factor  # need to fill in the last time value
            L4_D_Capacity[t+1] = L4_DC[t+1] * BL_Capacity_factor * L4_D_Capacity_Multiplier

    return t_all, y_t, \
           [ num_deliver_4, num_deliver_2, num_deliver_home, num_deliver_total, L4_D_Capacity, L2_D_Capacity ],\
           [ pos_HO, neg_HO ]

# FOR OTHER PLOTTING METHODS
# gest_age_t = pd.DataFrame.from_dict(gest_age_t)
# health_t = pd.DataFrame.from_dict(health_t)
# anc_t = pd.DataFrame.from_dict(anc_t)
# deliveries = pd.DataFrame.from_dict(deliveries)
# facilities = pd.DataFrame.from_dict(facilities)

# Bundle parameters for ODE solver
parameters = {
    't_change':0.0,
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

FP_sliders = many_sliders(FP_label,'FP_slider',FP_0,np.zeros(len(FP_0)),
                          np.array(FP_0)*4, num_rows=3) # add 0 to 1 slider for INCREASE ALL
FP_combination_sliders = many_sliders(FP_combination_label,'FP_combination_slider',FP_combination_0,
                                      np.zeros(len(FP_combination_0)),np.ones(len(FP_combination_0)),
                                      num_rows=1, num_cols=3, width=4)
FN_sliders = many_sliders(FN_label,'FN_slider',FN_0,np.zeros(len(FN_0)),
                          np.array(FN_0)*4, num_rows=3)
FN_combination_sliders = many_sliders(FN_combination_label,'FN_combination_slider',FN_combination_0,
                                      np.zeros(len(FN_combination_0)),np.ones(len(FN_combination_0)),
                                      num_rows=1, num_cols=3, width=4)
B_sliders = many_sliders(B_label,'B_slider',B_0,np.zeros(len(B_0)),np.array(B_0)*4, num_rows=5, num_cols=4, width=3)
# many_sliders(labels, type used in Input() as an identifier of group of sliders, initial values, min, max, ...
C_sliders = many_sliders(C_label,'C_slider',C_0,np.zeros(len(C_0)),np.ones(len(C_0)), num_rows=5, num_cols=4, width=3)

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
        dbc.Col([
            html.H5('Positive Combination Factors'),
            FP_combination_sliders
        ], width=6),
        dbc.Col([
            html.H5('Negative Combination Factors'),
            FN_combination_sliders
        ], width=6),
    ], className="pretty_container"
    ),

    dbc.Row([
        dbc.Col(html.H5('Positive Factors'),width=12),
        FP_sliders,
        ],className="pretty_container"
    ),

    dbc.Row([
        dbc.Col(html.H5('Negative Factors'),width=12),
        FN_sliders,
        ],className="pretty_container"
    ),

    dbc.Row([
        dbc.Col(html.H5('Beta coefficients'),width=12),
        B_sliders,
        ],className="pretty_container"
    ),

    dbc.Row([
        dbc.Col(html.H5('Constant coefficients for the model'),width=12),
        C_sliders,
        ],className="pretty_container"
    ),

    dbc.Row([
        dbc.Col(html.H5('Meta parameters'), width=3),
        dbc.Col([
            html.Div([
                make_slider(0, 'Time when factor changes will take place', 'P_slider', parameters['t_change'], 0, nt)
            ]),
        ], width=3),
        dbc.Col([
            html.Div([
                make_slider(1, 'Common coefficient for rate of change', 'P_slider', parameters['beta'], 0, 100)
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
    dash.dependencies.Output({'type': 'FN_slider_button', 'index': ALL}, 'style'),
    dash.dependencies.Output({'type': 'FN_combination_slider_button', 'index': ALL}, 'style'),
    [Input({'type': 'FN_slider_button', 'index': ALL}, 'n_clicks'),],
    [State({'type': 'FN_slider_button', 'index': ALL}, 'style'),
     State({'type': 'FN_combination_slider_button', 'index': ALL}, 'style'),],
)
def update_labels(FN_clicks, FN_styles, FN_combination_styles):
    return update_colors(FN_clicks, FN_styles, FN_combination_styles)

@app.callback(
    dash.dependencies.Output({'type': 'FP_slider', 'index': ALL}, 'value'),  # simple trial
    dash.dependencies.Output({'type': 'FN_slider', 'index': ALL}, 'value'),  # simple trial
    [Input({'type': 'FP_combination_slider', 'index': ALL}, 'value'),
     Input({'type': 'FN_combination_slider', 'index': ALL}, 'value'),],
    [State({'type': 'FP_slider', 'index': ALL}, 'value'),
     State({'type': 'FN_slider', 'index': ALL}, 'value'),
     State({'type': 'FP_slider', 'index': ALL}, 'max'),
     State({'type': 'FN_slider', 'index': ALL}, 'max'),
     State({'type': 'FP_slider_button', 'index': ALL}, 'style'),
     State({'type': 'FN_slider_button', 'index': ALL}, 'style')]
)
def update_combination_slider(FP_combination_values, FN_combination_values, FP_values, FN_values,
                              FP_max, FN_max, FP_style, FN_style):
    FP_0,FN_0 = get_factors_0()
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

    return update_F_values(FP_values, FP_0, FP_max, FP_style, FP_combination_values), \
           update_F_values(FN_values, FN_0, FN_max, FN_style, FN_combination_values)

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
     Input({'type':'FN_slider','index':ALL}, 'value'),
     Input({'type':'B_slider','index':ALL}, 'value'),
     Input({'type':'C_slider','index':ALL}, 'value'),
     Input({'type':'P_slider','index':ALL}, 'value'),],
    [State({'type':'FP_slider','index':ALL}, 'max'),
     State({'type':'FN_slider','index':ALL}, 'max'),],
)
def update_graph(S_values,FP_values,FN_values,B_values,C_values,P_values,FP_max,FN_max): # each argument is one of Input(...)
    # S_values_global = np.array(S_values) # used for sensitivities
    # F_values_global = np.array(F_values)
    # SLIDER VALUES GETS PASSED TO THE MODEL TO COMPUTE THE MODEL RESULTS (e.g., y_t = stocks over time)
    t_all, y_t, num_d, pos_neg_HO = calc_y(S_values,FP_values,FN_values,B_values,C_values,P_values)
    # num_deliver_home, num_deliver_2, num_deliver_4, num_deliver_total, L2_D_Capacity, L4_D_Capacity = num_d
    k_range_1A  = range(0,6)
    k_range_1B  = range(6,11)
    k_range_1C  = range(11,16)
    k_range_2A  = range(16,len(S_label))
    def y_max(k_range, y=y_t, increments=5):
        if isinstance(y,list):
            max_y = 0
            for k in k_range:
                max_y = max(max_y, max(y[k]))
        else:
            max_y = np.amax(np.array(y)[:, k_range])

        return np.ceil(increments * max_y) / increments

    fig_1A = {
        'data':[{
            'x': t_all,
            'y': y_t[:,k],
            'name': S_label[k]
        } for k in k_range_1A],
        'layout': {
            'title':  'POLICY',
            'xaxis':{'title':'Time (months)'},
            'yaxis':{'range':[0,y_max(k_range_1A)], 'title':'Stocks (normalized units)'}
        }
    }

    fig_1B = {
        'data':[{
            'x': t_all,
            'y': y_t[:,k],
            'name': S_label[k]
        } for k in k_range_1B],
        'layout': {
            'title':  'RESOURCES',
            'xaxis':{'title':'Time (months)'},
            'yaxis':{'range':[0,y_max(k_range_1B)], 'title':'Stocks (normalized units)'}
        }
    }

    fig_1C = {
        'data':[{
            'x': t_all,
            'y': y_t[:,k],
            'name': S_label[k]
        } for k in k_range_1C],
        'layout': {
            'title':  'SERVICE READINESS',
            'xaxis':{'title':'Time (months)'},
            'yaxis':{'range':[0,y_max(k_range_1C)], 'title':'Stocks (normalized units)'}
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

    num_deliveries_labels = ['Level 4/5','Level 2/3','Home','Total','Capacity 4/5','Capacity 2/3'] # total is unused
    fig_2B = {
        'data':[{
            'x': t_all,
            'y': num_d[k],
            'name': num_deliveries_labels[k]
        } for k in [0,1,2,4,5]], # don't need total, so just the first three
        'layout': {
            'title':  'Deliveries over time',
            'xaxis':{'title':'Time (months)'},
            'yaxis':{'range':[0,y_max([0,1,2,4,5],num_d)], 'title':'Deliveries'}
        }
    }

    HO_labels = ['Home delivery','L2', 'L4', 'Total']
    fig_2C = {
        'data':[{
            'x': t_all,
            'y': pos_neg_HO[1][:,k],
            'name': HO_labels[k]
        } for k in [2,1,0,3]],
        'layout': {
            'title':  'Negative birth outcomes over time',
            'xaxis':{'title':'Time (months)'},
            'yaxis':{'range':[0,y_max([2,1,0,3],pos_neg_HO[1])], 'title':'Number of dyads'}
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
