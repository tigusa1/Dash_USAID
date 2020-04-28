import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go

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

def f(y, t, parameters): # 12 variables
    S_GM, S_IN, S_LE, S_MD, S_PHV, S_PG, S_PSV, S_SD, S_SV, S_SA, S_TM, S_UN = y
    
    Access_to_Abortion, Access_to_Contraception, Bad_Governance, Bully, Deportation, Economy, \
        Economy_Opportunities, Exposure_to_Violent_Media, Extortion, Family_Breakdown, Family_Cohesion, \
        Gang_Affiliation, Gang_Cohesion, Gang_Control, Interventions, Impunity_Governance, Machismo, Mental_Health, \
        Neighborhood_Stigma, School_Quality, Territorial_Fights, Victimizer, Youth_Empowerment = parameters
    
    D_S_GM = \
        + 0.1*(Exposure_to_Violent_Media - S_LE - Family_Cohesion - Mental_Health) \
        + 0.1*(Deportation) \
        - (S_GM)*( \
            + 0.1*(Gang_Cohesion + S_SA) \
            + 0.1*(Territorial_Fights - S_LE) \
            + 0.1*(Gang_Control - Impunity_Governance) \
        )

    D_S_IN = \
        + 0.01*(S_LE) \
        - (S_IN)*(
            + 0.1*S_PG
        )

    D_S_LE = \
        + 0.01*(Gang_Control - Impunity_Governance) \
        - 0.01*(S_LE) \
        + 0.01*(-Impunity_Governance) \
        - 0.01*(Exposure_to_Violent_Media + Bad_Governance - Youth_Empowerment + Neighborhood_Stigma)

    D_S_MD = \
        + 0.01*(Economy + S_PHV) \
        + 0.01*(Machismo - Economy_Opportunities) \
        - 0.01*(Deportation) \
        - 0.01*(S_MD - Family_Cohesion)

    D_S_PG = \
        + 0.01*(S_PG + S_IN) \
        + 0.01*(Exposure_to_Violent_Media) \
        + 0.01*(S_PSV) \
        + 0.01*(-Gang_Control + Victimizer + S_GM) \
        + 0.01*(Family_Breakdown + Exposure_to_Violent_Media - Mental_Health) \
        - 0.01*(Exposure_to_Violent_Media - S_LE - Family_Cohesion - Mental_Health)

    D_S_PHV = \
        + 0.01*(Exposure_to_Violent_Media + Bad_Governance - Youth_Empowerment + Neighborhood_Stigma) \
        + 0.01*(Territorial_Fights - S_LE) \
        - 0.01*(S_PHV + S_SA) \
        - 0.01*(Family_Breakdown + Exposure_to_Violent_Media - Mental_Health) \
        - 0.01*(S_PHV) \
        - 0.01*(Economy + S_PHV) \
        - 0.01*(-Impunity_Governance)

    D_S_PSV = \
        + 0.01*(S_SV) \
        + 0.01*(S_PHV + S_SA) \
        + 0.01*(S_MD - Family_Cohesion) \
        + 0.01*(S_SA + Gang_Cohesion) \
        - 0.01*(S_PSV + S_SA) \
        - 0.01*(S_PSV) \
        - 0.01*(Bully - Interventions - School_Quality)

    D_S_SD = \
        + 0.01*(Access_to_Abortion - Interventions + S_UN + S_TM) \
        + 0.01*(Bully - Interventions - School_Quality) \
        + 0.01*(S_PHV) \
        - 0.01*(Neighborhood_Stigma + Extortion - Economy_Opportunities)

    D_S_SV = \
        + 0.01*(-Economy_Opportunities + Gang_Affiliation - S_LE - Mental_Health + S_SD + S_SA) \
        + 0.01*(S_GM) \
        + 0.01*(S_PSV + S_SA) \
        + 0.01*(S_UN + Exposure_to_Violent_Media + Family_Breakdown - Youth_Empowerment - Gang_Control) \
        - 0.01*(S_SV - Access_to_Contraception) \
        - 0.01*(S_SV) \
        - 0.01*(-Gang_Control + Victimizer + S_GM)

    D_S_SA = \
        + 0.01*(S_GM) \
        - 0.01*(-Economy_Opportunities + Gang_Affiliation - S_LE - Mental_Health + S_SD + S_SA)

    D_S_TM = \
        + 0.01*(S_SV - Access_to_Contraception) \
        - 0.01*(Access_to_Abortion - Interventions + S_UN + S_TM)

    D_S_UN = \
        + 0.01*(Neighborhood_Stigma + Extortion - Economy_Opportunities) \
        - 0.01*(S_UN + Exposure_to_Violent_Media - Youth_Empowerment + Family_Breakdown - Gang_Control) \
        - 0.01*(Machismo - Economy_Opportunities) \
        - 0.01*(Exposure_to_Violent_Media)

    D_y = [ D_S_GM, D_S_IN, D_S_LE, D_S_MD, D_S_PHV, D_S_PG, D_S_PSV, D_S_SD, D_S_SV, D_S_SA, D_S_TM, D_S_UN ]

    return D_y

# Parameters
Access_to_Abortion = 0.2
Access_to_Contraception = 0.2
Bad_Governance = 0.2
Bully = 0.2
Deportation = 0.2
Economy = 0.2
Economy_Opportunities = 0.2
Exposure_to_Violent_Media = 0.2
Extortion = 0.2
Family_Breakdown = 0.2
Family_Cohesion = 0.2
Gang_Affiliation = 0.2
Gang_Cohesion = 0.2
Gang_Control = 0.2
Interventions = 0.2
Impunity_Governance = 0.2
Machismo = 0.2
Mental_Health = 0.2
Neighborhood_Stigma = 0.2
School_Quality = 0.2
Territorial_Fights = 0.2
Victimizer = 0.2
Youth_Empowerment = 0.2

# Initial values
S_GM_0 = 0.3
S_IN_0 = 0.4
S_LE_0 = 0.3
S_MD_0 = 0.3
S_PHV_0 = 0.3
S_PG_0 = 0.3
S_PSV_0 = 0.3
S_SD_0 = 0.3
S_SV_0 = 0.3
S_SA_0 = 0.3
S_TM_0 = 0.3
S_UN_0 = 0.3

# Bundle parameters for ODE solver
params = [
    Access_to_Abortion, Access_to_Contraception, Bad_Governance, Bully, Deportation, Economy,
    Economy_Opportunities, Exposure_to_Violent_Media, Extortion, Family_Breakdown, Family_Cohesion, 
    Gang_Affiliation, Gang_Cohesion, Gang_Control, Interventions, Impunity_Governance, Machismo, Mental_Health, 
    Neighborhood_Stigma, School_Quality, Territorial_Fights, Victimizer, Youth_Empowerment
]

# Bundle initial conditions for ODE solver
y_0 = [ S_GM_0, S_IN_0, S_LE_0, S_MD_0, S_PHV_0, S_PG_0, S_PSV_0, S_SD_0, S_SV_0, S_SA_0, S_TM_0, S_UN_0 ]

y_label = [
    'Gang_Membership', 'Incarceration', 'Law_Enforcement', 'Migration_Displacement',
    'Physical_Violence', 'Positive_Gang_Perception', 'Psychological_Violence',
    'School_Dropouts', 'Sexual_Violence', 'Substance_Abuse', 'Teenager_Mothers', 'Unemployment'
]

# Make time array for solution
t_stop = 100.
t_increment = 0.3
t = np.arange(0., t_stop, t_increment)

# Call the ODE solver
y_t = odeint(f, y_0, t, args=(params,))

print(y_t[-3:,:]) # last result2

fig = go.Figure()

for i in range(len(y_label)):
    fig.add_trace(go.Scatter(x=t, y=y_t[:,i],
                    mode='lines',
                    name=y_label[i]))

fig.show()
