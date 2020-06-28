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

Stocks = {
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
        'location':{
            'x':0.6,
            'y':0.5,
        },
        'text':'Psychological Violence',
    },
}

Factors = {
    'Deportation':{
        'location':{
            'x':0.85,
            'y':0.3,
        },
        'text':'Deportation',       
    }
}