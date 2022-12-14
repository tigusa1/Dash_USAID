import dash
import numpy as np
from dash import html, dcc
import dash_bootstrap_components as dbc # conda install -c conda-forge dash-bootstrap-components


def logistic(x):
    return 1 / (1 + np.exp(-np.mean(x)))

def slider_markers(start=0, end=1, red=None, blue=None, num_steps=20):
    eps = 0.0001
    max_value = end + eps
    min_value = start + eps

    def set_marks(nums, red, blue, step):
        marks = {}
        n_nums = len(nums)
        for idx in range(n_nums):
            num = nums[idx][0]  # num is used in the dictionary
            num_str = str(nums[idx][1])
            if idx == 0:
                num1 = nums[idx + 1][0]  # next num
                num_mid_0 = nums[0][0] - step
                num_mid_1 = (num + num1) / 2
            elif idx == n_nums-1:
                num0 = nums[idx - 1][0]  # next num
                num_mid_0 = (num0 + num) / 2
                num_mid_1 = nums[idx][0] + step
            else:
                num0 = nums[idx - 1][0]  # previous num
                num1 = nums[idx + 1][0]  # next num
                num_mid_0 = (num0 + num) / 2
                num_mid_1 = (num + num1) / 2

            is_red  = (not red  == None  and (red  >= num_mid_0) and (red  <= num_mid_1))  # need <= for the highest num
            is_blue = (not blue == None and (blue >= num_mid_0) and (blue <= num_mid_1))
            if idx % 4 == 0 or is_red or is_blue:
                marks[num] = {'label': num_str, 'style': {'fontSize': 12}}
                if is_blue:
                    marks[num]['style'] = {'fontSize': 13, 'color': 'PaleTurquoise1', 'background': '#19D3F3'}
                    blue = num
                if is_red:
                    marks[num]['style'] = {'fontSize': 13, 'color': '#f50', 'background': 'rgb(255,255,0)'}
                    red = num
            else:
                marks[num] = {'label': ''}
        return marks, red, blue

    if end == 1:
        step = (end - start) / num_steps
        nums = [[round(num, 4), round(num, 1)] for num in np.arange(min_value, max_value + step, step)]
        if red is not None:
            red = round(red, 4)
        if blue is not None:
            blue = round(blue, 4)
        marks, red, blue = set_marks(nums, red, blue, step)
    elif end < 1:
        step = (end - start) / num_steps
        nums = [[round(num, 4), round(num, 2)] for num in np.arange(min_value, max_value + step, step)]
        if red is not None:
            red = round(red, 4)
        if blue is not None:
            blue = round(blue, 4)
        marks, red, blue = set_marks(nums, red, blue, step)
    else:
        step = (end - start) / num_steps
        nums = [[round(num, 4), round(num, 1)] for num in np.arange(min_value, max_value + step, step)]
        if red is not None:
            red = round(red, 4)
        if blue is not None:
            blue = round(blue, 4)
        marks, red, blue = set_marks(nums, red, blue, step)

    return marks, step, min_value, max_value, red, blue


def make_slider(i, slider_label, slider_type, default_value, initial_value, min_value=0, max_value=1):
    if max_value == 0:
        max_value = 1.0
    marks, step, min_value, max_value, default_value, initial_value = \
        slider_markers(min_value, max_value, default_value, initial_value)
    return html.Div(children=[
        html.Button(slider_label,
                    id={'type': slider_type + '_button', 'index': i},
                    n_clicks=0,
                    style={'color': '#000', 'fontSize': 16, 'textTransform' : 'none'},  # red : f50
                    ),
        dcc.Slider(
            id={'type': slider_type, 'index': i},
            min=min_value,
            max=max_value,
            value=default_value,
            marks=marks,
            step=None
        ),
    ])


def many_sliders(slider_labels, slider_type, default_values, initial_values, min_values, max_values, num_rows=5, num_cols=6, width=2):
    if len(initial_values)==0:
        initial_values = default_values
    return \
        dbc.Row([
            dbc.Col([
                html.Div([
                    make_slider(i, slider_labels[i], slider_type, default_values[i], initial_values[i], min_values[i], max_values[i]) \
                    for i in range(k * num_rows, min((k + 1) * num_rows, len(default_values)))
                ])
            ], width=width) for k in range(0, num_cols)
        ], style={'width': '100%'})

