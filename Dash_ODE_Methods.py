import dash
import numpy as np
from dash import html, dcc
import dash_bootstrap_components as dbc # conda install -c conda-forge dash-bootstrap-components


def logistic(x):
    return 1 / (1 + np.exp(-np.mean(x)))

def slider_markers(start=0, end=1, red=None, num_steps=20):
    eps = 0.0001
    max_value = end + eps
    min_value = start + eps

    def set_marks(nums, red):
        marks = {}
        n_nums = len(nums)
        for idx in range(n_nums):
            num = nums[idx][0]  # num is used in the dictionary
            num1 = nums[min(idx + 1, n_nums - 1)][0]  # next num
            num0 = nums[max(idx - 1, 0)][0]  # previous num
            num_mid_0 = (num0 + num) / 2
            num_mid_1 = (num + num1) / 2
            num_str = str(nums[idx][1])
            is_red = red and (red >= num_mid_0) and (red <= num_mid_1)  # need <= for the highest num
            if idx % 4 == 0 or is_red:
                marks[num] = {'label': num_str, 'style': {'fontSize': 12}}
                if is_red:
                    marks[num]['style'] = {'fontSize': 12, 'color': '#f50'}
                    red = num
            else:
                marks[num] = {'label': ''}
        return marks, red

    if end == 1:
        step = (end - start) / num_steps
        nums = [[round(num, 4), round(num, 1)] for num in np.arange(min_value, max_value + step, step)]
        if red is not None:
            red = round(red, 4)
        marks, red = set_marks(nums, red)
    elif end < 1:
        step = (end - start) / num_steps
        nums = [[round(num, 4), round(num, 2)] for num in np.arange(min_value, max_value + step, step)]
        if red is not None:
            red = round(red, 4)
        marks, red = set_marks(nums, red)
    else:
        step = (end - start) / num_steps
        nums = [[round(num, 4), round(num, 1)] for num in np.arange(min_value, max_value + step, step)]
        if red is not None:
            red = round(red, 4)
        marks, red = set_marks(nums, red)

    return marks, step, min_value, max_value, red


def make_slider(i, slider_label, slider_type, default_value, min_value=0, max_value=1):
    if max_value == 0:
        max_value = 1.0
    marks, step, min_value, max_value, default_value = slider_markers(min_value, max_value, default_value)
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


def many_sliders(slider_labels, slider_type, default_values, min_values, max_values, num_rows=5, num_cols=6, width=2):
    return \
        dbc.Row([
            dbc.Col([
                html.Div([
                    make_slider(i, slider_labels[i], slider_type, default_values[i], min_values[i], max_values[i]) \
                    for i in range(k * num_rows, min((k + 1) * num_rows, len(default_values)))
                ])
            ], width=width) for k in range(0, num_cols)
        ], style={'width': '100%'})

