import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go

def f(y, t, params):
    Unemployment, School_Dropouts = y # unpack current values of y
    Extortion, Access_to_Abortion = params  # unpack parameters
    dy = [ # list of dy/dt=f functions
        -0.1*Unemployment + 0.2*Access_to_Abortion,     
        -0.1*School_Dropouts
        -0.1*Unemployment + 0.2*Extortion,
     ]
    return dy

# Parameters
Extortion = 2.0
Access_to_Abortion = 1.5

# Initial values
Unemployment_0 = 1.0
School_Dropouts_0 = 2.0

# Bundle parameters for ODE solver
params = [Extortion, Access_to_Abortion]

# Bundle initial conditions for ODE solver
y0 = [Unemployment_0, School_Dropouts_0]

def Euler(f, y_0, t, parameters):
    h = t[1] - t[0]
    n = len(t)
    m = len(y_0)
    y = np.zeros([n,m])
    y[0,:] = y_0
    for i in range(n):
        dy = f(y[i,:],t[i],parameters)
        print(i)
        print(dy)
        y[i+1,:] = np.multiply(dy,h)
    return y

# Make time array for solution
t_stop = 10.
t_increment = 0.1
t = np.arange(0., t_stop, t_increment)

# Call the ODE solver
# psoln = odeint(f, y0, t, args=(params,))
psoln = Euler(f, y0, t, params)

print(psoln[-5]) # last result

# fig = go.Figure(data=go.Scatter(x=t, y=psoln[0]))
# fig.show()
