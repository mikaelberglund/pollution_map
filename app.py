import pandas as pd
import plotly.graph_objs as go
import ee
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from get_aq import get_data, get_last, get_IC, get_ee_dataset


mapbox_access_token = 'pk.eyJ1IjoiYmVyZ2x1bmRtaWthZWwiLCJhIjoiY2s5OW9mbGRuMDVzeTNtanluaXY0MjJ5ciJ9.EoSpUqLEDapNCy5eZjkJRQ'
maptitle = 'Recorded NO2 pollution in the troposhpere (lower atmosphere)'

dfcities = pd.read_csv('worldcities.csv')

ee.Initialize()
# Define the roi
area = ee.Geometry.Polygon([[18.001586, 59.354705],[18.101210, 59.356161],[18.105495, 59.294224],[17.987303, 59.299511]])

def getfig():
    print('Getting Fig')
    fig = go.Figure()
    return fig

fig = getfig()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2(maptitle),
    dcc.Dropdown(id="dropdown-cities",value='Stockholm',options=[
        {'label': i, 'value': i} for i in dfcities.city.unique()
    ], multi=False, placeholder='Filter by city...'),
    dcc.Graph(id="my-graph", figure=fig)],
    style={"height" : "100%", "width" : "100%"},
    className="container")

@app.callback(
    dash.dependencies.Output('my-graph', 'figure'),
    [dash.dependencies.Input('dropdown-cities', 'value')])
def update_coordinates(city):
    print(city)
    r = max(np.array((dfcities[dfcities.city == city].population)/5E6)[0],1) #coordinate radius
    print('Radius is ' +str(r))
    start = '2020-06-12'
    end = '2020-07-30'
    #r = 0.2
    templat = dfcities[dfcities.city == city].lat
    templat = np.array(templat)[0]
    templon = dfcities[dfcities.city == city].lng
    templon = np.array(templon)[0]
    temparea = ee.Geometry.Rectangle(templon+2*r,templat+r,
                                     templon-2*r,templat-r)
    ee_dataset = get_ee_dataset()
    landsat = get_IC(ee_dataset.loc[0, :][0], temparea, start, end, ee_dataset.loc[0, :][1])
    df, date = get_last(landsat, start, end, temparea, i, ee_dataset)
    #tempdf, tempppdf = getdf(temparea)
    tempfig = getfig()
    return tempfig

if __name__ == '__main__':
    app.run_server(debug=True)