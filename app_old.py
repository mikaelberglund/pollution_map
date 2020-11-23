import os
from netCDF4 import Dataset
import numpy as np
from plotly.graph_objs import *
import xarray as xr
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import ee
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import geopandas as gpd

mapbox_access_token = 'pk.eyJ1IjoiYmVyZ2x1bmRtaWthZWwiLCJhIjoiY2s5OW9mbGRuMDVzeTNtanluaXY0MjJ5ciJ9.EoSpUqLEDapNCy5eZjkJRQ'
maptitle = 'Recorded NO2 pollution in the troposhpere (lower atmosphere)'

dfcities = pd.read_csv('worldcities.csv')

ee.Initialize()
# Define the roi
area = ee.Geometry.Polygon([[18.001586, 59.354705], \
                            [18.101210, 59.356161], \
                            [18.105495, 59.294224], \
                            [17.987303, 59.299511]])

# export the latitude, longitude and array
def LatLonImg(img,ar):
    img = img.addBands(ee.Image.pixelLonLat())

    img = img.reduceRegion(reducer=ee.Reducer.toList(), \
                           geometry=ar, \
                           maxPixels=1e13, \
                           scale=800, tileScale=8);

    data = np.array((ee.Array(img.get("result")).getInfo()))
    lats = np.array((ee.Array(img.get("latitude")).getInfo()))
    lons = np.array((ee.Array(img.get("longitude")).getInfo()))
    return lats, lons, data


# covert the lat, lon and array into an image
def toImage(lats, lons, data):
    # get the unique coordinates
    uniqueLats = np.unique(lats)
    uniqueLons = np.unique(lons)

    # get number of columns and rows from coordinates
    ncols = len(uniqueLons)
    nrows = len(uniqueLats)

    # determine pixelsizes
    ys = uniqueLats[1] - uniqueLats[0]
    xs = uniqueLons[1] - uniqueLons[0]

    # create an array with dimensions of image
    arr = np.zeros([nrows, ncols], np.float32)  # -9999

    # fill the array with values
    counter = 0
    for y in range(0, len(arr), 1):
        for x in range(0, len(arr[0]), 1):
            if lats[counter] == uniqueLats[y] and lons[counter] == uniqueLons[x] and counter < len(lats) - 1:
                counter += 1
                arr[len(uniqueLats) - 1 - y, x] = data[counter]  # we start from lower left corner
    return arr

def getdf(temparea):
    columnheight = 1E4 #height of troposphere
    NO2gpermole = 46.0055 #factor to convert 1 mole of NO2 to grams
    tomicro=1E6 #convert to μg
    ee.Initialize()
    # define the image
    print('Getting DF')
    myCollection = ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_NO2").filterBounds(temparea)\
        .filterDate("2020-01-01", "2020-01-30").select(['tropospheric_NO2_column_number_density'])

    # get the median
    result = ee.Image(myCollection.median()).rename(['result'])

    # get the lon, lat and result as 1d array
    lat, lon, data = LatLonImg(result,temparea)
    print('DF Latitude: ' +str(lat.mean()) + ' & longitude: ' + str(lon.mean()))
    dftemp = pd.DataFrame(data)
    dftemp['lat'] = pd.DataFrame(lat)
    dftemp['lon'] = pd.DataFrame(lon)
    dftemp.columns = ['value','lat','lon']
    dftemp.value = dftemp*tomicro*NO2gpermole/columnheight
    print(dftemp.value.max())
    r = 0.2
    ppdftemp = pd.read_csv('global_power_plant_database.csv',usecols=[1,2,4,5,6,7])
    ppdftemp = ppdftemp[(ppdftemp.latitude.between(lat.mean() - r, lat.mean() + r)) &
                (ppdftemp.longitude.between(lon.mean() - r, lon.mean() + r))]
    ppdftemp = ppdftemp[ppdftemp.primary_fuel.isin(['Gas', 'Oil', 'Coal', 'Petcoke'])]
    return dftemp,ppdftemp

df, ppdf = getdf(area)

def getfig(dframe, ppdframe):
    print('Getting Fig')
    fig = px.density_mapbox(dframe, lat='lat', lon='lon', z='value', radius=25,
                           center=dict(lat=dframe.lat.mean(), lon=dframe.lon.mean()),
                            zoom=9, hover_data=['value'],range_color=[0,4],
                           mapbox_style="stamen-terrain", height=800,
                            title='Estimated NO2 μg/m³')
    #fossil = ppdframe[ppdframe.primary_fuel.isin(['Gas','Oil','Coal','Petcoke'])]
    fig.add_trace(
        go.Scattermapbox(
            lat=ppdframe.latitude,
            lon=ppdframe.longitude,
            mode='markers',
            marker=go.scattermapbox.Marker(size=20),
            text='Power plant: '+ppdframe.name+' Fuel: '+ppdframe.primary_fuel))

    # fig = go.Figure(go.Densitymapbox(lat=dframe.lat, lon=dframe.lon, z=dframe.value, radius=20,
    #                  hovertext=dframe.value,zmin=0,zmax=0.005,{width: 600, height: 400, mapbox: {style: 'stamen-terrain'}}))

    print('Fig Latitude: ' + str(dframe.lat.mean()) + ' & longitude: ' + str(dframe.lon.mean()))
    return fig

fig = getfig(df,ppdf)

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
    #r = 0.2
    templat = dfcities[dfcities.city == city].lat
    templat = np.array(templat)[0]
    templon = dfcities[dfcities.city == city].lng
    templon = np.array(templon)[0]
    temparea = ee.Geometry.Rectangle(templon+2*r,templat+r,
                                     templon-2*r,templat-r)
    print('Getting closer')
    tempdf, tempppdf = getdf(temparea)
    tempfig = getfig(tempdf, tempppdf)
    return tempfig

if __name__ == '__main__':
    app.run_server(debug=True)