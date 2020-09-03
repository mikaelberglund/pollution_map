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
import math
import datashader as ds

mapbox_access_token = 'pk.eyJ1IjoiYmVyZ2x1bmRtaWthZWwiLCJhIjoiY2s5OW9mbGRuMDVzeTNtanluaXY0MjJ5ciJ9.EoSpUqLEDapNCy5eZjkJRQ'
maptitle = 'Recorded NO2 pollution compared against Covid-19 mortality'

dfcities = pd.read_csv('worldcities.csv')
verbose=True
dfpollution = pd.read_excel('http://www.airviro.smhi.se/RUS/rapporter/lansrapport_stockholm_Cu.xls')


ee.Initialize()
# Define the roi
# area = ee.Geometry.Polygon([[18.001586, 59.354705], \
#                            [18.101210, 59.356161], \
#                            [18.105495, 59.294224], \
#                            [17.987303, 59.299511]])

# export the latitude, longitude and array

################################### FUNCTIONS ###################################
def LatLonImg(img,ar):
    img = img.addBands(ee.Image.pixelLonLat())

    img = img.reduceRegion(reducer=ee.Reducer.toList(), \
                           geometry=ar, \
                           #maxPixels=1e13,crs='EPSG:4326',crsTransform=);
                           #maxPixels=1e13,scale=3000, tileScale=8);
                           #maxPixels=1e3,scale=1, tileScale=2);
                           #maxPixels=1e9, bestEffort=True,crs='EPSG:4326',scale=200);
                            maxPixels = 262144, bestEffort = True, crs = 'EPSG:4326', scale = 200);

    img = ee.Image(img).sampleRectangle(region=ar)

    data = np.array((ee.Array(img.get("result")).getInfo()))
    lats = np.array((ee.Array(img.get("latitude")).getInfo()))
    lons = np.array((ee.Array(img.get("longitude")).getInfo()))
    return lats, lons, data


# covert the lat, lon and array into an image
def toImage(lats, lons, data):

    # get the unique coordinates
    uniqueLats = np.unique(lats) ###SORTERA ALLA ME EN DF FÖRST?
    uniqueLons = np.unique(lons)
    # print('Sorting data')
    #data, lats, lons = np.sort(np.array([data, lats, lons]), axis=1)
    # tarr = np.array([data, lats, lons])
    # # tarr[1,:] = tarr[1,np.argsort(tarr[1,:])]
    # # tarr[2,:] = tarr[2,np.argsort(tarr[2,:])]
    # tarr = sorted(tarr, key=lambda row: row[2])
    # tarr = sorted(tarr, key=lambda row: row[1])
    # data, lats, lons = tarr
    # print('Sorted data')
    # get number of columns and rows from coordinates
    ncols = len(uniqueLons)
    nrows = len(uniqueLats)

    ##### Convert to image, using np.reshape
    # Pad data array with zeros, if it's not possible to convert to image.
    diff = ncols * nrows - len(data)
    #data = lons
    if diff != 0:
        data = np.pad(data,[0,diff])
        lats = np.pad(lats, [0, diff])
        lons = np.pad(lons, [0, diff])
    data = np.array([data,lats,lons])
    #arr = np.fliplr(np.reshape(data[::-1], [ncols, nrows]))
    arr = np.fliplr(np.reshape(data, [ncols, nrows,3]))

    # determine pixelsizes
    ys = uniqueLats[1] - uniqueLats[0]
    xs = uniqueLons[1] - uniqueLons[0]

    # ##### create an array with dimensions of image
    # arr = np.zeros([nrows, ncols], np.float32)  # -9999
    #
    # # fill the array with values
    # counter = 0
    # for y in range(0, len(arr), 1):
    #     for x in range(0, len(arr[0]), 1):
    #         if lats[counter] == uniqueLats[y] and lons[counter] == uniqueLons[x] and counter < len(lats) - 1:
    #             counter += 1
    #             arr[len(uniqueLats) - 1 - y, x] = data[counter]  # we start from lower left corner
    return arr

def getcoord(thedfcities,thecity,ther):
    templat = thedfcities[thedfcities.city == thecity].lat
    templat = np.array(templat)[0]
    templon = thedfcities[thedfcities.city == thecity].lng
    templon = np.array(templon)[0]
    #temparea = ee.Geometry.Rectangle(coords = [templon-2*ther,templat-ther,templon+2*ther,templat+ther],proj='EPSG:4326')
    temparea = ee.Geometry.Rectangle(coords=[templon - 2 * ther, templat - ther, templon + 2 * ther, templat + ther],
                                     proj=None, geodesic=False)
    # coords = xMin, yMin, xMax, yMax
    return temparea

#area = getcoord(dfcities,thecity='Stockholm',ther = 0.5)

def getdf(temparea):
    columnheight = 1E4 #height of troposphere
    NO2gpermole = 46.0055 #factor to convert 1 mole of NO2 to grams
    tomicro=1E6 #convert to μg
    ee.Initialize()
    # define the image
    myCollection = ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_NO2").filterBounds(temparea)\
        .filterDate("2020-01-01", "2020-01-30").select(['tropospheric_NO2_column_number_density'])

    # get the median
    result = ee.Image(myCollection.median()).rename(['result'])

    # get the lon, lat and result as 1d array
    lat, lon, data = LatLonImg(result,temparea)
    im = toImage(lat, lon, data)
    ### TESTING DATASHADER:
    from colorcet import fire
    import datashader.transfer_functions as tf
    im = tf.shade(xr.DataArray(im), cmap=fire)[::-1].to_pil()

    dftemp = pd.DataFrame(data)
    dftemp['lat'] = pd.DataFrame(lat)
    dftemp['lon'] = pd.DataFrame(lon)
    dftemp.columns = ['value','lat','lon']
    dftemp.value = dftemp*tomicro*NO2gpermole/columnheight
    r = 4
    cdftemp = pd.read_csv('Bing-COVID19-Data.csv',usecols=[1,2,3,4,5,8,9,12,13,14])
    cdftemp = cdftemp[(cdftemp.Latitude.between(lat.mean() - r, lat.mean() + r)) &
                (cdftemp.Longitude.between(lon.mean() - r, lon.mean() + r))]
    cdftemp = cdftemp[pd.to_datetime(cdftemp.Updated).between('2020-05-01', '2020-05-01')]
    cdftemp = cdftemp[cdftemp.AdminRegion1.notna()]
    if cdftemp.AdminRegion2.notna().sum() > 0:
        cdftemp = cdftemp[cdftemp.AdminRegion2.notna()]
    deathnan = cdftemp.Deaths.isna().sum()
    summarytemp = 'There are ' + str(cdftemp.shape[0]) + ' regions where ' + str(deathnan) + \
                  ' region(s) lack data on mortality.'
    cdftemp = cdftemp[cdftemp.Deaths.notna()]
    cdftemp = cdftemp[cdftemp.Confirmed.notna()]
    if verbose:
        print('DF Latitude: ' + str(lat.mean()) + ' & longitude: ' + str(lon.mean()))
        print('The maximum recorded pollution in the area is :' + str(dftemp.value.max()))
        print(summarytemp)
    return dftemp,cdftemp,summarytemp, im

#df, ppdf,_, image = getdf(area)

def getfig(dframe, ppdframe,imagetemp, tarea):
    #d = dframe[dframe.lat == dframe.lat.median()][dframe.lon == dframe.lon.median()]
    dF = pd.DataFrame()
    dF[['lon', 'lat']] = pd.DataFrame(tarea.coordinates().getInfo()[0])
    dF = pd.DataFrame(dF.mean()).T
    fig = px.scatter_mapbox(dF[:1], lat='lat', lon='lon', zoom=6)
    cooords = np.array(pd.DataFrame(tarea.coordinates().getInfo()[0]).loc[1:,:])
    if verbose:
        print('Scatter coordinates are Latitude:' + str(np.array(dF[:1].lat)[0])+
              ' & Longitude: '+ str(np.array(dF[:1].lon)[0]))
        print('Image coordinates are ' + str(cooords))
    fig.update_layout(mapbox_style="carto-darkmatter", #"stamen-terrain",
                      mapbox_layers=[
                          {
                              "sourcetype": "image",
                              "source": imagetemp,
                              "coordinates": cooords,
                              "opacity":0.35
                          }]
                      )
    htext = ppdframe.AdminRegion1+': '+ppdframe.AdminRegion2+': '+ppdframe.Deaths.astype(str)
    fig.add_trace(
        go.Scattermapbox(
            lat=ppdframe.Latitude,
            lon=ppdframe.Longitude,
            mode='markers',
            #marker=go.scattermapbox.Marker(size=ppdframe.Deaths/100),
            marker=go.scattermapbox.Marker(size=(200*ppdframe.Deaths) / (1+ppdframe.Confirmed)),
            text=htext
        ))
    #fig.show()
    #if verbose:
        #print('Fig Latitude: ' + str(dframe.lat.mean()) + ' & longitude: ' + str(dframe.lon.mean()))
    return fig


#fig = getfig(df,ppdf,image, area)

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#external_stylesheets = ['https://raw.githubusercontent.com/plotly/dash-app-stylesheets/master/dash-drug-discovery-demo-stylesheet.css']
external_stylesheets = ['https://codepen.io/chriddyp/pen/dZVMbK.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#app = dash.Dash(__name__)

################################### LAYOUT ###################################
app.layout = html.Div([
    html.H2(maptitle),
    dcc.Dropdown(id="dropdown-cities",value='Stockholm',options=[
        {'label': i, 'value': i} for i in dfcities.city.unique()
    ], multi=False, placeholder='Filter by city...'),
    html.Br(),
    html.Div(id='summary'),
    #dcc.Graph(id="my-graph", figure=fig)],
    dcc.Graph(id="my-graph")],
    style={"height" : "100%", "width" : "100%"},
    className="container")

################################### CALLBACK ###################################
@app.callback(
    [dash.dependencies.Output('my-graph', 'figure'),
    dash.dependencies.Output('summary','children')],
    [dash.dependencies.Input('dropdown-cities', 'value')])
def update_coordinates(city):
    #r = max(2,min(np.array((dfcities[dfcities.city == city].population)/5E6)[0],6)) #coordinate radius
    r = 1 #0.2 funkar
    if verbose:
        print()
        print(city)
        print('Radius is ' +str(r))
    temparea = getcoord(dfcities,city,r)

    tempdf, tempppdf, tempsummary, imag = getdf(temparea)
    tempfig = getfig(tempdf, tempppdf,imag, temparea)
    return tempfig,tempsummary

if __name__ == '__main__':
    app.run_server(debug=True,port=8051)