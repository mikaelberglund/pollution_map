# -*- coding: utf-8 -*-

import requests as re
import pandas as pd
import ee
import numpy as np
import datetime
import datetime as dt
from http.client import IncompleteRead
from aq_functions import get_IC, get_ee_dataset
import time

#!earthengine authenticate
ee.Initialize()

if False:
    dftot = pd.read_pickle('dftot.pkl')
    dfs = pd.read_pickle('dfs.pkl')
    dfm = pd.read_pickle('dfm.pkl')

# def get_ee_dataset():
#     ee_dataset = pd.DataFrame([
#         ["COPERNICUS/S5P/NRTI/L3_NO2",'tropospheric_NO2_column_number_density'],
#         ["COPERNICUS/S5P/NRTI/L3_NO2",'stratospheric_NO2_column_number_density'],
#         ['COPERNICUS/S5P/NRTI/L3_AER_AI','absorbing_aerosol_index'],
#         ['NASA/GLDAS/V021/NOAH/G025/T3H','Rainf_tavg'],
#         ['NASA/GLDAS/V021/NOAH/G025/T3H','Wind_f_inst'],
#         ['NASA/GLDAS/V021/NOAH/G025/T3H', 'Tair_f_inst'],
#         ['NASA/GLDAS/V021/NOAH/G025/T3H', 'Qair_f_inst'],
#         ['COPERNICUS/S5P/NRTI/L3_CO','CO_column_number_density']
#     ],columns=['dataset','bands'])
#     return ee_dataset

# def get_IC(dataset,temparea,start,end,band):
#     landsat = ee.ImageCollection(dataset)
#     landsat = landsat.filterBounds(temparea)
#     landsat = landsat.filterDate(start, end)
#     landsat = landsat.select([band])
#     return landsat

def get_last(imagecol,s,e,ar,i,ee_dataset):
    test = np.array(imagecol.max().reduceRegion(reducer=ee.Reducer.toList(), geometry=ar, maxPixels=1e13, scale=70,
                                                tileScale=4).getInfo().get('tropospheric_NO2_column_number_density'))
    if test.shape[0] > 0:
    # if True:
        if test.mean() > 0:
        # if np.array(imagecol.max().reduceRegion(reducer=ee.Reducer.toList(), geometry=ar, maxPixels=1e13, scale=70,
        #                                             tileScale=4).getInfo().get(
        #     'tropospheric_NO2_column_number_density')).mean() > 0:
            found_last = False
            day = 1
            d = dt.datetime.strptime(e, '%Y-%m-%d')
            while ~found_last:
                im_test = imagecol.filterDate(start = d - dt.timedelta(days=day),opt_end = d)
                im_test = ee.Image(im_test.first()).unmask()
                if verbose:
                    print(d - dt.timedelta(days=day))
                try:
                    if (im_test.getInfo() is not None):
                        ds = dt.datetime.strftime(d - dt.timedelta(days=day), '%Y-%m-%d')
                        testds = dt.datetime.strptime(ds,'%Y-%m-%d')>=dt.datetime.strptime(s,'%Y-%m-%d')
                         # TODO: Sätt maxPixels så att alla får samma storlek?
                        if testds:
                            date = dt.datetime.utcfromtimestamp(ee.Date(im_test.get('system:time_start')).getInfo()['value'] / 1000.)\
                                .strftime('%Y-%m-%d %H:%M')
                            im_test = im_test.addBands(ee.Image.pixelLonLat())
                            im_test = im_test.reduceRegion(reducer=ee.Reducer.toList(), geometry=ar, maxPixels=1e13, scale=70,tileScale=4) #TODO: Sätt maxPixels så att alla får samma storlek?
                            if (np.array(im_test.getInfo().get('tropospheric_NO2_column_number_density')).mean() > 0):
                                found_last = True
                                ### Fetch for all combinations of datasets and bands in ee_dataset
                                dftemp = pd.DataFrame()
                                for j in range(0, len(ee_dataset)):
                                    ds = ee_dataset.loc[j, :].dataset
                                    b = ee_dataset.loc[j, :].bands
                                    IC = get_IC(ds,ar,s,e,b)
                                    im_test = IC.filterDate(start=d - dt.timedelta(days=day), opt_end=d)
                                    im_test = ee.Image(im_test.first()).unmask()
                                    im_test = im_test.addBands(ee.Image.pixelLonLat())
                                    im_test = im_test.reduceRegion(reducer=ee.Reducer.toList(), geometry=ar, maxPixels=1e13,
                                                                   scale=70, tileScale=4)
                                    if False:
                                        print('Value has shape: '+str(np.shape(im_test.getInfo().get(b)))+' and lat has: ' +
                                              str(np.shape(im_test.getInfo().get('latitude'))))
                                    if np.shape(im_test.getInfo().get(b)) == np.shape(im_test.getInfo().get('latitude')):
                                        dft = pd.DataFrame(im_test.getInfo())
                                        dft['date'] = date
                                        dft['id'] = i
                                        val = dft.columns.drop(['latitude', 'longitude', 'date', 'id'])[0]
                                        dft['measurement'] = val
                                        dft = dft.rename(columns={val: 'pixel_value'})
                                        dftemp = dftemp.append(dft,ignore_index=True)
                                if verbose:
                                    print('Found data on date: '+str(d - dt.timedelta(days=day)))
                                    print('For location: ' + str(i) + '. Shape: ' + str(dftemp.shape))
                                return dftemp,date
                            else:
                                day += 1
                        elif (im_test.getInfo() is None) & testds:
                            day += 1
                        elif ~testds:
                            return pd.DataFrame(columns=['latitude','longitude','date','location']),0
                except:
                    if verbose:
                        print('No data found for location: ' + str(i))
                    return pd.DataFrame(columns=['latitude', 'longitude', 'date', 'location']), 0
        else:
            if verbose:
                print('No data found for location: '+ str(i))
            return pd.DataFrame(columns=['latitude', 'longitude', 'date', 'location']), 0
    else:
        if verbose:
            print('No data found for location: ' + str(i))
        return pd.DataFrame(columns=['latitude', 'longitude', 'date', 'location']), 0


def get_data(locations,country,start,end):
    ee_dataset = get_ee_dataset()
    today=dt.datetime.strftime(dt.date.today(),'%Y-%m-%d')
    r = 0.01
    dfm = pd.DataFrame()
    dfs = pd.DataFrame()
    j = 0
    for i in locations.id:
        # location = re.get('https://api.openaq.org/v1/locations/' + str(i))
        # if location.status_code == 200:
        #     l = location.json()['results']['location']
        #     if verbose:
        #         print('Fetching '+ str(j) + '/'+str(locations.id.shape[0])+' location: '+str(i) + ' called: '+str(l))
        #         j += 1
        #     templat = location.json()['results']['coordinates']['latitude']
        #     templon = location.json()['results']['coordinates']['longitude']
        if True:
            dftemp = pd.read_pickle('locations.pkl')
            templat = dftemp[dftemp.id == i].coordinates.values[0].get('latitude')
            templon = dftemp[dftemp.id == i].coordinates.values[0].get('longitude')
            # measurements = re.get('https://api.openaq.org/v1/measurements?coordinates=' + str(templat) + ',' + str(
            #     templon) + '&date_from=' + str(start) + '&date_to=' + str(
            #     end) + '&radius=10000&parameter=no2&limit=1000')
            # if len(measurements.json()['results']) > 0:
            temparea = ee.Geometry.Rectangle(templon + 2 * r, templat + r, templon - 2 * r, templat - r)
            landsat = get_IC(ee_dataset.loc[0,:][0],temparea,start,end,ee_dataset.loc[0,:][1])
            df,date = get_last(landsat,start, end, temparea,i,ee_dataset)
            df['location'] = dftemp[dftemp.id == i].location.values[0]
            dfs = dfs.append(df)
            if date!=0:
                measurements = re.get(
                    'https://api.openaq.org/v1/measurements?coordinates=' + str(templat) + ',' + str(templon) +'&date_from='+
                    dt.datetime.strftime(dt.datetime.strptime(date, '%Y-%m-%d %H:%M') - dt.timedelta(hours=1), '%Y-%m-%d %H:%M')+
                    '&date_to='+dt.datetime.strftime(dt.datetime.strptime(date, '%Y-%m-%d %H:%M') + dt.timedelta(hours=1), '%Y-%m-%d %H:%M')+
                    '&radius=10000&parameter=no2&limit=1000')
                if measurements.status_code == 200:
                    measurements = pd.DataFrame(measurements.json()['results'])
                    dfm = dfm.append(measurements)
                if False:
                    print('dfm: '+str(dfm.shape)+ ' & measurements: '+str(measurements.shape))
            # elif len(measurements.json()['results']) == 0:
            #     print('Error: Found zero measurements for location: '+str(i))

    if dfm.shape[0]>0:
        dfm.date = dfm.date.apply(pd.Series).utc
        dfm.date = pd.to_datetime(dfm.date)
        dfs.date = pd.to_datetime(dfs.date, format='%Y-%m-%d %H:%M', utc = True)
        dfs.date = dfs.date.dt.round(freq='H')
        # dftot = pd.merge(dfm,dfs,how='left',left_on=['location','date'],right_on=['location','date']) #TODO: Jag kanske måste göra en mer manuel merge för datum där jag väljer närmaste datum istället.
        dftott = pd.merge(dfm,dfs,how='left',left_on=['location'],right_on=['location'])
        #if dftott.count()[id] > 0:
        if dftott.count()['id'] > 0:
            print('Number of id in dftott is: ' + str(dftott.count()['id']))
            dftot = pd.DataFrame()
            for i in locations.id:
                df = dftott[dftott.id == i]
                df = df[(df.date_x - df.date_y)== (df.date_x - df.date_y).min()].drop(['date_y','coordinates'],axis='columns')
                dftot = dftot.append(df)
            dftot = dftot.dropna(axis='rows').drop_duplicates()
            locations[['longitude','latitude']] = locations.coordinates.apply(pd.Series)

            if True:
                dfs.to_pickle('dfs '+str( country )+' '+str( today )+'.pkl')
                dfm.to_pickle('dfm '+str( country )+' '+str( today )+'.pkl')
            l = []
            for i in dftot.id.unique():
                for j in dftot.measurement.unique():
                    dft = dftot[dftot.measurement == j]
                    im = dft[dft.id == i][['latitude','longitude','pixel_value']]. \
                        drop_duplicates().pivot('latitude','longitude','pixel_value').values
                    l.append(np.shape(im))
            pd.DataFrame(l).max()
            x_train = np.empty(shape=(1, pd.DataFrame(l).max()[0], pd.DataFrame(l).max()[1], len(dftot.measurement.unique())))
            y_train = np.empty(shape=(1))

            temp = x_train
            for i in dftot.id.unique():
                k=0
                for j in df.measurement.unique():
                    dft = dftot[dftot.measurement == j]
                    im = dft[dft.id == i][['latitude', 'longitude', 'pixel_value']] \
                        .drop_duplicates().pivot('latitude','longitude','pixel_value').values
                    if np.shape(im)[1]<pd.DataFrame(l).max()[1]:
                        im = np.pad(im, pad_width=((0,0),(0,1)), mode='edge')
                    if np.shape(im)[0]<pd.DataFrame(l ).max()[0]:
                        im = np.pad(im, pad_width=((0,1),(0,0)), mode='edge')
                    temp[0, :, :, k] = im
                    k=k+1
                x_train = np.append(x_train, temp, axis=0)
                y_train = np.append(y_train,[dftot[dftot.id == i].value.unique()[0]],axis=0)
            if verbose:
                print('x_train has shape: ' +str(np.shape(x_train)))
                print('y_train has shape: ' +str(np.shape(y_train)))
            np.save('x_train '+str( country )+' '+str( today )+'.npy', x_train)
            np.save('y_train '+str( country )+' '+str( today )+'.npy', y_train)
    elif dfm.shape[0] == 0:
        print('Error: Found zero measurements (dfm)')

    if verbose:
        print('Finished '+str( country )+ ' at '+str( today ))

def get_loc(c):
    dfl = pd.read_pickle('locations.pkl')
    if dfl[dfl.country == c].shape[0] == 0:
        # try:
        #     locations = re.get('https://api.openaq.org/v1/locations?country[]=' + str(c))
        #     if locations.status_code == 200:
        #         dft = pd.DataFrame(locations.json()['results'])
        #     else:
        #         dft = pd.DataFrame()
        # except:
        #     t = 60
        #     print('Pause fetching of location for '+str(t)+' seconds for location '+str(c))
        #     time.sleep(t)
        #     locations = re.get('https://api.openaq.org/v1/locations?country[]=' + str(c))
        #     if locations.status_code == 200:
        #         dft = pd.DataFrame(locations.json()['results'])
        #     else:
        #         dft = pd.DataFrame()
        locations = re.get('https://api.openaq.org/v1/locations?country[]=' + str(c))
        if locations.status_code == 200:
            dft = pd.DataFrame(locations.json()['results'])
        else:
            t = 60*5
            while locations.status_code != 200:
                print('Pause fetching of location for '+str(t)+' seconds for location '+str(c))
                time.sleep(t)
                locations = re.get('https://api.openaq.org/v1/locations?country[]=' + str(c))
            dft = pd.DataFrame(locations.json()['results'])
        dfl = dfl.append(dft)
        dfl.to_pickle('locations.pkl')
    locations = dfl[dfl.country == c]
    return locations

try:
    countries = pd.read_pickle('countries.pkl')
except:
    countries = re.get('https://api.openaq.org/v1/countries')
    countries = pd.DataFrame(countries.json()['results'])
    countries.to_pickle('countries.pkl')

### Define which date ranges to loop the data fetching through.
verbose = True
start_date = '2020-02-01'
delta = 16
end_date = '2020-12-21'
i = 1
start_country = 12
end_country = 94
while dt.datetime.strptime(end_date,'%Y-%m-%d') > dt.datetime.strptime(start_date,'%Y-%m-%d')+dt.timedelta(days=i*delta):
    start = dt.datetime.strftime(dt.datetime.strptime(start_date, '%Y-%m-%d') + dt.timedelta(days=(i - 1) * delta),
                                 '%Y-%m-%d')
    end = dt.datetime.strftime(dt.datetime.strptime(start_date, '%Y-%m-%d') + dt.timedelta(days=i * delta), '%Y-%m-%d')
    if verbose:
        print('Fetching for date range: ' + str(start) + ' to ' + str(end))
    for c in countries.sort_values(by='locations',axis='rows', ascending=False)[start_country:end_country].code:
        # locations = re.get('https://api.openaq.org/v1/locations?country[]='+str(c))
        # if locations.status_code == 200:
        #     locations = pd.DataFrame(locations.json()['results'])
        locations = get_loc(c)
        if locations.shape[0] > 0:
            get_data(locations,c,start,end)
        else:
            print('Location '+str(c)+' could not be fetched.')
    i += 1
