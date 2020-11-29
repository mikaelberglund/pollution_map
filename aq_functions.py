import ee
import pandas as pd

def get_IC(dataset,temparea,start,end,band):
    landsat = ee.ImageCollection(dataset)
    landsat = landsat.filterBounds(temparea)
    landsat = landsat.filterDate(start, end)
    landsat = landsat.select([band])
    return landsat

def get_ee_dataset():
    ee_dataset = pd.DataFrame([
        ["COPERNICUS/S5P/NRTI/L3_NO2",'tropospheric_NO2_column_number_density'],
        ["COPERNICUS/S5P/NRTI/L3_NO2",'stratospheric_NO2_column_number_density'],
        ['COPERNICUS/S5P/NRTI/L3_AER_AI','absorbing_aerosol_index'],
        ['NASA/GLDAS/V021/NOAH/G025/T3H','Rainf_tavg'],
        ['NASA/GLDAS/V021/NOAH/G025/T3H','Wind_f_inst'],
        ['NASA/GLDAS/V021/NOAH/G025/T3H', 'Tair_f_inst'],
        ['NASA/GLDAS/V021/NOAH/G025/T3H', 'Qair_f_inst'],
        ['COPERNICUS/S5P/NRTI/L3_CO','CO_column_number_density']
    ],columns=['dataset','bands'])
    return ee_dataset