# Get the wind distribution at a site and height over 20 years
# NREL 2020
# Patrick Duffy
# Credit to Michael Rossol for rex; FLORIS dev team for the wind rose

import h5py
import numpy as np
import pandas as pd
import glob
import pickle
import rex
from rex.renewable_resource import WindResource 
from rex.resource_extraction import WindX
from rex.resource import Resource
import pandas as pd
import os
from time import perf_counter


def make_roses_on_eagle(lat, lon, h):
    """Puts together the pieces into one function which can be run in parallel"""
    dump_annual_timeseries(lat, lon, h)
    combine_annual_timeseries(lat, lon, h)
    make_roses_from_csv(lat, lon, h)

def dump_annual_timeseries(lat, lon, h):
    """ Dumps the csvs with time series for the 20 years of wind speed and direction data"""
    # Time the process
    start = perf_counter()

    # Loop through the years
    for year in range(2000,2021):
        # Filename
        filename = '/datasets/WIND/Great_Lakes/Great_Lakes_' + str(year) + '.h5'
        print(year)
        # Open the file
        with WindX(filename) as f:
            # Initialize main dataframes
            site_data = pd.DataFrame()
            time_index = f.time_index
            site_data['time_index'] = time_index
            if year ==  2019:
                print(f.time_index)
            # Retrieve data at the lat, long and , heights
            ws_at_h = 'windspeed_' + str(h) + 'm'
            wd_at_h = 'winddirection_' + str(h) + 'm'
            site_data['ws'] = f.get_lat_lon_ts(ws_at_h, (lat,lon))
            site_data['wd'] = f.get_lat_lon_ts(wd_at_h, (lat,lon))
            print(np.mean(site_data['ws']))

            # save to csv
            latstr = str(round(lat, 5))
            latstr = latstr.replace('.','_')
            lonstr = str(round(lon, 5))
            lonstr = lonstr.replace('.','_')
            site_data.to_csv(str(int(h))+'m_lat'+latstr+'_long'+lonstr+'_'+str(year)+'.csv')

    stop = perf_counter()
    print(stop-start, 'seconds to dump each of 20 years individually')


def combine_annual_timeseries(lat, lon, h):
    "Loop through csvs and appends them to one 20 year one and saves that. Deletes csvs from individual years."
    start = perf_counter()
    years = range(2000,2021)
    bigboi = pd.DataFrame() # the full 20 years of data

    latstr = str(round(lat, 5))
    latstr = latstr.replace('.','_')
    lonstr = str(round(lon, 5))
    lonstr = lonstr.replace('.','_')

    # append each year to the bigboi full 20 years of data
    for year in years:
        print(year)
        yr_df = pd.read_csv(str(int(h)) + 'm_lat' + latstr + '_long' + lonstr + '_' + str(year) + '.csv', index_col=0)
        bigboi = bigboi.append(yr_df)
        print(np.mean(yr_df['ws']))
        os.remove(str(int(h)) + 'm_lat' + latstr + '_long' + lonstr + '_' + str(year) + '.csv')

    # save to csv
    bigboi.to_csv(str(int(h)) + 'm_lat' + latstr + '_long' + lonstr + '_20years.csv', index=False)
    stop = perf_counter()
    print(stop-start, 'seconds to process into one 20 year dataframe')


def make_roses_from_csv(lat, lon, h):
    """ Makes a pickled wind rose from a csv and deletes the 20 year csv"""
    # Name some data
    latstr = str(round(lat, 5))
    latstr = latstr.replace('.','_')
    lonstr = str(round(lon, 5))
    lonstr = lonstr.replace('.','_')
    rose_start = perf_counter()
    csv_path = str(int(h)) + 'm_lat' + latstr + '_long' + lonstr + '_20years.csv'
    
    # Make the pickled python file
    # Load csv data
    d = pd.read_csv(csv_path)
    print('csv data loaded ')
  
    df = make_wind_rose_from_user_data(wd_raw=d['wd'], ws_raw=d['ws'])
    print('Freq tot', df['freq_val'].sum())

    # Save the pickled python file
    pfilename = str(int(h))+'mlat'+latstr+'long'+lonstr+'.p'
    wd=np.arange(0, 360, 5.)
    num_wd = len(wd)
    wd_step = wd[1] - wd[0]
    
    ws=np.arange(0, 31, 1.)
    ws_step = ws[1] - ws[0]
    num_ws = len(ws)
    pickle.dump([num_wd, num_ws, wd_step, ws_step, wd, ws, df], open(pfilename, "wb"))
    os.remove(str(int(h)) + 'm_lat' + latstr + '_long' + lonstr + '_20years.csv')
    roses_stop = perf_counter()
    print('Rose pickled in :', roses_stop-rose_start, ' seconds')


def make_wind_rose_from_user_data(wd_raw,
                                    ws_raw,
                                    *args,
                                    wd=np.arange(0, 360, 5.),
                                    ws=np.arange(0, 31, 1.)):

    """
    Given user-specified arrays of wind direction, wind speed, and 
    additional optional variables, return a dataframe containing the 
    normalized frequency of each set of wind speed, wind direction,
    and any additional variables specified.

    Args:
        wd_raw (array-like): An array-like list of all wind directions 
            used to calculate the normalized frequencies
        wd_raw (array-like): An array-like list of all wind speeds 
            used to calculate the normalized frequencies
        *args: Variable length argument list consisting of alternating 
            string arguments, array-like arguments, and np.array objects. 
            The strings indicate the names of additional variables to include 
            in the wind rose where the array-like arguments contain values of 
            the variables used to calculate the frequencies and the np.array 
            objects specify the bin limits for the variable. 
        wd (np.array, optional): Wind direction bin limits.
            Defaults to np.arange(0, 360, 5.).
        ws (np.array, optional): Wind speed bin limits.
            Defaults to np.arange(0, 26, 1.).

    Returns:
        df (pd.DataFrame): DataFrame with wind speed and direction
            (and any other additional variables specified) values and corresponding 
            frequencies.
    """

    df = pd.DataFrame()

    # convert inputs to np.array
    wd_raw = np.array(wd_raw)
    ws_raw = np.array(ws_raw)

    # Start by simply round and wrapping the wind direction and wind speed columns
    df['wd'] = wrap_360(wd_raw.round())
    df['ws'] = ws_raw.round()

    # Loop through *args and assign new dataframe columns after cutting into possibly irregularly-spaced bins
    for in_var in range(0,len(args),3):
        df[args[in_var]] = np.array(args[in_var+1])
        
        # Cut into bins, make first and last bins extend to -/+ infinity
        var_edges = np.append(0.5*(args[in_var+2][1:]+args[in_var+2][:-1]),np.inf)
        var_edges = np.append(-np.inf,var_edges)
        df[args[in_var]] = pd.cut(df[args[in_var]], var_edges, labels=args[in_var+2])

    # Now group up
    df['freq_val'] = 1.
    df = df.groupby([c for c in df.columns if c != 'freq_val']).sum()
    df['freq_val'] = df.freq_val.astype(float) / df.freq_val.sum()
    df = df.reset_index()

    # Save the df at this point
    #df = df
    
    # Resample onto the provided wind speed and wind direction binnings
    df = resample_wind_speed(df=df,ws=ws)
    df = resample_wind_direction(df=df,wd=wd)

    return df


#########################################################################    
def resample_wind_speed(df, ws=np.arange(0, 31, 1.)):
    """
    Modify the default bins for sorting wind speed.

    Args:
        df (pd.DataFrame): Wind speed data
        ws (np.array, optional): Vector of wind speed bins for
            WindRose. Defaults to np.arange(0, 26, 1.).

    Returns:
        df (pd.DataFrame): Resampled wind speed for WindRose
    """
    # Make a copy of incoming dataframe
    df = df.copy(deep=True)
    # Get the wind step
    ws_step = ws[1] - ws[0]
    # Ws
    ws_edges = (ws - ws_step / 2.0)
    ws_edges = np.append(ws_edges, np.array(ws[-1] + ws_step / 2.0))
    # Cut wind speed onto bins
    df['ws'] = pd.cut(df.ws, ws_edges, labels=ws)
    # Regroup
    df = df.groupby([c for c in df.columns if c != 'freq_val']).sum()

    # Fill nans
    df = df.fillna(0)

    # Reset the index
    df = df.reset_index()
    # Set to float
    for c in [c for c in df.columns if c != 'freq_val']:
        df[c] = df[c].astype(float)
        df[c] = df[c].astype(float)

    return df


def resample_wind_direction(df, wd=np.arange(0, 360, 5.)):
    """
    Modify the default bins for sorting wind direction.
    Args:
        df (pd.DataFrame): Wind direction data
            wd (np.array, optional): Vector of wind direction bins
            for WindRose. Defaults to np.arange(0, 360, 5.).

    Returns:
        df (pd.DataFrame): Resampled wind direction for WindRose
    """
    # Make a copy of incoming dataframe
    df = df.copy(deep=True)

    # Get the wind step
    wd_step = wd[1] - wd[0]

    # Get bin edges
    wd_edges = (wd - wd_step / 2.0)
    wd_edges = np.append(wd_edges, np.array(wd[-1] + wd_step / 2.0))

    # Get the overhangs
    negative_overhang = wd_edges[0]
    positive_overhang = wd_edges[-1] - 360.

    # Need potentially to wrap high angle direction to negative for correct binning
    df['wd'] = wrap_360(df.wd)
    if negative_overhang < 0:
        print('Correcting negative Overhang:%.1f' % negative_overhang)
        df['wd'] = np.where(df.wd.values >= 360. + negative_overhang,
                            df.wd.values - 360., df.wd.values)

    # Check on other side
    if positive_overhang > 0:
        print('Correcting positive Overhang:%.1f' % positive_overhang)
        df['wd'] = np.where(df.wd.values <= positive_overhang,
                            df.wd.values + 360., df.wd.values)

    # Cut into bins
    df['wd'] = pd.cut(df.wd, wd_edges, labels=wd)

    # Regroup
    df = df.groupby([c for c in df.columns if c != 'freq_val']).sum()

    # Fill nans
    df = df.fillna(0)

    # Reset the index
    df = df.reset_index()

    # Set to float Re-wrap
    for c in [c for c in df.columns if c != 'freq_val']:
        df[c] = df[c].astype(float)
        df[c] = df[c].astype(float)
    df['wd'] = wrap_360(df.wd)

    return df

def wrap_360(x):
    """
    Wrap an angle to between 0 and 360

    Returns:
        [array]: angles in specified interval
    """
    x = np.where(x < 0., x + 360., x)
    x = np.where(x >= 360., x - 360., x)
    return (x)
