# Run get_resource_one_site in parallel on Eagle
# NREL 2020
# Patrick Duffy

import os
import pandas as pd
from rex.utilities.execution import SpawnProcessPool
from get_resource_one_site import dump_annual_timeseries, combine_annual_timeseries, make_roses_from_csv


def make_roses_on_eagle(lat, lon, h):
    """Puts together the pieces into one function which can be run in parallel"""
    dump_annual_timeseries(lat, lon, h)
    combine_annual_timeseries(lat, lon, h)
    make_roses_from_csv(lat, lon, h)

if __name__=='__main__':
    # Load list of coordinates
    sites = pd.read_csv('great_lakes_sites.csv')

    # Spawn parallel processes to produce csvs for each site
    with SpawnProcessPool() as ex:
        h = 167
        for i in range(len(sites['latitude'])):
            lat = sites['latitude'][i]
            lon = sites['longitude'][i]
            print(lat, lon)
            ex.submit(make_roses_on_eagle, lat, lon, h)
            #ex.submit(make_roses_from_csv, lat, lon, h)
