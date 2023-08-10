from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
from shapely.geometry import Point, Polygon, mapping
import xarray as xr

from library.FH_Hydrosat import FH_Hydrosat


def read_ameriflux(data_path, header=0, na_values=[-9999], utc_offset=7):
    """
    Reads a standard Ameriflux data file as csv.

    Converts timestamps to datetime, adjusts time to UTC, 
    drops NA values, and sets the start date as index.

    Parameters
    ----------
    data_path: str
        Full file path
    header: int
        Row in which header is found
    na_values: list
        List of null values for the dataset
    utc_offset: int
        Number of hours offset from UTC dataset is in
    
    Returns
    -------
    DataFrame
        The dataset as a pandas DataFrame
    """
    # Read data if path exists
    try:
        df = pd.read_csv(data_path, header=header, na_values=na_values)
    except (FileNotFoundError):
        print(f'File at path {data_path} not found')

    # Save value column names
    value_cols = df.columns[2:]

    # Convert timestamp objects
    df['start'] = df['TIMESTAMP_START'].apply(
        lambda x: datetime.strptime(str(x), "%Y%m%d%H%M.0")
        )
    df['end'] = df['TIMESTAMP_END'].apply(
        lambda x: datetime.strptime(str(x), "%Y%m%d%H%M.0")
        )
    
    # Convert to UTC time
    df['start'] = df['start'] + timedelta(hours=utc_offset)
    df['end'] = df['end'] + timedelta(hours=utc_offset)
    df['start'] = df['start'].dt.tz_localize('UTC')
    df['end'] = df['end'].dt.tz_localize('UTC')

    # Drop NA
    df = df.dropna(subset=value_cols, how='all')

    df = df.set_index('start')
    col_order = (['end', 'TIMESTAMP_START', 'TIMESTAMP_END'] 
                 + value_cols.to_list())
    df = df[col_order]

    return df


def connect_to_collection(catalog, aoi, collection, start_date, end_date):
    if type(aoi) == Point:
        search = catalog.search(
            collections=collection,
            intersects=aoi,
            datetime=[start_date, end_date],
            max_items=1000
        )
    elif type(aoi) == Polygon:
        search = catalog.search(
            collections=collection,
            aoi=aoi,
            datetime=[start_date, end_date],
            max_items=1000
        )

    return search.item_collection()


def read_and_clip_items(items, asset, clip_gdf, dims=("band", "y", "x")):
    itemjson = items.to_dict()
    features = itemjson['features']

    asset_list = {
        f["properties"]["datetime"] : f["assets"][asset]["href"] 
        for f in features
        }

    da_list = []

    for timestamp in sorted(asset_list.keys()):   
        filepath = asset_list[timestamp]
        with rasterio.open(filepath) as src:
            out_image, out_transform = rasterio.mask.mask(
                src, clip_gdf.geometry.apply(mapping), crop=True)
            out_meta = src.meta

        # Update metadata after mask
        out_meta.update({"driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform})
        try:
            timestamp_naive = datetime.strptime(
                timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
        except(ValueError):
            timestamp_naive = datetime.strptime(
                timestamp, "%Y-%m-%dT%H:%M:%SZ")

        # Assign coordinate values from out_meta
        coords = {
            "x": out_meta["transform"].c + out_meta["transform"].a 
            * np.arange(out_meta["width"]),
            "y": out_meta["transform"].f + out_meta["transform"].e 
            * np.arange(out_meta["height"]),
            "time": timestamp_naive
        }

        # Create a new DataArray with updated coordinates
        da = xr.DataArray(
            out_image.squeeze(),
            coords=coords,
            dims=dims,
        )

        # Append to da_list
        da_list.append(da)

    # Concatenate all data arrays
    rio_clip = xr.concat(da_list, dim='time')

    # Set the CRS (Coordinate Reference System) for the DataArray
    rio_clip = rio_clip.rio.set_crs(out_meta["crs"])

    return rio_clip


def ndvi_from_collection(items, geom_point, tolerance, red_band, nir_band, 
                         name):
    assets = items[0].to_dict()['assets'].keys()
    if len(items) > 0 and 'surface_reflectance' in assets:
        res_full = FH_Hydrosat(items, asset='surface_reflectance')
        res_dt = res_full.datetime

        red_ts = res_full.point_time_series_from_items(
            geom_point, tol=tolerance, nproc=6, band=red_band)
        nir_ts = res_full.point_time_series_from_items(
            geom_point, tol=tolerance, nproc=6, band=nir_band)

        ndvi_ts = (
            (np.array(nir_ts) - np.array(red_ts)) 
            / (np.array(nir_ts) + np.array(red_ts))
        )
        ndvi_dt = res_dt

        ndvi_df = pd.DataFrame(
            {'ndvi': ndvi_ts,
             'datetime': pd.to_datetime(ndvi_dt)}).sort_values(by='datetime')
        ndvi_df.index = (
            pd.to_datetime(ndvi_df['datetime'].dt.strftime('%Y-%m-%d'))
        )

        ndvi_series = ndvi_df['ndvi'].astype('float')
        ndvi_series.name = name

        return ndvi_series