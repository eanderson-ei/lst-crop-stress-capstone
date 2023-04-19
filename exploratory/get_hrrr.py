def get_hrrr_point_data(lat, lon, start, days):
    ''' 
    Fetch met data from the HRRR model system.

    Parameters:
    ----------
    lat, lon: float
        Coords from which to pull grid cell data. 
    start: str
        This is the start date for which data will be retrieved.
    days: int
        Number of days of data to retrieve.

    Return:
    -------
    ds_point: xarray dataset
        Met data for specified point.
    '''
    import pandas
    from herbie import FastHerbie

    lat = 42
    lon = -93.7
    sd = "2020-07-01 00:00"

    # Create a range of dates
    DATES = pandas.date_range(
        start=sd,
        periods=days,
        freq="1D",
    )

    # Define forecast lead time (or analysis).
    fxx = range(0, 1)

    # Make FastHerbie Object.
    FH = FastHerbie(DATES, model="hrrr", fxx=fxx)

    # Read a subset of the data with xarray.
    ds = FH.xarray("TMP:2 m", remove_grib=False)

    # Get data values nearest single point
    ds_point = ds.herbie.nearest_points(points=(lon, lat))

    return (ds_point)
