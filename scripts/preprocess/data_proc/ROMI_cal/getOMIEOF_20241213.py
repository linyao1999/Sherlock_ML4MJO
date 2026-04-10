from pathlib import Path
import os.path

import mjoindices.olr_handling as olr
import mjoindices.omi.omi_calculator as omi
import mjoindices.empirical_orthogonal_functions as eof
import mjoindices.principal_components as pc
import mjoindices.evaluation_tools
import numpy as np
import xarray as xr 
import pandas as pd 

# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint cpu --account=m3312

class OLRData:
    """
    This class serves as a container for spatially distributed and temporally resolved OLR data.

    A filled object of this class has to be provided by the user in order to start the OMI calculation.

    :param olr: The OLR data as a 3-dim array. The three dimensions correspond to time, latitude, and longitude, in this
        order.
    :param time: The temporal grid as 1-dim array of :class:`numpy.datetime64` dates.
    :param lat: The latitude grid as 1-dim array.
    :param long: The longitude grid as 1-dim array.
    """

    def __init__(self, olr: np.ndarray, time: np.ndarray, lat: np.ndarray, long: np.ndarray) -> None:
        """
        Initialization of basic variables.
        """
        if olr.shape[0] != time.size:
            raise ValueError('Length of time grid does not fit to first dimension of OLR data cube')
        if olr.shape[1] != lat.size:
            raise ValueError('Length of lat grid does not fit to second dimension of OLR data cube')
        if olr.shape[2] != long.size:
            raise ValueError('Length of long grid does not fit to third dimension of OLR data cube')
        self._olr = olr.copy()
        self._time = time.copy()
        self._lat = lat.copy()
        self._long = long.copy()

    @property
    def olr(self):
        """
        The OLR data as a 3-dim array. The three dimensions correspond to time, latitude, and longitude, in this
        order.
        """
        return self._olr

    @property
    def time(self):
        """
        The temporal grid as 1-dim array of :class:`numpy.datetime64` dates.
        """
        return self._time

    @property
    def lat(self):
        """
        The latitude grid as 1-dim array.
        """
        return self._lat

    @property
    def long(self):
        """
        The longitude grid as 1-dim array.
        """
        return self._long

    def __eq__(self, other: "OLRData") -> bool:
        """
        Override the default Equals behavior
        """
        return (np.all(self.lat == other.lat)
                and np.all(self.long == other.long)
                and np.all(self.time == other.time)
                and np.all(self.olr == other.olr))

    def close(self, other: "OLRData") -> bool:
        """
         Checks equality of two :class:`OLRData` objects, but allows for numerical tolerances.

        :param other: The object to compare with.

        :return: Equality of all members considering the default tolerances of :func:`numpy.allclose`
        """
        return (np.allclose(self.lat, other.lat)
                and np.allclose(self.long, other.long)
                and np.allclose(self.time.astype("float"), other.time.astype("float"))  # allclose does not work with datetime64
                and np.allclose(self.olr, other.olr))


    def get_olr_for_date(self, date: np.datetime64) -> np.ndarray:
        """
        Returns the spatially distributed OLR map for a particular date.

        :param date: The date, which hat to be exactly matched by one of the dates in the OLR time grid.

        :return: The excerpt of the OLR data as a 2-dim array. The two dimensions correspond to
            latitude, and longitude, in this order. Returns None if the date is not contained in the OLR time series.
        """
        cand = self.time == date
        if not np.all(cand == False):  # noqa: E712
            return np.squeeze(self.olr[cand, :, :])
        else:
            return None


    def extract_olr_matrix_for_doy_range(self, center_doy: int, window_length: int = 0,
                                         strict_leap_year_treatment: bool = False) -> np.ndarray:
        """
        Extracts the OLR data, which belongs to all DOYs around one center (center_doy +/- windowlength).

        Keep in mind that the OLR time series might span several years. In this case the center DOY is found more than
        once and the respective window in considered for each year.
        Example: 3 full years of data, centerdoy = 20, and window_length = 4 results in 3*(2*4+1) = 27 entries in the
        time axis

        :param center_doy: The center DOY of the window.
        :param window_length: The window length in DOYs on both sides of the center DOY. Hence, if the window is fully
            covered by the data, one gets 2*window_length + 1 entries per year in the result.
        :param strict_leap_year_treatment: see description in :meth:`mjoindices.tools.find_doy_ranges_in_dates`.

        :return: The excerpt of the OLR data as a 3-dim array. The three dimensions correspond to
            time, latitude, and longitude, in this order.
        """
        inds, doys = tools.find_doy_ranges_in_dates(self.time, center_doy, window_length=window_length,
                                                    strict_leap_year_treatment=strict_leap_year_treatment)
        return self.olr[inds, :, :]


    def save_to_npzfile(self, filename: Path) -> None:
        """
        Saves the data arrays contained in the OLRData object to a numpy file.

        :param filename: The full filename.
        """
        np.savez(filename, olr=self.olr, time=self.time, lat=self.lat, long=self.long)

# ################ Settings. Change with respect to your system ###################
coarse_lat = np.arange(-20., 20.1, 2.0)
coarse_long = np.arange(0., 359.9, 2.0)

# # Choose a spatial grid, on which the values are computed.
olr_data_filename = '/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/daily/olr.day.nc'
eofnpzfile = '/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/scripts/data_proc/ROMI_cal/EOFs_daily1979to2012.npz'
# pctxtfile = '/global/homes/l/linyaoly/ERA5/reanalysis/ERA5_OMI_daily_Oct18.nc'

# Load the OLR data.
# This is the first place to insert your own OLR data, if you want to compute OMI for a different dataset.
ds = xr.open_dataset(olr_data_filename)
time = ds['time']
lon  = ds['lon']
lat  = ds['lat']
# convert unit to W/m2
olr0  = ds['olr'] 

if np.sum(olr0.isnull()).values:
    print('missing values in OLR data!')
    exit()

time = np.asarray(time)
lon  = np.asarray(lon)
lat  = np.asarray(lat)
olr0  = np.asarray(olr0)

raw_olr = OLRData(olr0, time, lat, lon)

del ds 
del olr0 

# Restrict dataset to the original length for the EOF calculation (Kiladis, 2014).
shorter_olr = olr.restrict_time_coverage(raw_olr, np.datetime64('1979-01-01'), np.datetime64('2012-12-31'))

# This is the line, where the spatial grid is changed.
interpolated_olr = olr.interpolate_spatial_grid(shorter_olr, coarse_lat, coarse_long)


# Calculate the eofs. In the postprocessing, the signs of the EOFs are adjusted and the EOF in a period
# around DOY 300 are replaced by an interpolation see Kiladis, 2014).
# The switch strict_leap_year_treatment has major implications only for the EOFs calculated for DOY 366 and causes only
# minor differences for the other DOYs. While the results for setting strict_leap_year_treatment=False are closer to the
# original values, the calculation strict_leap_year_treatment=True is somewhat more stringently implemented using
# built-in datetime functionality.
# See documentation of mjoindices.tools.find_doy_ranges_in_dates() for details.
eofs = omi.calc_eofs_from_olr(interpolated_olr,
                             sign_doy1reference=True,
                             interpolate_eofs=True,
                             strict_leap_year_treatment=False)
eofs.save_all_eofs_to_npzfile(eofnpzfile)

# ### Some diagnostic plots to evaluate the calculated EOFs.
# Load precalculated EOFs first.
eofs = eof.restore_all_eofs_from_npzfile(eofnpzfile)

# # ############## Calculation of the PCs ##################

# # Load EOFs
# eofs = eof.restore_all_eofs_from_npzfile(eofnpzfile)

# # Calculate the PCs.
# # Restrict calculation to the length of the official OMI time series.
# pcs = omi.calculate_pcs_from_olr(raw_olr,
#                                  eofs,
#                                  np.datetime64("1979-01-01"),
#                                  np.datetime64("2019-12-31"),
#                                  use_quick_temporal_filter=False)
# # Save PCs.
# pcs.save_pcs_to_txt_file(pctxtfile)
