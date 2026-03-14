import sys
import math
import numpy as np 
import xarray as xr 
from scipy import special
from pathlib import Path

def get_input(vn, dataflg, lat_range=[20, -20], c=51, m=10, mflg='all', wnx=9, wnxflg='all', rescaleflg=True, chunk_size=1000):
    """
    Optimized MJO data filtering using Meridional Hermite Projection and Zonal FFT.
    Memory-safe via time-chunked processing.
    """
    base_path = Path('/scratch/users/linyao/ML4MJO/data')
    prefix = "rescaled" if rescaleflg else "unscaled"
    out_dir = base_path / f'{prefix}_m{m}{mflg}_wnx{wnx}{wnxflg}'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define filenames
    suffix = "noaa.2x2.1979to2022based1979to2001.nc" if (vn == 'olr' and dataflg == 'noaa') else "1978to2025based1979to2001.nc"
    fn = base_path / "fltano120" / f"{vn}.fltano120.{suffix}"
    fname = out_dir / f"{vn}.fltano120.{suffix}"

    if fname.exists():
        print(f"existing output: {fname}")    
        # fname.unlink()
        return
    
    if not fn.exists():
        print(f"Input file not found: {fn}")
        return

    # -------- Lazy Load Data & Precompute Spatial Variables --------
    with xr.open_dataset(fn) as ds:
        # DO NOT call .values on the whole dataset. 
        # Keep it as a lazy Dask/Xarray object until we slice it.
        lat = ds['lat'].values
        lon = ds['lon'].values
        time_arr = ds['time'].values
        n_times = len(time_arr)

        print('load ds')

        # 1. Setup Meridional (Hermite) Basis once
        if mflg != 'off':
            beta = 2.28e-11
            L = np.sqrt(c / beta)
            y = lat * 110e3 / L  
            dy = abs(lat[0] - lat[1]) * 110e3 / L  
            
            m_list = np.arange(m) 
            phi = []
            for i in m_list:
                p = special.hermite(i)
                Hm = p(y)
                phim = np.exp(-y**2 / 2) * Hm / np.sqrt((2**i) * np.sqrt(np.pi) * math.factorial(i))
                phi.append(phim)
            phi = np.array(phi) # Shape: (m, lat)

        print('phi ready')
        
        # 2. Setup Subsetting Indices once
        lat_mask = (lat >= min(lat_range)) & (lat <= max(lat_range))
        lat_ind = np.where(lat_mask)[0]

        # -------- Process Data in Memory-Safe Time Chunks --------
        processed_chunks = []
        
        print(f"Processing {n_times} time steps in chunks of {chunk_size}...")
        for start in range(0, n_times, chunk_size):
            end = min(start + chunk_size, n_times)
            
            # Load only a tiny slice of time into memory, convert NaNs to 0s
            olr_chunk = ds[vn].isel(time=slice(start, end)).values
            olr_chunk = np.nan_to_num(olr_chunk, nan=0.0) 
            
            # --- Meridional Filtering ---
            olr_re = np.copy(olr_chunk)
            if mflg != 'off':
                # tjl dot mj -> tml
                olr_coeffs = np.einsum('tjl,mj->tml', olr_chunk, phi) * dy
                # tml dot mj -> tjl
                olr_re = np.einsum('tml,mj->tjl', olr_coeffs, phi)
                
                if mflg == 'resi':
                    olr_re = olr_chunk - olr_re

            # --- Zonal Filtering ---
            if wnxflg != 'off':
                coef_fft = np.fft.rfft(olr_re, axis=2)
                
                if wnxflg == 'all':
                    coef_fft[:, :, wnx+1:] = 0.0 
                elif wnxflg == 'resi':
                    coef_fft[:, :, :wnx+1] = 0.0
                    
                olr_re_fft = np.fft.irfft(coef_fft, n=olr_re.shape[2], axis=2)
            else:
                olr_re_fft = olr_re
            
            # --- Subset Spatial Domain ---
            olr_filtered = olr_re_fft[:, lat_ind, :]

            # --- Rescale ---
            if rescaleflg:
                raw_power = np.mean(olr_chunk[:, lat_ind, :]**2, axis=(1, 2), keepdims=True)
                filt_power = np.mean(olr_filtered**2, axis=(1, 2), keepdims=True)
                factor = np.sqrt(raw_power / np.where(filt_power == 0, 1, filt_power))
                olr_filtered *= factor

            processed_chunks.append(olr_filtered)
            print(f"  Processed steps {start} to {end}")

    # -------- Concatenate and Save Result --------
    # Because we subsetted the latitudes, the final array is much smaller and safe to concat
    final_olr = np.concatenate(processed_chunks, axis=0)
    
    ds_out = xr.Dataset(
        {vn: (("time", "lat", "lon"), final_olr.astype(np.float32))},
        coords={
            "time": time_arr,
            "lat": lat[lat_ind],
            "lon": lon
        }
    )
    
    encoding = {vn: {"zlib": True, "complevel": 1}}
    ds_out.to_netcdf(fname, encoding=encoding)
    print(f"Success: {fname}\n")

if __name__ == "__main__":
    mcut = 10 
    wnxcut = 9
    c = 51
    lat_r = 20

    for vn in ['olr']:
        for rescaleflg in [True, False]:
            get_input(vn, 'noaa', lat_range=[lat_r, -lat_r], c=c, m=mcut, mflg='all', wnx=1, wnxflg='off', rescaleflg=rescaleflg)
            get_input(vn, 'noaa', lat_range=[lat_r, -lat_r], c=c, m=mcut, mflg='all', wnx=wnxcut, wnxflg='all', rescaleflg=rescaleflg)
            get_input(vn, 'noaa', lat_range=[lat_r, -lat_r], c=c, m=mcut, mflg='resi', wnx=wnxcut, wnxflg='resi', rescaleflg=rescaleflg)

    for vn in ['olr', 'tcwv', 'u200', 'u850']:
        for rescaleflg in [True, False]:
            get_input(vn, 'era5', lat_range=[lat_r, -lat_r], c=c, m=mcut, mflg='all', wnx=1, wnxflg='off', rescaleflg=rescaleflg)
            get_input(vn, 'era5', lat_range=[lat_r, -lat_r], c=c, m=mcut, mflg='all', wnx=wnxcut, wnxflg='all', rescaleflg=rescaleflg)
            get_input(vn, 'era5', lat_range=[lat_r, -lat_r], c=c, m=mcut, mflg='resi', wnx=wnxcut, wnxflg='resi', rescaleflg=rescaleflg)



