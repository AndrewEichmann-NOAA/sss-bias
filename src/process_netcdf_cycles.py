#!/usr/bin/env python3
"""
Process NetCDF files by 6-hour cycles from obsForge data directory.

This script processes two types of NetCDF files:
1. SSS SMOS L2 data: sss/gdas.tHHz.sss_smos_l2.nc
2. In-situ Argo salt profile data: insitu/gdas.tHHz.insitu_salt_profile_argo.nc

Directory structure: gdas.YYYYMMDD/HH/ocean/
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import pandas as pd


class NetCDFCycleProcessor:
    """Process NetCDF files organized by 6-hour cycles."""
    
    def __init__(self, base_dir):
        """
        Initialize the processor.
        
        Parameters:
        -----------
        base_dir : str
            Base directory containing the gdas.YYYYMMDD subdirectories
        """
        self.base_dir = Path(base_dir)
        self.cycles = ['00', '06', '12', '18']  # 6-hour cycles
        
    def find_cycle_directories(self, start_date=None, end_date=None):
        """
        Find all cycle directories within the date range.
        
        Parameters:
        -----------
        start_date : datetime, optional
            Start date for processing (default: None, processes all)
        end_date : datetime, optional
            End date for processing (default: None, processes all)
            
        Returns:
        --------
        list : List of tuples (date, cycle_hour, directory_path)
        """
        cycle_dirs = []
        
        # Find all gdas.YYYYMMDD directories
        for date_dir in sorted(self.base_dir.glob('gdas.*')):
            if not date_dir.is_dir():
                continue
                
            # Extract date from directory name
            try:
                date_str = date_dir.name.split('.')[1]
                date = datetime.strptime(date_str, '%Y%m%d')
            except (IndexError, ValueError):
                print(f"Skipping invalid directory: {date_dir}")
                continue
            
            # Check if date is within range
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue
            
            # Check for cycle subdirectories
            for cycle in self.cycles:
                cycle_path = date_dir / cycle / 'ocean'
                if cycle_path.exists():
                    cycle_dirs.append((date, cycle, cycle_path))
        
        return cycle_dirs
    
    def load_sss_data(self, cycle_path):
        """
        Load SSS SMOS L2 data.
        
        Parameters:
        -----------
        cycle_path : Path
            Path to the cycle directory
            
        Returns:
        --------
        xarray.Dataset or None
        """
        sss_file = cycle_path / 'sss' / f'gdas.t{cycle_path.parent.name}z.sss_smos_l2.nc'
        
        if not sss_file.exists():
            print(f"SSS file not found: {sss_file}")
            return None
        
        try:
            # Open with group support to access ObsValue group
            ds = xr.open_dataset(sss_file, group='ObsValue')
            print(f"Loaded SSS data: {sss_file}")
            return ds
        except Exception as e:
            print(f"Error loading SSS file {sss_file}: {e}")
            return None
    
    def load_insitu_data(self, cycle_path):
        """
        Load in-situ Argo salt profile data.
        
        Parameters:
        -----------
        cycle_path : Path
            Path to the cycle directory
            
        Returns:
        --------
        tuple: (ObsValue dataset, MetaData dataset) or (None, None)
        """
        insitu_file = cycle_path / 'insitu' / f'gdas.t{cycle_path.parent.name}z.insitu_salt_profile_argo.nc'
        
        if not insitu_file.exists():
            print(f"In-situ file not found: {insitu_file}")
            return None, None
        
        try:
            # Open ObsValue group for salinity data
            ds_obsvalue = xr.open_dataset(insitu_file, group='ObsValue')
            # Open MetaData group for depth data
            ds_metadata = xr.open_dataset(insitu_file, group='MetaData')
            print(f"Loaded in-situ data: {insitu_file}")
            return ds_obsvalue, ds_metadata
        except Exception as e:
            print(f"Error loading in-situ file {insitu_file}: {e}")
            return None, None
    
    def process_cycle(self, date, cycle, cycle_path):
        """
        Process data for a single cycle.
        
        Parameters:
        -----------
        date : datetime
            Date of the cycle
        cycle : str
            Cycle hour ('00', '06', '12', or '18')
        cycle_path : Path
            Path to the cycle directory
        """
        print(f"\n{'='*60}")
        print(f"Processing: {date.strftime('%Y-%m-%d')} {cycle}Z")
        print(f"{'='*60}")
        
        # Load both datasets
        sss_ds = self.load_sss_data(cycle_path)
        insitu_obsvalue, insitu_metadata = self.load_insitu_data(cycle_path)
        
        # Generic processing for SSS data
        if sss_ds is not None:
            self.process_sss(sss_ds, date, cycle)
            sss_ds.close()
        
        # Generic processing for in-situ data
        if insitu_obsvalue is not None and insitu_metadata is not None:
            self.process_insitu(insitu_obsvalue, insitu_metadata, date, cycle)
            insitu_obsvalue.close()
            insitu_metadata.close()
    
    def process_sss(self, ds, date, cycle):
        """
        Generic processing for SSS data.
        
        Parameters:
        -----------
        ds : xarray.Dataset
            SSS dataset (from ObsValue group)
        date : datetime
            Date of the cycle
        cycle : str
            Cycle hour
        """
        print("\n  SSS SMOS L2 Data Summary:")
        print(f"  Dimensions: {dict(ds.sizes)}")
        
        # Access seaSurfaceSalinity from the ObsValue group dataset
        try:
            if 'seaSurfaceSalinity' in ds.variables:
                data = ds['seaSurfaceSalinity'].values
            else:
                print(f"  Available variables: {list(ds.variables)}")
                print(f"  Warning: seaSurfaceSalinity not found")
                return
            
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                print(f"\n  seaSurfaceSalinity Statistics:")
                print(f"    Min:    {np.min(valid_data):.4f} PSU")
                print(f"    Max:    {np.max(valid_data):.4f} PSU")
                print(f"    Mean:   {np.mean(valid_data):.4f} PSU")
                print(f"    Median: {np.median(valid_data):.4f} PSU")
                print(f"    Std:    {np.std(valid_data):.4f} PSU")
                print(f"    Count:  {len(valid_data)}")
            else:
                print(f"\n  seaSurfaceSalinity: No valid data (all NaN)")
        except Exception as e:
            print(f"\n  Error accessing seaSurfaceSalinity: {e}")
            print(f"  Available variables: {list(ds.variables)}")
    
    def process_insitu(self, ds_obsvalue, ds_metadata, date, cycle):
        """
        Generic processing for in-situ data.
        
        Parameters:
        -----------
        ds_obsvalue : xarray.Dataset
            In-situ dataset from ObsValue group
        ds_metadata : xarray.Dataset
            In-situ dataset from MetaData group
        date : datetime
            Date of the cycle
        cycle : str
            Cycle hour
        """
        print("\n  In-situ Argo Salt Profile Data Summary:")
        print(f"  Dimensions: {dict(ds_obsvalue.sizes)}")
        
        # Access salinity from ObsValue and metadata from MetaData
        try:
            if 'salinity' not in ds_obsvalue.variables:
                print(f"  Available variables in ObsValue: {list(ds_obsvalue.variables)}")
                print(f"  Warning: salinity not found")
                return
            
            required_vars = ['depth', 'latitude', 'longitude', 'originalDateTime']
            for var in required_vars:
                if var not in ds_metadata.variables:
                    print(f"  Available variables in MetaData: {list(ds_metadata.variables)}")
                    print(f"  Warning: {var} not found in MetaData")
                    return
            
            salinity_data = ds_obsvalue['salinity'].values
            depth_data = ds_metadata['depth'].values
            latitude_data = ds_metadata['latitude'].values
            longitude_data = ds_metadata['longitude'].values
            datetime_data = ds_metadata['originalDateTime'].values
            
            # Filter by depth <= 5
            depth_mask = depth_data <= 5.0
            
            total_obs = len(salinity_data)
            filtered_obs = np.sum(depth_mask)
            
            print(f"  Total observations: {total_obs}")
            print(f"  Observations with depth <= 5m: {filtered_obs} ({100*filtered_obs/total_obs:.1f}%)")
            
            # Apply depth filter
            salinity_filtered = salinity_data[depth_mask]
            depth_filtered = depth_data[depth_mask]
            latitude_filtered = latitude_data[depth_mask]
            longitude_filtered = longitude_data[depth_mask]
            datetime_filtered = datetime_data[depth_mask]
            
            # Create DataFrame for grouping
            df = pd.DataFrame({
                'latitude': latitude_filtered,
                'longitude': longitude_filtered,
                'datetime': datetime_filtered,
                'salinity': salinity_filtered,
                'depth': depth_filtered
            })
            
            # Remove rows with NaN salinity
            df = df.dropna(subset=['salinity'])
            
            if len(df) == 0:
                print(f"\n  salinity: No valid data after depth filtering (all NaN or depth > 5m)")
                return
            
            # Group by location and datetime, take mean of salinity and depth
            grouped = df.groupby(['latitude', 'longitude', 'datetime']).agg({
                'salinity': 'mean',
                'depth': 'mean'
            }).reset_index()
            
            salinity_means = grouped['salinity'].values
            depth_means = grouped['depth'].values
            
            print(f"  Unique location-datetime combinations: {len(grouped)}")
            
            if len(salinity_means) > 0:
                print(f"\n  salinity Statistics (depth <= 5m, averaged by location-datetime):")
                print(f"    Min:    {np.min(salinity_means):.4f} PSU")
                print(f"    Max:    {np.max(salinity_means):.4f} PSU")
                print(f"    Mean:   {np.mean(salinity_means):.4f} PSU")
                print(f"    Median: {np.median(salinity_means):.4f} PSU")
                print(f"    Std:    {np.std(salinity_means):.4f} PSU")
                print(f"    Count:  {len(salinity_means)}")
                
                print(f"\n  depth Statistics (averaged by location-datetime):")
                print(f"    Min:    {np.min(depth_means):.4f} m")
                print(f"    Max:    {np.max(depth_means):.4f} m")
                print(f"    Mean:   {np.mean(depth_means):.4f} m")
            else:
                print(f"\n  salinity: No valid data after grouping")
        except Exception as e:
            print(f"\n  Error accessing salinity/depth: {e}")
            print(f"  ObsValue variables: {list(ds_obsvalue.variables)}")
            print(f"  MetaData variables: {list(ds_metadata.variables)}")
    
    def run(self, start_date=None, end_date=None):
        """
        Run the processor for all cycles in the date range.
        
        Parameters:
        -----------
        start_date : datetime, optional
            Start date for processing
        end_date : datetime, optional
            End date for processing
        """
        print(f"Scanning directory: {self.base_dir}")
        
        cycle_dirs = self.find_cycle_directories(start_date, end_date)
        
        if not cycle_dirs:
            print("No cycle directories found!")
            return
        
        print(f"Found {len(cycle_dirs)} cycle directories to process")
        
        for date, cycle, cycle_path in cycle_dirs:
            self.process_cycle(date, cycle, cycle_path)


def main():
    """Main entry point."""
    # Configuration
    base_dir = '/Users/afeman/Desktop/work/sss-bias/data/common_obsForge'
    
    # Optional: Specify date range
    start_date = datetime(2024, 1, 1)
#    end_date = datetime(2024, 12, 31)
    end_date = datetime(2024, 1, 31)
    # start_date = None
    # end_date = None
    
    # Create processor and run
    processor = NetCDFCycleProcessor(base_dir)
    processor.run(start_date=start_date, end_date=end_date)


if __name__ == '__main__':
    main()
