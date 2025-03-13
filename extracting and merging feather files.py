#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 23:39:07 2025

@author: charliemurray
"""

import pandas as pd
import os
from pathlib import Path

def merge_feather_files(root_folder, feather_file="merged_data.feather", csv_file="merged_data.csv"):
    """ Recursively merges all Feather files in subdirectories and saves both Feather and CSV formats. """
    feather_files = list(Path(root_folder).rglob("*.feather"))  # Find all Feather files
    
    if not feather_files:
        print("❌ No Feather files found.")
        return None

    # Read and merge all Feather files
    dataframes = [pd.read_feather(str(f)) for f in feather_files]
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Save as Feather
    merged_df.to_feather(feather_file)
    print(f"✅ Merged {len(feather_files)} Feather files into: {feather_file}")

    # Save as CSV
    merged_df.to_csv(csv_file, index=False)
    print(f"✅ Also saved as CSV: {csv_file}")

    return merged_df

# Usage Example:
merged_data = merge_feather_files("/Users/charliemurray/Documents/all_cohesionless_data/")
