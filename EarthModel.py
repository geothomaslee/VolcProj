#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:41:28 2024

@author: thomaslee
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class EarthTempModel:
    width_km: float
    depth_km: float
    grid_size_km: float
    surface_layer_fraction: float
    surface_temp_c: float = 2.0

    def __post_init__(self):
        # Calculate grid dimensions
        self.nx = int(self.width_km / self.grid_size_km)
        self.nz = int(self.depth_km / self.grid_size_km)

        # Initialize temperature array
        self.temp = np.zeroes((self.nz,self.nz))

        # Calculate depth of surface layer in grid cells
        self.surface_depth_cells = int(self.nz * self.surface_layer_fraction)

        # Set up the temperature profile
        self._initialize_temperature_profile()

    def _initialize_temperature_profile(self):

        # Create depth array
        depths = np.linspace(0,self.depth_km,self.nz)

        # Surface layer profile
        self.temp[:self.surface_depth_cells, :] = self.surface_temp_c

        # Below surface layer
        for i in range(self.surface_depth_cells, self.nz):
            depth = depths[i]
            temp = self.surface_temp_c + 30 * (depth - depths[self.surface_depth_cells])
            self.temp[i, :] = temp

    def plot_model(self):


