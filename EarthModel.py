#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:41:28 2024

@author: thomaslee
"""

from dataclasses import dataclass
from math import sqrt, floor, ceil
from typing import TypeAlias, Union
Number: TypeAlias = Union[int, float]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import trange

@dataclass
class HeatParam:
    thermal_conductivity: float # Watts per meter*Kelvin
    surface_heat_flow: float # Watts per square meter
    specific_heat: float # Joules per kilogram*Kelvin
    density: float # Kilograms per cubic meter

    def __post_init__(self):
        self.thermal_diffusivity = (self.thermal_conductivity / (
                                    self.density * self.specific_heat))

@dataclass
class MagmaIntrusion:
    depth: float
    radius: float
    temperature: float
    initial_melt_fraction: float=1

@dataclass
class EarthTempModel:
    width_km: float
    depth_km: float
    grid_size_km: float
    surface_layer_fraction: float
    water_temp_c: float = 2.0
    heat_params: HeatParam = None
    magma_intrusion: MagmaIntrusion = None

    def __post_init__(self):
        # Calculate grid dimensions
        self.nx = int(self.width_km / self.grid_size_km)
        self.nz = int(self.depth_km / self.grid_size_km)

        x_steps = np.linspace(0,self.width_km,self.nx)
        z_steps = np.linspace(0,self.depth_km,self.nz)
        self.x_grid, self.z_grid = np.meshgrid(x_steps,z_steps)

        # Initialize temperature array
        self.temp = np.zeros((self.nz,self.nx))

        # Calculate depth of surface layer in grid cells
        self.surface_depth_cells = int(self.nz * self.surface_layer_fraction)

        if self.heat_params is None:
            self.heat_params = HeatParam(
                                    thermal_conductivity=3,
                                    surface_heat_flow=87)

        # Set up the temperature profile
        self._initialize_temperature_profile()

        if self.magma_intrusion:
            self._intrude_magma_body()

        self.cmap = None

    def _initialize_temperature_profile(self):
        # Create depth array
        depths = np.linspace(0,self.depth_km,self.nz)

        # Surface layer profile
        self.temp[:self.surface_depth_cells, :] = self.water_temp_c

        # Below surface layer
        for i in range(self.surface_depth_cells, self.nz):
            depth = depths[i]
            temp = ((self.heat_params.surface_heat_flow /
                     self.heat_params.thermal_conductivity) * depth +
                     self.water_temp_c)
            self.temp[i, :] = temp

    def _distance_between_cells(self, x1: float, y1: float, x2: float, y2: float):
        xdiff = np.abs(x2-x1)
        ydiff = np.abs(y2-y1)
        dist = sqrt((xdiff**2) + (ydiff**2))
        return dist

    def _intrude_magma_body(self):
        x_index = floor(np.shape(self.x_grid)[1] / 2)
        z_index = self._find_nearest_index(val=self.magma_intrusion.depth,
                                           data=self.z_grid[:,x_index])

        intrusion_x = self.x_grid[z_index,x_index]
        intrusion_z = self.z_grid[z_index,x_index]

        for i in range(np.shape(self.temp)[0]):
            for j in range(np.shape(self.temp)[1]):
                cell_x = self.x_grid[i,j]
                cell_z = self.z_grid[i,j]

                dist = self._distance_between_cells(
                            intrusion_x,
                            intrusion_z,
                            cell_x,
                            cell_z)

                if dist <= self.magma_intrusion.radius:
                    self.temp[i,j] = self.magma_intrusion.temperature

    def _create_two_scale_colormap(self):
        lows = plt.cm.YlOrRd(np.linspace(0.2,0.8,128))
        highs = plt.cm.hot(np.linspace(0.3,1,128))
        colors = np.vstack((lows,highs))
        two_scale_cmap = ListedColormap(colors)
        return two_scale_cmap

    def _create_colormap(self):
        min_temp = model.water_temp_c
        max_temp = model.magma_intrusion.temperature

        transition_point = (30 - min_temp) / (max_temp - min_temp)

        custom_colormap = LinearSegmentedColormap.from_list(
            "custom temp",
            [
                (0.0, (0.0,0.0,0.6)),
                (transition_point, (1.0,1,0,1.0)),
                (1.0,(0.6,0.0,0.0))
            ],
            N=256)

        self.cmap = custom_colormap
        return custom_colormap

    def _find_nearest_index(self, val, data):
        difference_array = np.absolute(data-val)
        index = difference_array.argmin()
        return index

    def update_temp(self,new_temp):
        self.temp = new_temp

    def plot(self):
        fig, ax = plt.subplots()

        if self.cmap:
            cmap = self.cmap.copy()
        else:
            cmap = self._create_colormap()
        im = ax.pcolormesh(self.x_grid,self.z_grid,self.temp,cmap=cmap,
                           shading='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right",size="5%",pad=0.05)
        cbar = plt.colorbar(im,cax=cax)
        cbar.set_label('Temperature (C)',fontsize=14)

        ax.invert_yaxis()
        ax.set_aspect('equal')
        plt.show()

@dataclass
class ModelingParams:
    n_steps: Number # Number of steps
    dt: float # Time-step size, seconds
    dx: float # Horizontal grid-spacing, km
    dz: float # Vertical grid-spacing, km
    alpha: float # Thermal diffusivity
    boundary_temp_surface: float # Surface temperature (C)
    boundary_temp_bottom: float # Bottom boundary temperature (C)

def thermal_diffusion_pde(
        params: ModelingParams, # Modeling Parameters
        model: EarthTempModel,
        verbose: bool=False): # Input Earth Model

        temp = model.temp.copy()
        history = [model.temp.copy()]

        dx = params.dx * 1000 # Grids from EarthTempModel are in km
        dz = params.dz * 1000 # Convert to m to maintain consistent units

        # von Neumann stability analysis
        stability_x = params.alpha * params.dt / (dx*dx)
        stability_z = params.alpha * params.dt / (dz*dz)
        total_stability = stability_x + stability_z

        if total_stability > 0.5:
            print('Warning: Solution may be unstable. Try reducing grid size or time step')
            print(f'Stability factor: {total_stability} (should be <= 0.5)')
        else:
            if verbose:
                print('Model should be stable based on von Neumann analysis')
                print(f'Stability factor: {total_stability} (should be <= 0.5)')

        nz, nx = temp.shape

        for step in trange(params.n_steps):
            new_temp = temp.copy()

            # Update interior points using finite difference method
            for i in range(1, nz-1):
                for j in range(1, nx-1):
                    d2tdx2 = (temp[i,j+1] - 2*temp[i,j] + temp[i,j-1]) / (dx*dx)
                    d2tdz2 = (temp[i+1,j] - 2*temp[i,j] + temp[i-1,j]) / (dz*dz)
                    new_temp[i,j] = temp[i,j] + params.dt * params.alpha * (d2tdx2 + d2tdz2)

            # Apply boundary conditions
            new_temp[0,:] = params.boundary_temp_surface
            new_temp[-1,:] = params.boundary_temp_bottom

            new_temp[:,0] = new_temp[:,1] # Left boundary
            new_temp[:,-1] = new_temp[:,-2] # Right boundary

            temp = new_temp
            history.append(temp.copy())

        return temp



heat_params = HeatParam(
                surface_heat_flow=110,
                thermal_conductivity=3,
                specific_heat = 790,
                density = 2800)

intrusion = MagmaIntrusion(
                depth=3,
                radius=1.5,
                temperature=900,
                initial_melt_fraction=1)

model = EarthTempModel(
            width_km=25,
            depth_km=10,
            grid_size_km=0.04,
            surface_layer_fraction=0.05,
            heat_params=heat_params,
            magma_intrusion=intrusion)

params = ModelingParams(
            n_steps = 1000,
            dt = 5*365*24*60*60,
            dx = model.grid_size_km,
            dz = model.grid_size_km,
            alpha = heat_params.thermal_diffusivity,
            boundary_temp_surface=5,
            boundary_temp_bottom=float(model.temp[-1,0]))

model.plot()
model.update_temp(thermal_diffusion_pde(params,model,verbose=True))
model.plot()








