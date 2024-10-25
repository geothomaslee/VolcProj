#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:41:28 2024

@author: thomaslee
"""

import os
import io
import pickle
from glob import glob
from dataclasses import dataclass, field
from math import sqrt, floor
from typing import TypeAlias, Union
Number: TypeAlias = Union[int, float]

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import trange

@dataclass
class HeatParam:
    """Background heat params representing the "general" domain"""
    thermal_conductivity: float # Watts per meter*Kelvin
    surface_heat_flow: float # Watts per square meter
    specific_heat: float # Joules per kilogram*Kelvin
    density: float # Kilograms per cubic meter

    def __post_init__(self):
        self.thermal_diffusivity = (self.thermal_conductivity / (
                                    self.density * self.specific_heat))

@dataclass
class MagmaIntrusion:
    depth: float # Depth of the center of the magma chamber
    radius: float # Radius of the magma chamber
    temperature: float # Initial temperature of magma
    initial_melt_fraction: float=1 # Melt fraction

@dataclass
class SaturatedLayer:
    """Properties of the saturated layer at the top of the model domain"""
    thickness_fraction: float # Thickness as a fraction of model domain thickness
    water_temperature: float # Celsius
    thermal_conductivity: float # Watters per meter*kelvin
    specific_heat: float # Joules per kilogram*Kelvin
    density: float # Kilograms per cubic meter
    residence_time: float # Years

    def __post_init__(self):
        self.thermal_diffusivity = (self.thermal_conductivity / (
                                    self.density * self.specific_heat))


@dataclass
class EarthTempModel:
    width_km: float
    depth_km: float
    grid_size_km: float
    surface_temp_c: float = 2.0
    heat_params: HeatParam = None
    magma_intrusion: MagmaIntrusion = None
    saturated_layer: SaturatedLayer = None

    def __post_init__(self):
        # Calculate grid dimensions
        self.nx = int(self.width_km / self.grid_size_km)
        self.nz = int(self.depth_km / self.grid_size_km)

        x_steps = np.linspace(0,self.width_km,self.nx)
        z_steps = np.linspace(0,self.depth_km,self.nz)
        self.x_grid, self.z_grid = np.meshgrid(x_steps,z_steps)

        # Initialize temperature array, but fill with 0
        self.temp = np.zeros((self.nz,self.nx))

        if self.saturated_layer is None:
            self.saturated_layer = SaturatedLayer(
                                        thickness_fraction=0.05,
                                        water_temperature=2,
                                        thermal_conductivity=3,
                                        specific_heat=790,
                                        density=2800,
                                        residence_time=1)

        self.water_temp_c = self.saturated_layer.water_temperature
        self.surface_layer_fraction = self.saturated_layer.thickness_fraction

        # Calculate depth of surface layer in grid cells
        self.surface_depth_cells = int(self.nz * self.surface_layer_fraction)

        if self.heat_params is None:
            self.heat_params = HeatParam(
                                    thermal_conductivity=3,
                                    surface_heat_flow=87)

        # Enforce diffusivity has not yet been calculated
        self._has_initialized_diffusivity = False

        # Set up the temperature profile
        self._initialize_temperature_profile()

        # Set up thermal diffusivity array
        self._initialize_thermal_diffusivity()

        if self.magma_intrusion:
            self._intrude_magma_body()

        self._enforce_surface_layer()

        self.initial_melted_cells = self._get_melted_cells()

        self.cmap = None

    def _initialize_temperature_profile(self):
        """Create background basic geotherm profile"""
        # Create depth array
        depths = np.linspace(0,self.depth_km,self.nz)

        # Surface layer profile
        self._enforce_surface_layer()

        # Below surface layer
        for i in range(self.surface_depth_cells, self.nz):
            depth = depths[i]
            temp = ((self.heat_params.surface_heat_flow /
                     self.heat_params.thermal_conductivity) * depth +
                     self.surface_temp_c)
            self.temp[i, :] = temp

    def _initialize_thermal_diffusivity(self):
        thermal_diffusivity = np.zeros((self.nz,self.nx))
        thermal_diffusivity.fill(self.heat_params.thermal_diffusivity)
        self.thermal_diffusivity = thermal_diffusivity
        self._has_initialized_diffusivity = True

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

    def _enforce_surface_layer(self):
        self.temp[:self.surface_depth_cells, :] = self.saturated_layer.water_temperature

        if self._has_initialized_diffusivity:
            self.thermal_diffusivity[:self.surface_depth_cells, :] = (
                self.saturated_layer.thermal_diffusivity)

    def _get_melted_cells(self, solidus: float=650.0):
        count = (self.temp > solidus).sum()
        self.melted_cells = count
        return count

    def _create_two_scale_colormap(self):
        lows = plt.cm.YlOrRd(np.linspace(0.2,0.8,128))
        highs = plt.cm.hot(np.linspace(0.3,1,128))
        colors = np.vstack((lows,highs))
        two_scale_cmap = ListedColormap(colors)
        return two_scale_cmap

    def _create_colormap(self):
        min_temp = self.saturated_layer.water_temperature
        max_temp = self.magma_intrusion.temperature

        transition_point = (30 - min_temp) / (max_temp - min_temp)

        custom_colormap = LinearSegmentedColormap.from_list(
            "custom temp",
            [
                (0.0, (0.0,0.0,0.6)),
                (transition_point, (1.0,1,0,1.0)),
                (1.0,(0.6,0.0,0.0))
            ],
            N=1024)

        self.cmap = custom_colormap
        return custom_colormap

    def _find_nearest_index(self, val, data):
        difference_array = np.absolute(data-val)
        index = difference_array.argmin()
        return index

    def update_temp(self,new_temp,dt=None):
        self.temp = new_temp
        """
        if dt is not None:
            if dt % self.saturated_layer.residence_time == 0:
                self._enforce_surface_layer()
        """

    def plot(self,show=True,frame=None,dt=0,param_dict=None):
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

        self._get_melted_cells()
        melt_percent = (self.melted_cells / self.initial_melted_cells)*100

        ax.set_title(f'Percentage of Original Melt Still Above Solidus: {melt_percent:.2f}%')

        if frame is not None:
            fig.suptitle(f'Time Elapsed: {frame*dt} Years',y=0.8)

        if param_dict is not None:
            text = '\n'.join(f'{key}: {value}' for key, value in param_dict.items())

            text_box = AnchoredText(text,
                                    loc='lower left',
                                    #bbox_to_anchor=(0.05,0.0),
                                    prop=dict(size=8),
                                    bbox_transform=ax.transAxes,
                                    frameon=True)

            ax.add_artist(text_box)

        plt.tight_layout()

        if show is True:
            plt.show()

        self.fig = fig
        self.ax = ax

        return fig

@dataclass
class ModelingParams:
    n_steps: Number # Number of steps
    dt: float # Time-step size, years
    dx: float # Horizontal grid-spacing, km
    dz: float # Vertical grid-spacing, km
    boundary_temp_surface: float # Surface temperature (C)
    boundary_temp_bottom: float # Bottom boundary temperature (C)

@dataclass
class Animator:
    length: float # GIF length in seconds
    frames: [Image.Image] = field(default_factory=list)

    def _save_figure_to_frame(self,fig):
        buf = io.BytesIO()
        fig.savefig(buf,format='png',bbox_inches='tight',dpi=100)
        buf.seek(0)
        frame = Image.open(buf)
        self.frames.append(frame)
        return frame

    def _save_animation(self, path: str):
        #duration = int((self.length * 100) / len(self.frames))
        self.frames[0].save(
            path,
            save_all=True,
            append_images=self.frames[1:],
            optimize=False,
            duration=5,
            loop=0)

    def close(self):
        self.frames = []

def thermal_diffusion_pde(
        params: ModelingParams, # Modeling Parameters
        model: EarthTempModel,
        animate: bool=True,
        verbose: bool=False): # Input Earth Model

        if model.saturated_layer.residence_time < params.dt:
            model.saturated_layer.residence_time = params.dt

        if model.saturated_layer.residence_time % params.dt != 0:
            raise ValueError('Saturated Layer Residence Time Must be Divisible By Dt')

        temp = model.temp.copy()
        history = [model.temp.copy()]

        dx = params.dx * 1000 # Grids from EarthTempModel are in km
        dz = params.dz * 1000 # Convert to m to maintain consistent units

        dt = params.dt * 365 * 24 * 60 * 60

        max_alpha = np.max(model.thermal_diffusivity)

        # von Neumann stability analysis
        stability_x = max_alpha * dt / (dx*dx)
        stability_z = max_alpha * dt / (dz*dz)
        total_stability = stability_x + stability_z

        if total_stability > 0.5:
            print('Warning: Solution may be unstable. Try reducing time_step or increasing grid size')
            print(f'Stability factor: {total_stability} (should be <= 0.5)')
        else:
            if verbose:
                print('Model should be stable based on von Neumann analysis')
                print(f'Stability factor: {total_stability} (should be <= 0.5)')

        nz, nx = temp.shape

        if animate is True:
            animator = Animator(length=5)

        param_dictionary = {'Time Step' : f'{params.dt} Years',
                            'Saturated Layer Thickness' : f'{model.saturated_layer.thickness_fraction * model.depth_km} km',
                            'Groundwater Temp' : f'{model.saturated_layer.water_temperature} C',
                            'Water-Saturated Diffusivity' : f'{round(model.saturated_layer.thermal_diffusivity,10)} (m^2/s)',
                            'Solidus Temp' : '650 C'}

        melt_percents = []
        times = np.arange(0,params.dt*params.n_steps,params.dt)
        surface_layer_max_temps = []

        for step in trange(params.n_steps):

            current_surface_max = np.max(temp[:model.surface_depth_cells,:])
            surface_layer_max_temps.append(current_surface_max)


            new_temp = temp.copy()

            if (step*params.dt) % model.saturated_layer.residence_time == 0:
                if verbose:
                    print(f'\nStep: {step}: Enforcing surface layer reset')
                    print(f'Max surface temperature before reset: {current_surface_max}')
                model._enforce_surface_layer()
                if verbose:
                    print(f'Max surface temp after reset: {np.max(temp[:model.surface_depth_cells, :])}')

            # Update interior points using finite difference method
            for i in range(1, nz-1):
                for j in range(1, nx-1):
                    alpha = model.thermal_diffusivity[i,j]
                    d2tdx2 = (temp[i,j+1] - 2*temp[i,j] + temp[i,j-1]) / (dx*dx)
                    d2tdz2 = (temp[i+1,j] - 2*temp[i,j] + temp[i-1,j]) / (dz*dz)
                    new_temp[i,j] = temp[i,j] + dt * alpha * (d2tdx2 + d2tdz2)

            # Apply boundary conditions
            #new_temp[0,:] = params.boundary_temp_surface # Don't think I need to enforce this condition?
            new_temp[-1,:] = params.boundary_temp_bottom

            new_temp[:,0] = new_temp[:,1] # Left boundary
            new_temp[:,-1] = new_temp[:,-2] # Right boundary

            temp = new_temp
            history.append(temp.copy())

            cwd = os.getcwd()
            if not os.path.isdir(f'{cwd}/Frames'):
                os.mkdir(f'{cwd}/Frames')

            model.update_temp(new_temp,dt=params.dt)

            fig = model.plot(show=False,
                             frame=step,
                             dt=params.dt,
                             param_dict=param_dictionary)

            if animate is True:
                animator._save_figure_to_frame(fig)

            plt.close(fig)

            melt_percent = (model.melted_cells / model.initial_melted_cells)*100
            melt_percents.append(melt_percent)

        if animate is True:
            animator._save_animation(f'{cwd}/Frames/ModelAnimation.gif')
            animator.close()

        return times,melt_percents

def plot_melt_history(times,melt_percent):
    fig, ax = plt.subplots()
    ax.plot(times,melt_percent)

    ax.set_xlabel('Years')
    ax.set_ylabel('Melt Fraction')
    fig.title('Percentage of Original Intrusion Still Above Solidus (>650C)')

    plt.show()

def saveObj(obj, filename):
    """Quick function for pickling a file"""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def loadObj(filename):
    """Quick function for pickling a file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)



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

for residence_time in [1,5,10,20,60]:
    saturated_layer = SaturatedLayer(
                        thickness_fraction=0.05,
                        water_temperature=0,
                        thermal_conductivity=30,
                        specific_heat=7900,
                        density=2800,
                        residence_time=residence_time)

    model = EarthTempModel(
                width_km=25,
                depth_km=10,
                grid_size_km=0.1,
                heat_params=heat_params,
                magma_intrusion=intrusion,
                saturated_layer=saturated_layer)

    params = ModelingParams(
                n_steps = 1000,
                dt = 1,
                dx = model.grid_size_km,
                dz = model.grid_size_km,
                boundary_temp_surface=5,
                boundary_temp_bottom=float(model.temp[-1,0]))


    times, melt_percent = thermal_diffusion_pde(params,model,animate=False,verbose=False)
    saveObj(times,filename=f'{os.getcwd()}/times_residence_time_{residence_time}.pkl')
    saveObj(melt_percent,filename=f'{os.getcwd()}/melt_percent_residence_time_{residence_time}.pkl')



fig, ax = plt.subplots()
time_files = sorted(glob(f'{os.getcwd()}/times_residence_time*.pkl'),key=lambda x: int(x.split('.')[-2].split('_')[-1]))
melt_percent_files =  sorted(glob(f'{os.getcwd()}/melt_percent_residence_time*.pkl'),key=lambda x: int(x.split('.')[-2].split('_')[-1]))


for i, time_pkl in enumerate(time_files):
    times = loadObj(time_pkl)
    melt_percent = loadObj(melt_percent_files[i])

    if i == 0:
        previous_melt_percent = []

    if i > 0:
        change = [melt_percent[x] - previous_melt_percent[x] for x in range(len(melt_percent))]
        print(change)

    residence_time = int(time_pkl.split('.')[-2].split('_')[-1])

    ax.plot(times,melt_percent,label=f'{residence_time} Years')

    previous_melt_percent = melt_percent.copy()

ax.legend()
plt.show()





