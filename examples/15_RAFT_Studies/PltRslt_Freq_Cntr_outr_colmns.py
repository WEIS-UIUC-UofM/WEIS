"""
Simple script to show how to grab all cases from a DOE run. User can then
postprocess or plot further.
"""

import glob
import os
import sys
import time
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import LinearLocator
from matplotlib import cm

import numpy as np
import pandas as pd 
import multiprocessing as mp 
import pickle as pkl
import matplotlib.pyplot as plt

import openmdao.api as om
from weis.aeroelasticse import FileTools

def load_OMsql(log):
    print('loading {}'.format(log))
    cr = om.CaseReader(log)
    rec_data = {}
    driver_cases = cr.list_cases('driver')
    cases = cr.get_cases('driver')
    for case in cases:
        for key in case.outputs.keys():
            if key not in rec_data:
                rec_data[key] = []
            rec_data[key].append(case[key])
        
    return rec_data


if __name__ == '__main__':
    
    # Multiprocssing?
    post_multi = False

    # sql outfile directory
    run_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))# os.path.dirname(os.path.realpath(__file__))   
    #output_dir = os.path.join(run_dir, "outputs/15_RAFT_optimization_freq_step1_Cntr_Clmns_and_Outr_Clmns_Reduced_included") #Design of Experiment
    output_dir = os.path.join(run_dir, "outputs/15_RAFT_optimization_freq_step1") #opt coByla
    doe_logs = glob.glob(os.path.join(output_dir,'log_opt.sql*'))
    if len(doe_logs) < 1:
        raise FileExistsError('No output logs to post process!')
        
    # Remove the 'meta' log
    for idx, log in enumerate(doe_logs):
        if 'meta' in log:
            doe_logs.pop(idx)

    # run multiprocessing
    if post_multi:
        cores = mp.cpu_count()
        pool = mp.Pool(min(len(doe_logs), cores))

        # load sql file
        outdata = pool.map(load_OMsql, doe_logs)
        pool.close()
        pool.join()
    # no multiprocessing
    else:
        outdata = [load_OMsql(log) for log in doe_logs]

    collected_data = {}
    for data in outdata:
        for key in data.keys():
            if key not in collected_data.keys():
                collected_data[key] = []

            for key_idx, _ in enumerate(data[key]):
                if key=='costs.turbine_number':
                    data[key][key_idx]=int(data[key][key_idx])
                if isinstance(data[key][key_idx], int):
                    collected_data[key].append(np.array(data[key][key_idx]))
                elif len(data[key][key_idx]) == 1:
                    try:
                        collected_data[key].append(np.array(data[key][key_idx][0]))
                    except:
                        collected_data[key].append(np.array(data[key][key_idx]))
                else:
                    collected_data[key].append(np.array(data[key][key_idx]))

    df = pd.DataFrame.from_dict(collected_data)

    #-------------------------------------Design Variables------------------------------------
    main_column_outer_diameter_vec=np.squeeze(outdata[0]['floating.memgrp0.outer_diameter_in'])
    side_columns_outer_diameter_vec=np.squeeze(outdata[0]['floating.memgrp1.outer_diameter_in'])
    tower_section_height_vec=np.squeeze(outdata[0]['towerse.tower_section_height'])
    tower_length=np.squeeze(outdata[0]['tower_grid.height']) # == np.sum(tower_section_height_vec,axis=1) default turbine: 15 t0 125 --> 1e30 m
    peak_shaving_vec= np.squeeze(outdata[0]['tune_rosco_ivc.ps_percent'])
    joints_xyz_matrix=np.squeeze(outdata[0]['floating.joints_xyz'])
    platfom_columns_height_vec=np.array([np.squeeze(outdata[0]['floating.joints_xyz'])[i][1][2]-np.squeeze(outdata[0]['floating.joints_xyz'])[i][0][2] for i in np.arange(len(main_column_outer_diameter_vec)) ])
    radial_platfom_columns_spacing_vec=np.array([np.sqrt(np.squeeze(outdata[0]['floating.joints_xyz'])[i][2][0]**2+np.squeeze(outdata[0]['floating.joints_xyz'])[i][2][1]**2) for i in np.arange(len(main_column_outer_diameter_vec))])
    rotor_diameter_vec=np.squeeze(outdata[0]['configuration.rotor_diameter_user'])
    tower_diameter_matriix=np.squeeze(outdata[0]['tower.diameter'])
    #--------------------------------------------------------------------------------------

    #-------------------------------------Constraints------------------------------------
    Max_PtfmPitch_vc=np.squeeze(outdata[0]['raft.Max_PtfmPitch'])
    Std_PtfmPitch_vec=np.squeeze(outdata[0]['raft.Std_PtfmPitch'])
    height_constraint_vec=np.squeeze(outdata[0]['towerse.height_constraint'])
    tower_slope_vec=np.squeeze(outdata[0]['towerse.slope'])
    nacelle_acceleration_vec=np.squeeze(outdata[0]['raft.response_nacelle acceleration'])
    #--------------------------------------------------------------------------------------

    #-------------------------------------Objectives------------------------------------
    pltfrm_mass_vec=np.squeeze(outdata[0]['raft.platform_mass'])
    AEP_vec=np.squeeze(outdata[0]['rotorse.rp.AEP'])
    cost_vec=np.squeeze(outdata[0]['tcc.turbine_cost'])
    turbine_cost_kW_vec=np.squeeze(outdata[0]['tcc.turbine_cost_kW'])
    turbine_mass_tcc_vec=np.squeeze(outdata[0]['tcc.turbine_mass_tcc'])
    finance_lcoe_vec=np.squeeze(outdata[0]['financese.lcoe']) #what is this?
    Cp_matrix=np.squeeze(outdata[0]['rotorse.rp.powercurve.Cp'])
    blade_mass_vec=np.squeeze(outdata[0]['rotorse.blade_mass'])
    tower_mass_vec=np.squeeze(outdata[0]['towerse.tower_mass'])
    tower_cost_vec=np.squeeze(outdata[0]['tcc.tower_cost'])
    floatingse_structurall_mass=np.squeeze(outdata[0]['floatingse.system_structural_mass'])
    floatingse_platform_mass_vec=np.squeeze(outdata[0]['floatingse.platform_mass'])
    main_column_mass=np.squeeze(outdata[0]['floatingse.member0.structural_cost'])
    side_columns_mass=np.squeeze(outdata[0]['floatingse.member1.structural_cost'])
    tower_top_deflection_vec=np.squeeze(outdata[0]['towerse.tower.top_deflection'])
    tower_deflection_matrix=np.squeeze(outdata[0]['towerse.tower.tower_deflection'])
    #--------------------------------------------------------------------------------------

    """data = {
    'main_column_outer_diameter_vec': main_column_outer_diameter_vec,
    'side_columns_outer_diameter_vec': side_columns_outer_diameter_vec,
    'tower_section_height_vec': tower_section_height_vec,
    'tower_length': tower_length,
    'peak_shaving_vec': peak_shaving_vec,
    'joints_xyz_matrix': joints_xyz_matrix,
    'platfom_columns_height_vec': platfom_columns_height_vec,
    'radial_platfom_columns_spacing_vec': radial_platfom_columns_spacing_vec,
    'rotor_diameter_vec': rotor_diameter_vec,
    'tower_diameter_matriix': tower_diameter_matriix,
    'Max_PtfmPitch_vc': Max_PtfmPitch_vc,
    'Std_PtfmPitch_vec': Std_PtfmPitch_vec,
    'height_constraint_vec': height_constraint_vec,
    'tower_slope_vec': tower_slope_vec,
    'nacelle_acceleration_vec': nacelle_acceleration_vec,
    'pltfrm_mass_vec': pltfrm_mass_vec,
    'AEP_vec': AEP_vec,
    'cost_vec': cost_vec,
    'turbine_cost_kW_vec': turbine_cost_kW_vec,
    'turbine_mass_tcc_vec': turbine_mass_tcc_vec,
    'finance_lcoe_vec': finance_lcoe_vec,
    'Cp_matrix': Cp_matrix,
    'blade_mass_vec': blade_mass_vec,
    'tower_mass_vec': tower_mass_vec,
    'tower_cost_vec': tower_cost_vec,
    'floatingse_structurall_mass': floatingse_structurall_mass,
    'main_column_mass': main_column_mass,
    'side_columns_mass': side_columns_mass,
    'tower_top_deflection_vec': tower_top_deflection_vec,
    'tower_deflection_matrix': tower_deflection_matrix
}

    with open('data.pkl', 'wb') as f:
        pkl.dump(data, f)"""


    main_clmns_mesh=main_column_outer_diameter_vec.reshape((41,41))[11:,:]
    side_clmns_mesh=side_columns_outer_diameter_vec.reshape((41,41))[11:,:]
    Max_PtfmPitch_vc_mesh=Max_PtfmPitch_vc.reshape((41,41))[11:,:]
    Std_PtfmPitch_vec_mesh=Std_PtfmPitch_vec.reshape((41,41))[11:,:]
    pltfrm_mass_vec_mesh=pltfrm_mass_vec.reshape((41,41))[11:,:]
    AEP_vec_mesh=AEP_vec.reshape((41,41))[11:,:]
    cost_vec_mesh=cost_vec.reshape((41,41))[11:,:]
    finance_lcoe_vec_mesh=finance_lcoe_vec.reshape((41,41))[11:,:]
    floatingse_structurall_mass_mesh=floatingse_structurall_mass.reshape((41,41))[11:,:]
    main_column_mass_mesh=main_column_mass.reshape((41,41))[11:,:]
    side_columns_mass_mesh=side_columns_mass.reshape((41,41))[11:,:]
    floatingse_platform_mass_vec_mesh=floatingse_platform_mass_vec.reshape((41,41))[11:,:]

    #--------------------------------- Max Pltfrm Pitch-----------------------------------------
    """
    # Normalize the data
    norm = Normalize(vmin=Max_PtfmPitch_vc_mesh.min(), vmax=Max_PtfmPitch_vc_mesh.max())
    rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(Max_PtfmPitch_vc_mesh, alpha=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rgba_colors, aspect='auto',origin= 'lower')
    # Set x ticks to side_clmns_mesh[0,:]
    x_indices = np.arange(0, Max_PtfmPitch_vc_mesh.shape[1], 2)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{main_clmns_mesh[0, i]:.1f}' for i in x_indices])
    # Set y ticks to side_clmns_mesh[:,0]
    y_indices = np.arange(0, Max_PtfmPitch_vc_mesh.shape[0], 2)
    y_indices = np.append(y_indices, y_indices[-1]+1)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f'{side_clmns_mesh[i, 0]:.1f}' for i in y_indices])
    ax.set_ylabel('$D_\mathrm{Side}$', fontsize=15)
    ax.set_xlabel('$D_\mathrm{Main}$', fontsize=15)
    ax.set_title('$\mathrm{Max\,\,Ptfm\,\,Pitch\,\,[deg]}$', fontsize=15)
    # Customize the colorbar ticks
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
    ax.plot(np.where(main_clmns_mesh[0,:]==10)[0][0],np.where(side_clmns_mesh[:,0]==12.5)[0][0] , 's', markersize=14, markerfacecolor='none', markeredgecolor='black', markeredgewidth =3)
    fig.savefig(os.getcwd()+'/outputs/' +
                    'Max_PtfmPitch_vc_mesh.pdf', format='pdf', dpi=300, bbox_inches='tight' )  


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #Plot the surface.
    surf = ax.plot_surface(main_clmns_mesh, side_clmns_mesh, Max_PtfmPitch_vc_mesh, cmap='Reds',
                       linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.set_ylabel('$D_\mathrm{Side}$', fontsize=15)
    ax.set_xlabel('$D_\mathrm{Main}$', fontsize=15)
    ax.set_zlabel(r'$\theta_p \,\, \mathrm{[deg]}$', fontsize=15)
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.1f}')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev=18, azim=-166)
    ax.dist = 15  # Set the distance from the viewer to the axes
    plt.show()
    fig.savefig(os.getcwd()+'/outputs/' +
                    'Max_PtfmPitch_vc_mesh_3D.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.35)  
    """
    #--------------------------------- STD Pltfrm Pitch-----------------------------------------
    """
    # Normalize the data
    norm = Normalize(vmin=Std_PtfmPitch_vec_mesh.min(), vmax=Std_PtfmPitch_vec_mesh.max())
    rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(Std_PtfmPitch_vec_mesh, alpha=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rgba_colors, aspect='auto',origin= 'lower')
    # Set x ticks to side_clmns_mesh[0,:]
    x_indices = np.arange(0, Std_PtfmPitch_vec_mesh.shape[1], 2)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{main_clmns_mesh[0, i]:.1f}' for i in x_indices])
    # Set y ticks to side_clmns_mesh[:,0]
    y_indices = np.arange(0, Std_PtfmPitch_vec_mesh.shape[0], 2)
    y_indices = np.append(y_indices, y_indices[-1]+1)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f'{side_clmns_mesh[i, 0]:.1f}' for i in y_indices])
    ax.set_ylabel('$D_\mathrm{Side}$', fontsize=15)
    ax.set_xlabel('$D_\mathrm{Main}$', fontsize=15)
    ax.set_title('$\mathrm{Std\,\,Ptfm\,\,Pitch\,\,[deg]}$', fontsize=15)
    # Customize the colorbar ticks
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
    ax.plot(np.where(main_clmns_mesh[0,:]==10)[0][0],np.where(side_clmns_mesh[:,0]==12.5)[0][0] , 's', markersize=14, markerfacecolor='none', markeredgecolor='black', markeredgewidth =3)
    fig.savefig(os.getcwd()+'/outputs/' +
                    'Std_PtfmPitch_vec_mesh.pdf', format='pdf', dpi=300, bbox_inches='tight' )  


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #Plot the surface.
    surf = ax.plot_surface(main_clmns_mesh, side_clmns_mesh, Std_PtfmPitch_vec_mesh, cmap='Reds',
                       linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.set_ylabel('$D_\mathrm{Side}$', fontsize=15)
    ax.set_xlabel('$D_\mathrm{Main}$', fontsize=15)
    ax.set_zlabel(r' Std - $\theta_p \,\, \mathrm{[deg]}$', fontsize=15)
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.1f}')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev=18, azim=-166)
    ax.dist = 15  # Set the distance from the viewer to the axes
    plt.show()
    fig.savefig(os.getcwd()+'/outputs/' +
                    'Std_PtfmPitch_vec_mesh_3D.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.35)  
    """
    #--------------------------------- floatingse_structurall_mass -----------------------------------------
    """
    # Normalize the data
    norm = Normalize(vmin=floatingse_structurall_mass_mesh.min(), vmax=floatingse_structurall_mass_mesh.max())
    rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(floatingse_structurall_mass_mesh, alpha=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rgba_colors, aspect='auto',origin= 'lower')
    # Set x ticks to side_clmns_mesh[0,:]
    x_indices = np.arange(0, floatingse_structurall_mass_mesh.shape[1], 2)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{main_clmns_mesh[0, i]:.1f}' for i in x_indices])
    # Set y ticks to side_clmns_mesh[:,0]
    y_indices = np.arange(0, floatingse_structurall_mass_mesh.shape[0], 2)
    y_indices = np.append(y_indices, y_indices[-1]+1)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f'{side_clmns_mesh[i, 0]:.1f}' for i in y_indices])
    ax.set_ylabel('$D_\mathrm{Side}$', fontsize=15)
    ax.set_xlabel('$D_\mathrm{Main}$', fontsize=15)
    ax.set_title('$m_p$', fontsize=15)
    # Customize the colorbar ticks
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
    ax.plot(np.where(main_clmns_mesh[0,:]==10)[0][0],np.where(side_clmns_mesh[:,0]==12.5)[0][0] , 's', markersize=14, markerfacecolor='none', markeredgecolor='black', markeredgewidth =3)
    fig.savefig(os.getcwd()+'/outputs/' +
                    'floatingse_structurall_mass_mesh.pdf', format='pdf', dpi=300, bbox_inches='tight' )  


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #Plot the surface.
    surf = ax.plot_surface(main_clmns_mesh, side_clmns_mesh, floatingse_structurall_mass_mesh, cmap='Reds',
                       linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.set_ylabel('$D_\mathrm{Side}$', fontsize=15)
    ax.set_xlabel('$D_\mathrm{Main}$', fontsize=15)
    ax.set_zlabel(r' Sm_p$', fontsize=15)
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.1f}')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev=18, azim=-166)
    ax.dist = 15  # Set the distance from the viewer to the axes
    plt.show()
    fig.savefig(os.getcwd()+'/outputs/' +
                    'floatingse_structurall_mass_mesh_3D.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.35)  
    """
    #--------------------------------- AEP_vec_mesh -----------------------------------------

    # Normalize the data
    norm = Normalize(vmin=AEP_vec_mesh.min(), vmax=AEP_vec_mesh.max())
    rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(AEP_vec_mesh, alpha=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rgba_colors, aspect='auto',origin= 'lower')
    # Set x ticks to side_clmns_mesh[0,:]
    x_indices = np.arange(0, AEP_vec_mesh.shape[1], 2)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{main_clmns_mesh[0, i]:.1f}' for i in x_indices])
    # Set y ticks to side_clmns_mesh[:,0]
    y_indices = np.arange(0, AEP_vec_mesh.shape[0], 2)
    y_indices = np.append(y_indices, y_indices[-1]+1)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f'{side_clmns_mesh[i, 0]:.1f}' for i in y_indices])
    ax.set_ylabel('$D_\mathrm{Side}$', fontsize=15)
    ax.set_xlabel('$D_\mathrm{Main}$', fontsize=15)
    ax.set_title('$AEP$', fontsize=15)
    # Customize the colorbar ticks
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
    ax.plot(np.where(main_clmns_mesh[0,:]==10)[0][0],np.where(side_clmns_mesh[:,0]==12.5)[0][0] , 's', markersize=14, markerfacecolor='none', markeredgecolor='black', markeredgewidth =3)
    fig.savefig(os.getcwd()+'/outputs/' +
                    'AEP_vec_mesh.pdf', format='pdf', dpi=300, bbox_inches='tight' )  


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #Plot the surface.
    surf = ax.plot_surface(main_clmns_mesh, side_clmns_mesh, AEP_vec_mesh, cmap='Reds',
                       linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.set_ylabel('$D_\mathrm{Side}$', fontsize=15)
    ax.set_xlabel('$D_\mathrm{Main}$', fontsize=15)
    ax.set_zlabel('AEP ', fontsize=15)
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.1f}')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev=18, azim=-166)
    ax.dist = 15  # Set the distance from the viewer to the axes
    plt.show()
    fig.savefig(os.getcwd()+'/outputs/' +
                    'AEP_vec_mesh_3D.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.35)  