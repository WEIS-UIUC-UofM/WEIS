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
    output_dir = os.path.join(run_dir, "outputs/15_RAFT_optimization_freq_step1_Optimization_clmnSpacing_sideColmnsdiam_lowPntoon_diam") #opt coByla
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
    #main_column_outer_diameter_vec=np.squeeze(outdata[0]['floating.memgrp0.outer_diameter_in'])
    Clmn_spacing_vec=np.squeeze(outdata[0]['floating.jointdv_0'])
    side_columns_outer_diameter_vec=np.squeeze(outdata[0]['floating.memgrp1.outer_diameter_in'])
    Y_pntoon_lower_vec=np.squeeze(outdata[0]['floating.memgrp5.outer_diameter_in'])
    """ Y_pntoon_upper_vec=np.squeeze(outdata[0]['floating.memgrp4.outer_diameter_in'])
    Y_pntoon_lower_vec=np.squeeze(outdata[0]['floating.memgrp5.outer_diameter_in'])
    Clmns_keel_vec=np.squeeze(outdata[0]['floating.jointdv_0'])
    Clmns_freeboard_vec=np.squeeze(outdata[0]['floating.jointdv_1'])
    Clmns_spacing_vec=np.squeeze(outdata[0]['floating.jointdv_2'])
    Clmns_Length_vec=Clmns_freeboard_vec-Clmns_keel_vec
    tower_section_height_vec=np.squeeze(outdata[0]['towerse.tower_section_height'])
    tower_length=np.squeeze(outdata[0]['tower_grid.height']) # == np.sum(tower_section_height_vec,axis=1) default turbine: 15 t0 125 --> 1e30 m
    peak_shaving_vec= np.squeeze(outdata[0]['tune_rosco_ivc.ps_percent'])
    joints_xyz_matrix=np.squeeze(outdata[0]['floating.joints_xyz'])
    platfom_columns_height_vec=np.array([np.squeeze(outdata[0]['floating.joints_xyz'])[i][1][2]-np.squeeze(outdata[0]['floating.joints_xyz'])[i][0][2] for i in np.arange(len(main_column_outer_diameter_vec)) ])
    radial_platfom_columns_spacing_vec=np.array([np.sqrt(np.squeeze(outdata[0]['floating.joints_xyz'])[i][2][0]**2+np.squeeze(outdata[0]['floating.joints_xyz'])[i][2][1]**2) for i in np.arange(len(main_column_outer_diameter_vec))])
    rotor_diameter_vec=np.squeeze(outdata[0]['configuration.rotor_diameter_user'])
    tower_diameter_matriix=np.squeeze(outdata[0]['tower.diameter'])"""
    #--------------------------------------------------------------------------------------

    #-------------------------------------Constraints------------------------------------
    Max_PtfmPitch_vc=np.squeeze(outdata[0]['raft.Max_PtfmPitch'])
    Max_PtfmPitch_vc[Max_PtfmPitch_vc > 10] = 10
    Std_PtfmPitch_vec=np.squeeze(outdata[0]['raft.Std_PtfmPitch'])
    Std_PtfmPitch_vec[Std_PtfmPitch_vec > 2] = 2.0
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

    Clmn_spacing_vec_mesh=Clmn_spacing_vec.reshape((8,8,8)).transpose(2, 0, 1) 
    side_columns_outer_diameter_vec_mesh=side_columns_outer_diameter_vec.reshape((8,8,8)).transpose(2, 0, 1) 
    Y_pntoon_lower_vec_mesh=Y_pntoon_lower_vec.reshape((8,8,8)).transpose(2, 0, 1) 
    Max_PtfmPitch_vc_mesh=Max_PtfmPitch_vc.reshape((8,8,8)).transpose(2, 0, 1) 
    Std_PtfmPitch_vec_mesh=Std_PtfmPitch_vec.reshape((8,8,8)).transpose(2, 0, 1) 
    pltfrm_mass_vec_mesh=pltfrm_mass_vec.reshape((8,8,8)).transpose(2, 0, 1) 
    AEP_vec_mesh=AEP_vec.reshape((8,8,8)).transpose(2, 0, 1) 
    cost_vec_mesh=cost_vec.reshape((8,8,8)).transpose(2, 0, 1) 
    finance_lcoe_vec_mesh=finance_lcoe_vec.reshape((8,8,8)).transpose(2, 0, 1) 
    floatingse_structurall_mass_mesh=floatingse_structurall_mass.reshape((8,8,8)).transpose(2, 0, 1) *1e-6
    main_column_mass_mesh=main_column_mass.reshape((8,8,8)).transpose(2, 0, 1) 
    side_columns_mass_mesh=side_columns_mass.reshape((8,8,8)).transpose(2, 0, 1) 
    floatingse_platform_mass_vec_mesh=floatingse_platform_mass_vec.reshape((8,8,8)).transpose(2, 0, 1) *1e-6

    #--------------------------------- Max Pltfrm Pitch-----------------------------------------

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.scatter(side_columns_outer_diameter_vec_mesh.flatten(), Y_pntoon_lower_vec_mesh.flatten(), Clmn_spacing_vec_mesh.flatten(), c=Max_PtfmPitch_vc_mesh.flatten(), cmap='Reds')
    fig.colorbar(surf, ax=ax, label='Max_PtfmPitch_vc_mesh')
    ax.set_xlabel('$D_\mathrm{clm- out}$', fontsize=15)
    ax.set_ylabel('$D_\mathrm{pntn- out}$', fontsize=15)
    ax.set_zlabel('$S_\mathrm{spacing}$', fontsize=15)
    #ax.set_zlabel('Clmns_spacing_vec_mesh')
    #plt.show()

    # Normalize the data
    norm = Normalize(vmin=Max_PtfmPitch_vc_mesh.min(), vmax=Max_PtfmPitch_vc_mesh.max())
    for ii in np.arange(8):
        rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(Max_PtfmPitch_vc_mesh[ii], alpha=1)
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(rgba_colors, aspect='auto', origin='lower')
        # Set x ticks to Y_pntoon_lower_vec_mesh[0,:]
        x_indices = np.arange(0, Max_PtfmPitch_vc_mesh[ii].shape[1], 1)
        ax.set_xticks(x_indices)
        ax.set_xticklabels([f'{side_columns_outer_diameter_vec_mesh[ii, 0, i]:.2f}' for i in x_indices])
        # Set y ticks to Y_pntoon_lower_vec_mesh[:,0]
        y_indices = np.arange(0, Max_PtfmPitch_vc_mesh[ii].shape[0], 1)
        y_indices = np.append(y_indices, y_indices[-1])
        ax.set_yticks(y_indices)
        ax.set_yticklabels([f'{Y_pntoon_lower_vec_mesh[ii, i, 0]:.1f}' for i in y_indices])
        ax.set_ylabel('$D_\mathrm{pntn- out}$', fontsize=15)
        ax.set_xlabel('$D_\mathrm{clm- out}$', fontsize=15)
        ax.set_title(r'$\mathrm{Max\,\,Ptfm\,\,Pitch\,\,[deg]}$' + f' - Clmn Spc : {Clmn_spacing_vec_mesh[ii][0][0]}', fontsize=15)
        # Customize the colorbar ticks
        cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
        # Annotate each element with its value
        for i in range(Max_PtfmPitch_vc_mesh[ii].shape[0]):
            for j in range(Max_PtfmPitch_vc_mesh[ii].shape[1]):
                ax.text(j, i, f'{Max_PtfmPitch_vc_mesh[ii][i, j]:.1f}', ha='center', va='center', color='black', fontsize=12)
        if Clmn_spacing_vec_mesh[ii][0][0] == 51.57:  # baseline
            ax.plot(np.where(side_columns_outer_diameter_vec_mesh[ii, 0, :] == 12.5)[0][0], np.where(Y_pntoon_lower_vec_mesh[ii, :, 0] == 9.6148)[0][0], 's', markersize=25, markerfacecolor='none', markeredgecolor='black', markeredgewidth=3)
        fig.savefig(os.getcwd() + '/outputs/' + f'Max_PtfmPitch_vc_mesh_outClmnsdiam_lowPntoon_clmSpacing{ii}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        #plt.show()  # Show the plot
    
    #--------------------------------- STD Pltfrm Pitch-----------------------------------------
    # Normalize the data
    norm = Normalize(vmin=Std_PtfmPitch_vec_mesh.min(), vmax=Std_PtfmPitch_vec_mesh.max())
    for ii in np.arange(8):
        rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(Std_PtfmPitch_vec_mesh[ii], alpha=1)
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(rgba_colors, aspect='auto', origin='lower')
        # Set x ticks to Y_pntoon_lower_vec_mesh[0,:]
        x_indices = np.arange(0, Std_PtfmPitch_vec_mesh[ii].shape[1], 1)
        ax.set_xticks(x_indices)
        ax.set_xticklabels([f'{side_columns_outer_diameter_vec_mesh[ii, 0, i]:.2f}' for i in x_indices])
        # Set y ticks to Y_pntoon_lower_vec_mesh[:,0]
        y_indices = np.arange(0, Std_PtfmPitch_vec_mesh[ii].shape[0], 1)
        y_indices = np.append(y_indices, y_indices[-1])
        ax.set_yticks(y_indices)
        ax.set_yticklabels([f'{Y_pntoon_lower_vec_mesh[ii, i, 0]:.1f}' for i in y_indices])
        ax.set_ylabel('$D_\mathrm{pntn- out}$', fontsize=15)
        ax.set_xlabel('$D_\mathrm{clm- out}$', fontsize=15)
        ax.set_title(r'$\mathrm{Std\,\,Ptfm\,\,Pitch\,\,[deg]}$' + f' - Clmn Spc : {Clmn_spacing_vec_mesh[ii][0][0]}', fontsize=15)
        # Customize the colorbar ticks
        cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
        # Annotate each element with its value
        for i in range(Std_PtfmPitch_vec_mesh[ii].shape[0]):
            for j in range(Std_PtfmPitch_vec_mesh[ii].shape[1]):
                ax.text(j, i, f'{Std_PtfmPitch_vec_mesh[ii][i, j]:.1f}', ha='center', va='center', color='black', fontsize=12)
        if Clmn_spacing_vec_mesh[ii][0][0] == 51.57:  # baseline
            ax.plot(np.where(side_columns_outer_diameter_vec_mesh[ii, 0, :] == 12.5)[0][0], np.where(Y_pntoon_lower_vec_mesh[ii, :, 0] == 9.6148)[0][0], 's', markersize=25, markerfacecolor='none', markeredgecolor='black', markeredgewidth=3)
        fig.savefig(os.getcwd() + '/outputs/' + f'Std_PtfmPitch_vec_mesh_outClmnsdiam_lowPntoon_clmSpacing{ii}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        #plt.show()  # Show the plot
    #--------------------------------- floatingse_structurall_mass -----------------------------------------
    # Normalize the data
    norm = Normalize(vmin=floatingse_structurall_mass_mesh.min(), vmax=floatingse_structurall_mass_mesh.max())
    for ii in np.arange(8):
        rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(floatingse_structurall_mass_mesh[ii], alpha=1)
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(rgba_colors, aspect='auto', origin='lower')
        # Set x ticks to Y_pntoon_lower_vec_mesh[0,:]
        x_indices = np.arange(0, floatingse_structurall_mass_mesh[ii].shape[1], 1)
        ax.set_xticks(x_indices)
        ax.set_xticklabels([f'{side_columns_outer_diameter_vec_mesh[ii, 0, i]:.2f}' for i in x_indices])
        # Set y ticks to Y_pntoon_lower_vec_mesh[:,0]
        y_indices = np.arange(0, floatingse_structurall_mass_mesh[ii].shape[0], 1)
        y_indices = np.append(y_indices, y_indices[-1])
        ax.set_yticks(y_indices)
        ax.set_yticklabels([f'{Y_pntoon_lower_vec_mesh[ii, i, 0]:.1f}' for i in y_indices])
        ax.set_ylabel('$D_\mathrm{pntn- out}$', fontsize=15)
        ax.set_xlabel('$D_\mathrm{clm- out}$', fontsize=15)
        ax.set_title(r'$m_p \,\, \mathrm{floatingse - structurall}$' + f' - Clmn Spc : {Clmn_spacing_vec_mesh[ii][0][0]}', fontsize=15)
        # Customize the colorbar ticks
        cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
        # Annotate each element with its value
        for i in range(floatingse_structurall_mass_mesh[ii].shape[0]):
            for j in range(floatingse_structurall_mass_mesh[ii].shape[1]):
                ax.text(j, i, f'{floatingse_structurall_mass_mesh[ii][i, j]/1000:.0f}', ha='center', va='center', color='black', fontsize=12)
        if Clmn_spacing_vec_mesh[ii][0][0] == 51.57:  # baseline
            ax.plot(np.where(side_columns_outer_diameter_vec_mesh[ii, 0, :] == 12.5)[0][0], np.where(Y_pntoon_lower_vec_mesh[ii, :, 0] == 9.6148)[0][0], 's', markersize=25, markerfacecolor='none', markeredgecolor='black', markeredgewidth=3)
        fig.savefig(os.getcwd() + '/outputs/' + f'floatingse_structurall_mass_mesh_outClmnsdiam_lowPntoon_clmSpacing{ii}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        #plt.show()  # Show the plot

    #--------------------------------- floatingse_platform_mass -----------------------------------------
    norm = Normalize(vmin=floatingse_platform_mass_vec_mesh.min(), vmax=floatingse_platform_mass_vec_mesh.max())
    for ii in np.arange(8):
        rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(floatingse_platform_mass_vec_mesh[ii], alpha=1)
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(rgba_colors, aspect='auto', origin='lower')
        # Set x ticks to Y_pntoon_lower_vec_mesh[0,:]
        x_indices = np.arange(0, floatingse_platform_mass_vec_mesh[ii].shape[1], 1)
        ax.set_xticks(x_indices)
        ax.set_xticklabels([f'{side_columns_outer_diameter_vec_mesh[ii, 0, i]:.2f}' for i in x_indices])
        # Set y ticks to Y_pntoon_lower_vec_mesh[:,0]
        y_indices = np.arange(0, floatingse_platform_mass_vec_mesh[ii].shape[0], 1)
        y_indices = np.append(y_indices, y_indices[-1])
        ax.set_yticks(y_indices)
        ax.set_yticklabels([f'{Y_pntoon_lower_vec_mesh[ii, i, 0]:.1f}' for i in y_indices])
        ax.set_ylabel('$D_\mathrm{pntn- out}$', fontsize=15)
        ax.set_xlabel('$D_\mathrm{clm- out}$', fontsize=15)
        ax.set_title(r'$m_p \,\, \mathrm{floatingse - platfrom}$' + f' - Clmn Spc : {Clmn_spacing_vec_mesh[ii][0][0]}', fontsize=15)
        # Customize the colorbar ticks
        cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
        # Annotate each element with its value
        for i in range(floatingse_platform_mass_vec_mesh[ii].shape[0]):
            for j in range(floatingse_platform_mass_vec_mesh[ii].shape[1]):
                ax.text(j, i, f'{floatingse_platform_mass_vec_mesh[ii][i, j]/1000:.0f}', ha='center', va='center', color='black', fontsize=12)
        if Clmn_spacing_vec_mesh[ii][0][0] == 51.57:  # baseline
            ax.plot(np.where(side_columns_outer_diameter_vec_mesh[ii, 0, :] == 12.5)[0][0], np.where(Y_pntoon_lower_vec_mesh[ii, :, 0] == 9.6148)[0][0], 's', markersize=25, markerfacecolor='none', markeredgecolor='black', markeredgewidth=3)
        fig.savefig(os.getcwd() + '/outputs/' + f'floatingse_platform_mass_vec_mesh__outClmnsdiam_lowPntoon_clmSpacing{ii}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        #plt.show()  # Show the plot
    #--------------------------------- AEP_vec_mesh -----------------------------------------
    norm = Normalize(vmin=AEP_vec_mesh.min(), vmax=AEP_vec_mesh.max())
    for ii in np.arange(8):
        rgba_colors = ScalarMappable(norm=norm, cmap='Reds').to_rgba(AEP_vec_mesh[ii], alpha=1)
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(rgba_colors, aspect='auto', origin='lower')
        # Set x ticks to Y_pntoon_lower_vec_mesh[0,:]
        x_indices = np.arange(0, AEP_vec_mesh[ii].shape[1], 1)
        ax.set_xticks(x_indices)
        ax.set_xticklabels([f'{side_columns_outer_diameter_vec_mesh[ii, 0, i]:.2f}' for i in x_indices])
        # Set y ticks to Y_pntoon_lower_vec_mesh[:,0]
        y_indices = np.arange(0, AEP_vec_mesh[ii].shape[0], 1)
        y_indices = np.append(y_indices, y_indices[-1])
        ax.set_yticks(y_indices)
        ax.set_yticklabels([f'{Y_pntoon_lower_vec_mesh[ii, i, 0]:.1f}' for i in y_indices])
        ax.set_ylabel('$D_\mathrm{pntn- out}$', fontsize=15)
        ax.set_xlabel('$D_\mathrm{clm- out}$', fontsize=15)
        ax.set_title('$AEP$', fontsize=15)
        ax.set_title(r'$\mathrm{AEP}$' + f' - Clmn Spc : {Clmn_spacing_vec_mesh[ii][0][0]}', fontsize=15)
        # Customize the colorbar ticks
        cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='Reds'), ax=ax)
        # Annotate each element with its value
        for i in range(AEP_vec_mesh[ii].shape[0]):
            for j in range(AEP_vec_mesh[ii].shape[1]):
                ax.text(j, i, f'{AEP_vec_mesh[ii][i, j]*1e-6:.1f}', ha='center', va='center', color='black', fontsize=12)
        if Clmn_spacing_vec_mesh[ii][0][0] == 51.57:  # baseline
            ax.plot(np.where(side_columns_outer_diameter_vec_mesh[ii, 0, :] == 12.5)[0][0], np.where(Y_pntoon_lower_vec_mesh[ii, :, 0] == 9.6148)[0][0], 's', markersize=25, markerfacecolor='none', markeredgecolor='black', markeredgewidth=3)
        fig.savefig(os.getcwd() + '/outputs/' + f'AEP_vec_mesh_outClmnsdiam_lowPntoon_clmSpacing{ii}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        #plt.show()  # Show the plot