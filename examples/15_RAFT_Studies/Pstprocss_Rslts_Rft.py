"""
Simple script to show how to grab all cases from a DOE run. User can then
postprocess or plot further.
"""

import glob
import os
import sys
import time

import numpy as np
import pandas as pd 
import multiprocessing as mp 

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
    post_multi = True

    # sql outfile directory
    run_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))# os.path.dirname(os.path.realpath(__file__))   
    output_dir = os.path.join(run_dir, "outputs/15_RAFT_optimization_freq_step1")
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
    rated_power=np.squeeze(outdata[0]['configuration.rated_power']) 
    #--------------------------------------------------------------------------------------

    #-------------------------------------Constraints------------------------------------
    Max_PtfmPitch_vc=np.squeeze(outdata[0]['raft.Max_PtfmPitch'])
    Std_PtfmPitch_vec=np.squeeze(outdata[0]['raft.Std_PtfmPitch'])
    height_constraint_vec=np.squeeze(outdata[0]['towerse.height_constraint'])
    tower_slope_vec=np.squeeze(outdata[0]['towerse.slope'])
    nacelle_acceleration_vec=np.squeeze(outdata[0]['raft.max_nac_accel']) # This is true 
    #--------------------------------------------------------------------------------------

    #-------------------------------------Objectives------------------------------------
    pltfrm_mass_vec=np.squeeze(outdata[0]['floatingse.platform_mass']) # This is true []     ##np.squeeze(outdata[0]['raft.platform_mass'])
    AEP_vec=np.squeeze(outdata[0]['rotorse.rp.AEP'])*1e-6 #[GWh]  This is true
    cost_vec=np.squeeze(outdata[0]['tcc.turbine_cost'])
    turbine_cost_kW_vec=np.squeeze(outdata[0]['tcc.turbine_cost_kW'])
    turbine_mass_tcc_vec=np.squeeze(outdata[0]['tcc.turbine_mass_tcc'])
    finance_lcoe_vec=np.squeeze(outdata[0]['financese.lcoe'])*1000 # correct LCOE [USD/MWH] 
    Cost=finance_lcoe_vec*AEP_vec*1000  # correct Cost [USD] 
    Cp_matrix=np.squeeze(outdata[0]['rotorse.rp.powercurve.Cp'])
    blade_mass_vec=np.squeeze(outdata[0]['rotorse.blade_mass'])
    tower_mass_vec=np.squeeze(outdata[0]['towerse.tower_mass'])
    tower_cost_vec=np.squeeze(outdata[0]['tcc.tower_cost'])
    floatingse_structurall_mass=np.squeeze(outdata[0]['floatingse.system_structural_mass'])
    main_column_mass=np.squeeze(outdata[0]['floatingse.member0.structural_cost'])
    side_columns_mass=np.squeeze(outdata[0]['floatingse.member1.structural_cost'])
    tower_top_deflection_vec=np.squeeze(outdata[0]['towerse.tower.top_deflection'])
    tower_deflection_matrix=np.squeeze(outdata[0]['towerse.tower.tower_deflection'])
    surge_freq=1/np.squeeze(outdata[0]['raft.surge_period']) # [HZ] # This is true 
    sway_freq=1/np.squeeze(outdata[0]['raft.sway_period']) # This is true 
    heave_freq=1/np.squeeze(outdata[0]['raft.heave_period'])# This is true 
    roll_freq=1/np.squeeze(outdata[0]['raft.roll_period'])# This is true 
    yaw_freq=1/np.squeeze(outdata[0]['raft.yaw_period'])# This is true 
    pitch_freq=1/np.squeeze(outdata[0]['raft.pitch_period'])# This is true 
    raft_freq_Hz=np.array([surge_freq, sway_freq, heave_freq, roll_freq, pitch_freq, yaw_freq])  # This is true 
    #'wisdem.wt.floatingse.load.tower.structural_frequencies':
    tower_structural_frequencies=np.squeeze(outdata[0]['towerse.tower.structural_frequencies']) # first frequency of all modes
    tower_foreft_freq=np.squeeze(outdata[0]['towerse.tower.fore_aft_freqs']) # This is true     # all frequencies of first mode (surge)
    tower_side2side_freq=np.squeeze(outdata[0]['towerse.tower.side_side_freqs']) # This is true  # all frequencies of seocnd mode (sway)
    #--------------------------------------------------------------------------------------

    """for key in outdata[0]:
        if 'omega' in key:
            print(key)
    design_space_vec_1=np.squeeze(outdata[0]['tune_rosco_ivc.omega_pc'])
            
    design_space_vec_2=np.squeeze(outdata[0]['configuration.rotor_diameter_user'])
    # Important
    column_spacing_data= np.squeeze(outdata[0]['floating.joints_xyz']) # related to column spacing
    main_column_diameter=np.squeeze(outdata[0]['floating.memgrp0.outer_diameter_in'])
    first_side_coumn_diameter=np.squeeze(outdata[0]['floating.memgrp1.outer_diameter_in'])
    second_side_coumn_diameter=np.squeeze(outdata[0]['floating.memgrp2.outer_diameter_in'])
    third_side_coumn_diameter=np.squeeze(outdata[0]['floating.memgrp3.outer_diameter_in'])
    peak_shaving= np.squeeze(outdata[0]['tune_rosco_ivc.ps_percent']) #
    #wisdem.wt.wt_init.configuration.rotor_diameter_user
    pltfrm_mass_space_vec=np.squeeze(outdata[0]['raft.platform_mass'])
    AEP_space_vec=np.squeeze(outdata[0]['rotorse.rp.AEP'])
    cost_vec=np.squeeze(outdata[0]['tcc.turbine_cost'])
    turbine_cost_kW_vec=np.squeeze(outdata[0]['tcc.turbine_cost_kW'])
    turbine_mass_tcc_vec=np.squeeze(outdata[0]['tcc.turbine_mass_tcc'])
    max_pitch_vec=np.squeeze(outdata[0]['tune_rosco_ivc.max_pitch'])
    max_pitch_rate_vec=np.squeeze(outdata[0]['control.max_pitch_rate'])
    max_nacelle_acceleration_vec=np.squeeze(outdata[0]['raft.max_nac_accel'])
    nacelle_acceleration_response_vec=np.squeeze(outdata[0]['raft.response_nacelle acceleration'])
    hub_height_vec=np.squeeze(outdata[0]['configuration.hub_height_user']) 
    tower_diameter_vec=np.squeeze(outdata[0]['tower.diameter']) """
    
    # write to file
    #outdata_fname = 'doe_outdata'
    #outdata_fpath = os.path.join(output_dir,outdata_fname) 
    #df.to_csv(outdata_fpath + '.csv', index=False)
    #print('Saved {}'.format(outdata_fpath + '.csv'))
    # FileTools.save_yaml(output_dir, outdata_fname + '.yaml', collected_data, package=2)  


    ss = pd.read_pickle("outputs/15_RAFT_optimization_freq_step1/refturb_output.pkl")
    

    comp_files = [
    'outputs/15_RAFT_optimization_freq_step1/refturb_output.csv'
    ]
    # only works on single values for now
    comp_values = [
    'floatingse.platform_mass',
    'financese.bos_per_kW',
    ]
    comp = {}
    for value in comp_values:
        comp[value] = []
    turbs = []
    for file in comp_files:
        turb = pd.read_csv(file)
        turbs.append(turb)
        for value in comp_values:
            i_val = turb.variables == value
            a = turb[i_val]['values'].to_numpy()
            val = float(a[0][1:-1])
            comp[value].append(val)

    iter_0 = pd.read_pickle("outputs/15_RAFT_Studies_Saeid/raft_designs/raft_design_0.pkl")
    iter_1 = pd.read_pickle("outputs/15_RAFT_Studies_Saeid/raft_designs/raft_design_1.pkl")
    iter_2 = pd.read_pickle("outputs/15_RAFT_Studies_Saeid/raft_designs/raft_design_2.pkl")

    k=1
    

