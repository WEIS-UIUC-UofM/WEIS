#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:40:11 2024

@author: elenaf3
"""
import numpy as np
import os
import time
import re
import pickle as pkl
import openmdao.api as om
# from wisdem.commonse.mpi_tools import MPI
from smt.surrogate_models import KRG
from weis.glue_code.gc_LoadInputs     import WindTurbineOntologyPythonWEIS
from runSBO import WindTurbineSMOpt
from weis.inputs.validation import load_yaml
import yaml
from weis.glue_code.runWEIS     import run_weis
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

####################################################################

run_dir                = os.path.dirname( os.path.realpath(__file__) ) + os.sep
ex_dir = "/home/elenaf3/Desktop/DC_WEIS_Feb/WEIS/examples/15_RAFT_Studies/"         #change this to your example folder path
fname_wt_input         = os.path.join(ex_dir,"..","06_IEA-15-240-RWT", "IEA-15-240-RWT_VolturnUS-S.yaml")
fname_modeling_options = ex_dir + "modeling_options_level1_doe.yaml"
# fname_analysis_options = ex_dir + "analysis_options_level1_doe_verification.yaml"
fname_analysis_options = ex_dir + "analysis_options_level1.yaml"
overridden_values = {}

wt_initial = WindTurbineOntologyPythonWEIS(fname_wt_input, fname_modeling_options, fname_analysis_options)
wt_init, modeling_options, opt_options = wt_initial.get_input_data()

# make the folder_output relative to the input, if it's a relative path
analysis_input_dir = os.path.dirname(opt_options['fname_input_analysis'])
opt_options['general']['folder_output'] = os.path.join(analysis_input_dir,opt_options['general']['folder_output'])

folder_output = opt_options['general']['folder_output']
opt_filename ='opt_output' + '.pkl'

####################################################################

sm_file = os.path.join(folder_output, os.path.splitext(opt_options['recorder']['file_name'])[0] + '.smt')
fname_ccdc_options = ex_dir + "ccdc_options.yaml"
ccdc_options = load_yaml(fname_ccdc_options) 
WTSMO = WindTurbineSMOpt(ccdc_options)
WTSMO.read_sm(sm_file, modeling_options)
sm2 = WTSMO.sm2
model = sm2[0] 

# get R^2 value from the SM (not necessary)      
R_sq_all=np.zeros((len(sm2),))
for ind_s in range(len(sm2)):
    R_sq_all[ind_s] = sm2[ind_s]['surrogate'].R_squared 
R_sq_min = min(R_sq_all)

# identify inputs requested in analysis yaml
def get_design_variables():
        dvs_analysis_keys = []
        
        # Set optimization design variables.
        rotorD_opt = opt_options["design_variables"]["rotor_diameter"]
        blade_opt = opt_options["design_variables"]["blade"]
        tower_opt = opt_options["design_variables"]["tower"]
        monopile_opt = opt_options["design_variables"]["monopile"]
        jacket_opt = opt_options["design_variables"]["jacket"]
        control_opt = opt_options["design_variables"]["control"]
        hub_opt = opt_options["design_variables"]["hub"]
        drive_opt = opt_options["design_variables"]["drivetrain"]
        float_opt = opt_options["design_variables"]["floating"]
        mooring_opt = opt_options["design_variables"]["mooring"]
        
        # -- Rotor & Blade --
        if rotorD_opt["flag"]:
            dvs_analysis_keys.append("configuration.rotor_diameter_user")
        
        twist_options = blade_opt["aero_shape"]["twist"]
        if twist_options["flag"]:
            dvs_analysis_keys.append("blade.opt_var.twist_opt")
        
        chord_options = blade_opt["aero_shape"]["chord"]
        if chord_options["flag"]:
            dvs_analysis_keys.append("blade.opt_var.chord_opt")
        
        # if blade_opt["aero_shape"]["af_positions"]["flag"]:
        #     n_af = modeling_options["WISDEM"]["RotorSE"]["n_af_span"]
        #     indices_af = range(blade_opt["aero_shape"]["af_positions"]["af_start"], n_af - 1)
        #     af_pos_init = wt_init["components"]["blade"]["outer_shape_bem"]["airfoil_position"]["grid"]
        #     step_size = self._get_step_size()
        #     lb_af = np.zeros(n_af)
        #     ub_af = np.zeros(n_af)
        #     for i in range(1, indices_af[0]):
        #         lb_af[i] = ub_af[i] = af_pos_init[i]
        #     for i in indices_af:
        #         lb_af[i] = 0.5 * (af_pos_init[i - 1] + af_pos_init[i]) + step_size
        #         ub_af[i] = 0.5 * (af_pos_init[i + 1] + af_pos_init[i]) - step_size
        #     lb_af[-1] = ub_af[-1] = 1.0
        #     dvs_analysis_keys.append(
        #         "blade.opt_var.af_position", indices=indices_af, lower=lb_af[indices_af], upper=ub_af[indices_af]
        #     )
        
        L_D_options = blade_opt["aero_shape"]["L/D"]
        if L_D_options["flag"]:
            dvs_analysis_keys.append("inn_af.L_D_opt")
        
        c_d_options = blade_opt["aero_shape"]["c_d"]
        if c_d_options["flag"]:
            dvs_analysis_keys.append("inn_af.c_d_opt")
        
        stall_options = blade_opt["aero_shape"]["stall_margin"]
        if stall_options["flag"]:
            dvs_analysis_keys.append("inn_af.stall_margin_opt")
        
        t_c_options = blade_opt["aero_shape"]["rthick"]
        if t_c_options["flag"]:
            dvs_analysis_keys.append("inn_af.r_thick_opt")
        
        z_options = blade_opt["aero_shape"]["z"]
        if z_options["flag"]:
            dvs_analysis_keys.append("inn_af.z")
        
        if "structure" in blade_opt and len(blade_opt["structure"]) > 0:
            
            for i in range(len(blade_opt["structure"])):
                k = blade_opt["layer_index_opt"][i]
                dvs_analysis_keys.append("blade.opt_var.layer_%d_opt"%k)
        
        # -- Tower --
        if tower_opt["outer_diameter"]["flag"]:
            dvs_analysis_keys.append("tower.diameter")
        
        if tower_opt["layer_thickness"]["flag"]:
            dvs_analysis_keys.append("tower.layer_thickness")
        
        # if tower_opt["E"]["flag"]:
        #     ivc = wt_opt.model.add_subsystem("E_ivc", om.IndepVarComp(), promotes=[])
        #     ivc.add_output("E_user", val=10.0, units="Pa")
        #     dvs_analysis_keys.append(
        #         "E_ivc.E_user",
        #         lower=tower_opt["E"]["lower_bound"],
        #         upper=tower_opt["E"]["upper_bound"],
        #         ref=1e9,
        #     )
        #     wt_opt.model.connect("E_ivc.E_user", "towerse.E_user")
        
        #     if modeling_options["flags"]["monopile"] or modeling_options["flags"]["jacket"]:
        #         wt_opt.model.connect("E_ivc.E_user", "fixedse.E_user")
        
               
        if tower_opt["rho"]["flag"]:
            dvs_analysis_keys.append("materials.rho" )
        
        # -- Monopile --
        if monopile_opt["outer_diameter"]["flag"]:
            dvs_analysis_keys.append("monopile.diameter")
        
        if monopile_opt["layer_thickness"]["flag"]:
            dvs_analysis_keys.append("monopile.layer_thickness")
        
        # -- Jacket --
        if jacket_opt["foot_head_ratio"]["flag"]:
            dvs_analysis_keys.append("jacket.foot_head_ratio")
        
        if jacket_opt["r_head"]["flag"]:
            dvs_analysis_keys.append("jacket.r_head")
        
        if jacket_opt["leg_diameter"]["flag"]:
            dvs_analysis_keys.append("jacket.leg_diameter")
        
        if jacket_opt["leg_thickness"]["flag"]:
            dvs_analysis_keys.append("jacket.leg_thickness")
        
        if jacket_opt["brace_diameters"]["flag"]:
            dvs_analysis_keys.append("jacket.brace_diameters")
        
        if jacket_opt["brace_thicknesses"]["flag"]:
            dvs_analysis_keys.append("jacket.brace_thicknesses")
        
        if jacket_opt["height"]["flag"]:
            dvs_analysis_keys.append("jacket.height")
        
        if jacket_opt["bay_spacing"]["flag"]:
            dvs_analysis_keys.append("jacket.bay_spacing")
        
        # -- Hub & Drivetrain --
        if hub_opt["cone"]["flag"]:
            dvs_analysis_keys.append("hub.cone")
        if hub_opt["hub_diameter"]["flag"]:
            dvs_analysis_keys.append("hub.diameter")
        if drive_opt["uptilt"]["flag"]:
            dvs_analysis_keys.append("nacelle.uptilt")
        
        if drive_opt["generator_length"]["flag"]:
            dvs_analysis_keys.append("nacelle.L_generator")
        
        for k in [
            "overhang",
            "distance_tt_hub",
            "distance_hub_mb",
            "distance_mb_mb",
            "gear_ratio",
            "bedplate_flange_width",
            "lss_diameter",
            "hss_diameter",
            "nose_diameter",
        ]:
            if drive_opt[k]["flag"]:
                dvs_analysis_keys.append("nacelle." + k,)
        
        for k in [
            "bedplate_wall_thickness",
            "bedplate_web_thickness",
            "bedplate_flange_thickness",
            "lss_wall_thickness",
            "hss_wall_thickness",
            "nose_wall_thickness",
        ]:
            if drive_opt[k]["flag"]:
                dvs_analysis_keys.append("nacelle." + k)
        
        # -- Floating --
        if float_opt["joints"]["flag"]:
            jointz = float_opt["joints"]["z_coordinate"]
            jointr = float_opt["joints"]["r_coordinate"]
        
            count = 0
            for k in range(len(jointz)):
                dvs_analysis_keys.append(f"floating.jointdv_{count}",)
                count += 1
        
            for k in range(len(jointr)):
                dvs_analysis_keys.append( f"floating.jointdv_{count}")
                count += 1
        
        if float_opt["members"]["flag"]:
            for kgrp in float_opt["members"]["groups"]:
                memname = kgrp["names"][0]
                idx = modeling_options["floating"]["members"]["name2idx"][memname]
                imem = modeling_options["floating"]["members"]["name"].index(memname)
                istruct = wt_init["components"]["floating_platform"]["members"][imem]["internal_structure"]
        
                if "diameter" in kgrp:
                    # dvs_analysis_keys.append(f"floating.memgrp{idx}.outer_diameter_in")
                    dvs_analysis_keys.append(f"floating.member_{memname}:outer_diameter")
                if "side_length_a" in kgrp:
                    dvs_analysis_keys.append(f"floating.memgrp{idx}.side_length_a_in")
                    dvs_analysis_keys.append(f"floating.memgrp{idx}.side_length_b_in" )
                if "thickness" in kgrp:
                    dvs_analysis_keys.append( f"floating.memgrp{idx}.layer_thickness_in")
                if "ballast" in kgrp and len(istruct["ballasts"]) > 0:
                    V_ballast = np.zeros(len(istruct["ballasts"]))
                    for j in range(V_ballast.size):
                        if "volume" in istruct["ballasts"][j]:
                            V_ballast[j] = istruct["ballasts"][j]["volume"]
                    iball = np.where(V_ballast > 0.0)[0]
                    if iball.size > 0:
                        dvs_analysis_keys.append(f"floating.memgrp{idx}.ballast_volume",)
                if "stiffeners" in kgrp:
                    if "ring" in kgrp["stiffeners"]:
                        if "size" in kgrp["stiffeners"]["ring"]:
                            pass
                        if "spacing" in kgrp["stiffeners"]["ring"]:
                            dvs_analysis_keys.append(f"floating.memgrp{idx}.ring_stiffener_spacing")
                    if "longitudinal" in kgrp["stiffeners"]:
                        if "size" in kgrp["stiffeners"]["longitudinal"]:
                            pass
                        if "spacing" in kgrp["stiffeners"]["longitudinal"]:
                            dvs_analysis_keys.append(f"floating.memgrp{idx}.axial_stiffener_spacing")
                if "axial_joints" in kgrp:
                   
                    dvs_analysis_keys.append(f"floating.memgrp{idx}.grid_axial_joints")
                    
        
        # -- Mooring --
        if mooring_opt["line_length"]["flag"]:
            dvs_analysis_keys.append("mooring.unstretched_length_in")
        
        if mooring_opt["line_diameter"]["flag"]:
            dvs_analysis_keys.append("mooring.line_diameter_in")
        
        if mooring_opt["line_mass_density_coeff"]["flag"]:
            dvs_analysis_keys.append("mooring.line_mass_density_coeff")
        
        if mooring_opt["line_stiffness_coeff"]["flag"]:
            dvs_analysis_keys.append("mooring.line_stiffness_coeff")
        
        if "user" in opt_options["design_variables"]:
            user_defined = opt_options["design_variables"]["user"]
            for i in range(len(user_defined)):
                name_i = user_defined[i]["name"]
                dvs_analysis_keys.append(name_i)
        
        # -- Control --
        control_opt = opt_options['design_variables']['control']
        if control_opt['servo']['pitch_control']['omega']['flag']:
            dvs_analysis_keys.append('tune_rosco_ivc.omega_pc')
        if control_opt['servo']['pitch_control']['zeta']['flag']:                            
            dvs_analysis_keys.append('tune_rosco_ivc.zeta_pc')
        if control_opt['servo']['torque_control']['omega']['flag']:
            dvs_analysis_keys.append('tune_rosco_ivc.omega_vs')
        if control_opt['servo']['torque_control']['zeta']['flag']:                                                    
            dvs_analysis_keys.append('tune_rosco_ivc.zeta_vs')
        if control_opt['servo']['ipc_control']['Kp']['flag']:
            dvs_analysis_keys.append('tune_rosco_ivc.IPC_Kp1p')
        if control_opt['servo']['ipc_control']['Ki']['flag']:
            dvs_analysis_keys.append('tune_rosco_ivc.IPC_Ki1p')
        if control_opt['servo']['pitch_control']['stability_margin']['flag']:
            dvs_analysis_keys.append('tune_rosco_ivc.stability_margin')
        if control_opt['flaps']['te_flap_end']['flag']:
            dvs_analysis_keys.append('dac_ivc.te_flap_end')
        if control_opt['flaps']['te_flap_ext']['flag']:
            dvs_analysis_keys.append('dac_ivc.te_flap_ext')
        if 'flap_control' in control_opt['servo']:
            if control_opt['servo']['flap_control']['flp_kp_norm']['flag']:
                dvs_analysis_keys.append('tune_rosco_ivc.flp_kp_norm')
            if control_opt['servo']['flap_control']['flp_tau']['flag']:
                dvs_analysis_keys.append('tune_rosco_ivc.flp_tau')

        if control_opt['ps_percent']['flag']:
            dvs_analysis_keys.append('tune_rosco_ivc.ps_percent')

        if control_opt['servo']['pitch_control']['Kp_float']['flag']:
            dvs_analysis_keys.append('tune_rosco_ivc.Kp_float')

        if control_opt['servo']['pitch_control']['ptfm_freq']['flag']:
            dvs_analysis_keys.append('tune_rosco_ivc.ptfm_freq')

        if opt_options['design_variables']['TMDs']['flag']:
            TMD_opt = opt_options['design_variables']['TMDs']

            # We only support one TMD for now
            for i_group, tmd_group in enumerate(TMD_opt['groups']):
                if 'mass' in tmd_group:
                    dvs_analysis_keys.append(
                        f'TMDs.TMD_IVCs.group_{i_group}_mass')
                if 'stiffness' in tmd_group:
                    dvs_analysis_keys.append(
                        f'TMDs.TMD_IVCs.group_{i_group}_stiffness')
                    if 'natural_frequency' in tmd_group:
                        raise Exception("natural_frequency and stiffness can not be design variables in the same group")
                if 'damping' in tmd_group:
                    dvs_analysis_keys.append(
                        f'TMDs.TMD_IVCs.group_{i_group}_damping')
                    if 'damping_ratio' in tmd_group:
                        raise Exception("damping_ratio and damping can not be design variables in the same group")
                if 'natural_frequency' in tmd_group:
                    dvs_analysis_keys.append(
                        f'TMDs.TMD_IVCs.group_{i_group}_natural_frequency')
                if 'damping_ratio' in tmd_group:
                    dvs_analysis_keys.append(
                        f'TMDs.TMD_IVCs.group_{i_group}_damping_ratio' )
        
        return dvs_analysis_keys

dvs_analysis_keys = get_design_variables()


# identify available inputs and outputs provided by the sm
opt_dv_sm = model['inputs']
opt_dv_key_sm = opt_dv_sm['keys']
bounds_sm = opt_dv_sm['bounds']
sm_to_analysis_idx = [opt_dv_key_sm.index(element) for element in dvs_analysis_keys if element in opt_dv_key_sm]
opt_dv_key = [opt_dv_key_sm[key] for key in sm_to_analysis_idx]
bounds = np.array([bounds_sm[:,key] for key in sm_to_analysis_idx]).T

# %%
# get some needed modeling info
if modeling_options['Level3']['flag']:
    floating_solve_component = 'aeroelastic'
elif modeling_options['Level1']['flag']:
    floating_solve_component = 'raft'
else:
    floating_solve_component = 'floatingse'

# identify constraints from analysis yaml
def get_constraints(opt_options): #these cons are taken from weis and wisdem gc_PoseOptimization, not all cons are included here
        const_keys = []
        const_bounds = np.array([[], []])

        # blade_constr = opt_options["constraints"]["blade"]
        # if blade_constr['tip_deflection']['flag']:
        #     # Remove generic WISDEM one
        #     name = 'tcons.tip_deflection_ratio'
        #     if name in wt_opt.model._responses:
        #         wt_opt.model._responses.pop( name )
        #     if name in wt_opt.model._static_responses:
        #         wt_opt.model._static_responses.pop( name )
                
        #     if blade_opt['structure']['spar_cap_ss']['flag'] or blade_opt['structure']['spar_cap_ps']['flag']:
        #         wt_opt.model.add_constraint('tcons_post.tip_deflection_ratio', upper=1.0)
        #     else:
        #         print('WARNING: the tip deflection is set to be constrained, but spar caps thickness is not an active design variable. The constraint is not enforced.')

        # if blade_constr["strains_spar_cap_ss"]["flag"]:
        #     # Remove generic WISDEM one
        #     name = 'rotorse.rs.constr.constr_max_strainU_spar'
        #     if name in wt_opt.model._responses:
        #         wt_opt.model._responses.pop( name )
        #     if name in wt_opt.model._static_responses:
        #         wt_opt.model._static_responses.pop( name )
        #     if blade_opt["structure"]["spar_cap_ss"]["flag"]:
        #         indices_strains_spar_cap_ss = range(blade_constr["strains_spar_cap_ss"]["index_start"], blade_constr["strains_spar_cap_ss"]["index_end"])
        #         wt_opt.model.add_constraint("rlds_post.constr.constr_max_strainU_spar", indices = indices_strains_spar_cap_ss, upper=1.0)

        # if blade_constr["strains_spar_cap_ps"]["flag"]:
        #     if (
        #         blade_opt["structure"]["spar_cap_ps"]["flag"]
        #         or blade_opt["structure"]["spar_cap_ps"]["equal_to_suction"]
        #     ):
        #         # Remove generic WISDEM one
        #         name = 'rotorse.rs.constr.constr_max_strainL_spar'
        #         if name in wt_opt.model._responses:
        #             wt_opt.model._responses.pop( name )
        #         if name in wt_opt.model._static_responses:
        #             wt_opt.model._static_responses.pop( name )
        #         indices_strains_spar_cap_ps = range(blade_constr["strains_spar_cap_ps"]["index_start"], blade_constr["strains_spar_cap_ps"]["index_end"])
        #         wt_opt.model.add_constraint("rlds_post.constr.constr_max_strainL_spar", indices = indices_strains_spar_cap_ps, upper=1.0)

        ### CONTROL CONSTRAINTS
        control_constraints = opt_options['constraints']['control']
        np.array([[-5,1], [5,-1]])
        # Flap control
        if control_constraints['flap_control']['flag']:

            const_keys.append('sse_tune.tune_rosco.flptune_coeff1')
            lower = control_constraints['flap_control']['min']
            upper = control_constraints['flap_control']['max']
            const_bounds = np.hstack((const_bounds,[[lower],[upper]]))
            const_keys.append('sse_tune.tune_rosco.flptune_coeff2')
            lower = control_constraints['flap_control']['min']
            upper = control_constraints['flap_control']['max']   
            const_bounds = np.hstack((const_bounds,[[lower],[upper]]))

        
        # Rotor overspeed
        if control_constraints['rotor_overspeed']['flag']:
            const_keys.append(f'{floating_solve_component}.rotor_overspeed')
            lower = control_constraints['rotor_overspeed']['min']
            upper = control_constraints['rotor_overspeed']['max']
            const_bounds = np.hstack((const_bounds,[[lower],[upper]]))

        
        # Add PI gains if overspeed is merit_figure or constraint
        # if control_constraints['rotor_overspeed']['flag'] or self.opt['merit_figure'] == 'rotor_overspeed':
        #     wt_opt.model.add_constraint('sse_tune.tune_rosco.PC_Kp',
        #         upper = 0.0)
        #     wt_opt.model.add_constraint('sse_tune.tune_rosco.PC_Ki', 
        #         upper = 0.0)  
        
        # Nacelle Accelleration magnitude
        if control_constraints['nacelle_acceleration']['flag']:
            const_keys.append(f'{floating_solve_component}.max_nac_accel')
            upper = control_constraints['nacelle_acceleration']['max']
            lower = -1e+30
            const_bounds = np.hstack((const_bounds,[[lower],[upper]]))
        
        # Max platform pitch
        if control_constraints['Max_PtfmPitch']['flag']:
            const_keys.append(f'{floating_solve_component}.Max_PtfmPitch')
            lower = -1e+30
            upper = control_constraints['Max_PtfmPitch']['max']
            const_bounds = np.hstack((const_bounds,[[lower],[upper]]))
       
        # Platform pitch motion
        if control_constraints['Std_PtfmPitch']['flag']:
            const_keys.append(f'{floating_solve_component}.Std_PtfmPitch')
            upper = control_constraints['Std_PtfmPitch']['max']
            lower = -1e+30
            const_bounds = np.hstack((const_bounds,[[lower],[upper]]))
            
        if control_constraints['Max_TwrBsMyt']['flag']:
            const_keys.append('aeroelastic.max_TwrBsMyt_ratio') 
            upper = 1.0
            lower = -1e+30
            const_bounds = np.hstack((const_bounds,[[lower],[upper]]))
            
        if control_constraints['DEL_TwrBsMyt']['flag']:
            const_keys.append('aeroelastic.DEL_TwrBsMyt_ratio')
            upper = 1.0
            lower = -1e+30
            const_bounds = np.hstack((const_bounds,[[lower],[upper]]))
            
        # Blade pitch travel
        if control_constraints['avg_pitch_travel']['flag']:
            const_keys.append('aeroelastic.avg_pitch_travel')
            upper = control_constraints['avg_pitch_travel']['max']
            lower = -1e+30
            const_bounds = np.hstack((const_bounds,[[lower],[upper]]))

        # Blade pitch duty cycle (number of direction changes)
        if control_constraints['pitch_duty_cycle']['flag']:
            const_keys.append('aeroelastic.pitch_duty_cycle')
            upper = control_constraints['pitch_duty_cycle']['max']
            lower = -1e+30
            const_bounds = np.hstack((const_bounds,[[lower],[upper]]))

        # Max offset
        if opt_options['constraints']['floating']['Max_Offset']['flag']:
            const_keys.append(f'{floating_solve_component}.Max_Offset')
            upper = opt_options['constraints']['floating']['Max_Offset']['max'] 
            lower = -1e+30
            const_bounds = np.hstack((const_bounds,[[lower],[upper]]))            
                                           
        # Tower constraints
        tower_constr = opt_options["constraints"]["tower"]
        monopile_constr = opt_options["constraints"]["monopile"]

        # if tower_constr["global_buckling"]["flag"] and self.modeling['Level3']['flag']:
        #     # Remove generic WISDEM one
        #     name = 'towerse.post.constr_global_buckling'
        #     if name in wt_opt.model._responses:
        #         wt_opt.model._responses.pop( name )
        #     if name in wt_opt.model._static_responses:
        #         wt_opt.model._static_responses.pop( name )
                
        #     wt_opt.model.add_constraint("towerse_post.constr_global_buckling", upper=1.0)
        
        # if tower_constr["shell_buckling"]["flag"] and self.modeling['Level3']['flag']:
        #     # Remove generic WISDEM one
        #     name = 'towerse.post.constr_shell_buckling'
        #     if name in wt_opt.model._responses:
        #         wt_opt.model._responses.pop( name )
        #     if name in wt_opt.model._static_responses:
        #         wt_opt.model._static_responses.pop( name )
                
        #     wt_opt.model.add_constraint("towerse_post.constr_shell_buckling", upper=1.0)
        
        # if tower_constr["stress"]["flag"] and self.modeling['Level3']['flag']:
        #     # Remove generic WISDEM one
        #     name = 'towerse.post.constr_stress'
        #     if name in wt_opt.model._responses:
        #         wt_opt.model._responses.pop( name )
        #     if name in wt_opt.model._static_responses:
        #         wt_opt.model._static_responses.pop( name )
                
        #     wt_opt.model.add_constraint("towerse_post.constr_stress", upper=1.0)
        
        if tower_constr["height_constraint"]["flag"]:
            const_keys.append("towerse.height_constraint")
            lower=tower_constr["height_constraint"]["lower_bound"]
            upper=tower_constr["height_constraint"]["upper_bound"]     
            const_bounds = np.hstack((const_bounds,[[lower],[upper]]))

        if tower_constr["stress"]["flag"] and not monopile_constr["stress"]["flag"]:
            const_keys.append("towerse.post.constr_stress") 
            upper=1.0
            lower = -1e+30
            const_bounds = np.hstack((const_bounds,[[lower],[upper]]))

        if tower_constr["global_buckling"]["flag"] and not monopile_constr["global_buckling"]["flag"]:
            const_keys.append("towerse.post.constr_global_buckling")
            upper=1.0
            lower = -1e+30
            const_bounds = np.hstack((const_bounds,[[lower],[upper]]))

        if tower_constr["shell_buckling"]["flag"] and not monopile_constr["shell_buckling"]["flag"]:
            const_keys.append("towerse.post.constr_shell_buckling") 
            upper=1.0
            lower = -1e+30
            const_bounds = np.hstack((const_bounds,[[lower],[upper]]))

        if tower_constr["d_to_t"]["flag"]:
            const_keys.append("towerse.constr_d_to_t")
            lower=tower_constr["d_to_t"]["lower_bound"]
            upper=tower_constr["d_to_t"]["upper_bound"]
            const_bounds = np.hstack((const_bounds,[[lower],[upper]]))

        if tower_constr["taper"]["flag"]:
            const_keys.append("towerse.constr_taper")
            lower=tower_constr["taper"]["lower_bound"]
            upper = 1e+30
            const_bounds = np.hstack((const_bounds,[[lower],[upper]]))

        if tower_constr["slope"]["flag"]:
            const_keys.append("towerse.slope")
            lower = -1e+30
            upper=1.0
            const_bounds = np.hstack((const_bounds,[[lower],[upper]]))

        if tower_constr["thickness_slope"]["flag"]:
            const_keys.append("towerse.thickness_slope")
            lower = -1e+30
            upper=1.0
            const_bounds = np.hstack((const_bounds,[[lower],[upper]]))

        if tower_constr["frequency"]["flag"]:
            const_keys.append("tcons.constr_tower_f_1Pmargin")
            upper=0.0
            lower = -1e+30
            const_bounds = np.hstack((const_bounds,[[lower],[upper]]))
            const_keys.append("tcons.constr_tower_f_NPmargin")
            upper=0.0
            lower = -1e+30
            const_bounds = np.hstack((const_bounds,[[lower],[upper]]))

        elif tower_constr["frequency_1"]["flag"]:
            # Cannot set both 1P/3P and hard freq values, so else-if statement here
            varstr = (
                "floatingse.structural_frequencies_0"
                if modeling_options["flags"]["floating"]
                else "towerse.tower.structural_frequencies"
            )
            const_keys.append(varstr)
            lower=tower_constr["frequency_1"]["lower_bound"]
            upper=tower_constr["frequency_1"]["upper_bound"]
            const_bounds = np.hstack((const_bounds,[[lower],[upper]]))
            

        # # Damage constraints
        # damage_constraints = self.opt['constraints']['damage']
        # if damage_constraints['tower_base']['flag'] and (self.modeling['Level2']['flag'] or self.modeling['Level3']['flag']):
        #     if self.modeling['Level3']['flag'] != True:
        #         raise Exception('Please turn on the call to OpenFAST if you are trying to optimize with tower_base damage constraint.')

        #     tower_base_damage_max = damage_constraints['tower_base']['max']
        #     if damage_constraints['tower_base']['log']:
        #         tower_base_damage_max = np.log(tower_base_damage_max)

        #     wt_opt.model.add_constraint('aeroelastic.damage_tower_base',upper = tower_base_damage_max)

        return const_keys, const_bounds


def run_model():    #run model w/o optimization to get x0
    #create copies of the analysis and modeling yaml
    name, ext = os.path.splitext(fname_analysis_options)
    fname_analysis_options_sim = f"{name}_sim{ext}"          # Add "_sim" before the file extension
    name, ext = os.path.splitext(fname_modeling_options)
    fname_modeling_options_sim = f"{name}_sim{ext}"          # Add "_sim" before the file extension
    
    # open copied yamls and modify them 
    with open(fname_analysis_options, 'r') as original_file:
        analysis_dict = yaml.safe_load(original_file)
        
    analysis_dict['recorder']['flag'] = False
    analysis_dict['driver']['design_of_experiments']['flag'] = False
    analysis_dict['driver']['optimization']['flag'] = False
    
    
    with open(fname_analysis_options_sim, 'w') as copy_file:
        yaml.safe_dump(analysis_dict, copy_file,sort_keys=False)
        
        #copy and edit modeling options file
    with open(fname_modeling_options, 'r') as original_file:
        modeling_dict = yaml.safe_load(original_file)
        
    modeling_dict['DLC_driver'] = {}
    modeling_dict['DLC_driver']['DLCs'] = [{'DLC': "1.1", 'ws_bin_size': 1, 'n_seeds': 1, 'analysis_time': 0.5, 'transient_time': 0.5}] 
    
    with open(fname_modeling_options_sim, 'w') as copy_file:
        yaml.safe_dump(modeling_dict, copy_file,sort_keys=False)
    
    # t0 = time.time()
    # run WEIS (only simulation) to get x0 values
    wt_opt_sim, modeling_options_sim, opt_options_sim = run_weis(fname_wt_input, fname_modeling_options_sim, fname_analysis_options_sim, overridden_values)
    
    # t1 = time.time()
    # total = t1-t0
    # print(total)
    return wt_opt_sim


# get constraints and objective
constraints_key, constraints_bounds = get_constraints(opt_options)
if opt_options['merit_figure']=='platform_mass':
    objective_key =  f'{floating_solve_component}.platform_mass'
else:
    objective_key = opt_options['merit_figure']

#get x0 values and f(x0)
DesignVars_x0 = np.zeros((len(opt_dv_key),))
wt_opt_sim = run_model()
for k in range(len(opt_dv_key)):
    var = opt_dv_key[k]
    x0 = wt_opt_sim.get_val(var)
    DesignVars_x0[k] = x0[0]
fx0 = wt_opt_sim.get_val(objective_key)   

# %%
#create dictionary with the optimization problem
DCA_problem = {
    'DesignVars': {
        'DesignVars_keys': opt_dv_key,
        'DesignVars_x0': DesignVars_x0,
        'bounds': bounds
    },
    'DesignParms': {
        'DesignParms_keys': [],
        'DesignParms_values': []
    },
    'objective': objective_key,
    'constraints': {
        'constraints_keys': constraints_key,
        'constraints_bounds': constraints_bounds
    }
}


def opt_all_dvs():  # run SM opt with all design variables
    WTSMO.get_opt_vars(DCA_problem) 
    WTSMO.run_optimization(DCA_problem)
    WTSMO.save_opt_output(opt_filename)
    
    #read optimization results
    try:
        with open(opt_filename, 'rb') as f:
            opt_output_optall = pkl.load(f)
    except:
        print('Unable to read optimization result file: {:}.'.format(opt_filename))
        raise Exception('Unable to read optimization result file: {:}.'.format(opt_filename))
    return opt_output_optall

# get results from optimization w/ all dvs
opt_output_optall = opt_all_dvs()
xst_optall = opt_output_optall['dvs']['dv_values'] 
fst_optall = opt_output_optall['objective']['objective_values']     
keys_optall = opt_output_optall['dvs']['dv_keys']
# get exit flag from runSBO (capability needs to be added)


# Reorder opt_dv_key, bounds and DesignVars_x0 according to xst_optall
key_to_x0 = dict(zip(opt_dv_key, DesignVars_x0))
opt_dv_key_ordered = keys_optall
DesignVars_x0_ordered = [key_to_x0[key] for key in keys_optall]
# Find the indices of the new order
index_map = [opt_dv_key.index(key) for key in opt_dv_key_ordered]
# Reorder the columns of bounds
bounds_ordered = bounds[:, index_map]

# Update the dictionary with the reordered lists
DCA_problem['DesignVars']['DesignVars_keys'] = opt_dv_key_ordered
DCA_problem['DesignVars']['DesignVars_x0'] = DesignVars_x0_ordered
DCA_problem['DesignVars']['DesignVars_x0'] = list(DCA_problem['DesignVars']['DesignVars_x0'])

# run optimizations with different dvs to get sensitivity matrix
req_dv = DCA_problem["DesignVars"]["DesignVars_keys"]
# req_design_parms = DCA_valid["DesignParms"]["DesignParms_keys"]
# %%
####################################################################
# method 1: Saeid's method
# xst_opt_fixed = np.zeros((len(req_dv)-1,len(req_dv)))
# fst_opt_fixed = np.zeros((len(req_dv),))
# for k in range(len(req_dv)):
#     new_param = req_dv[k]
#     DCA_req_fixed = {}
#     DCA_req_fixed['constraints'] = DCA_problem["constraints"]
#     DCA_req_fixed['DesignParms'] = {}
#     DCA_req_fixed["DesignParms"]["DesignParms_keys"] = DCA_problem["DesignParms"]["DesignParms_keys"] + [new_param]

#     DCA_req_fixed["DesignParms"]["DesignParms_values"] = np.concatenate((DCA_problem["DesignParms"]["DesignParms_values"],DCA_problem['DesignVars']["DesignVars_x0"][k]),None)
#     DCA_req_fixed['DesignVars'] = {}
#     DCA_req_fixed['DesignVars']["DesignVars_keys"] = DCA_problem['DesignVars']["DesignVars_keys"][:k]+DCA_problem['DesignVars']["DesignVars_keys"][k+1:]
#     DCA_req_fixed['objective'] = DCA_problem["objective"]
       
#     # run opt with all except design variables
#     WTSMO.get_opt_vars(DCA_req_fixed) 
#     WTSMO.run_optimization(DCA_req_fixed)
#     WTSMO.save_opt_output(opt_filename)
#     try:
#         with open(opt_filename, 'rb') as f:
#             opt_output = pkl.load(f)
#     except:
#         print('Unable to read optimization result file: {:}.'.format(opt_filename))
#         raise Exception('Unable to read optimization result file: {:}.'.format(opt_filename))
    
#     xst_opt_fixed[:,k] = opt_output['dvs']['dv_values']
#     print(opt_output['dvs']['dv_values'])
#     fst_opt_fixed[k]  = opt_output['objective']['objective_values']


# # write matrix with resutls
# xst_matrix = np.zeros((len(req_dv),len(req_dv)))

# for i in range(len(req_dv)):
#     xst_matrix[:i, i] = xst_opt_fixed[:i, i]
#     xst_matrix[i+1:, i] = xst_opt_fixed[i:, i]
#     xst_matrix[i,i] = DCA_problem['DesignVars']['DesignVars_x0'][i]
 
# Del_x_scaled_fixed = (np.diag(xst_matrix)- xst_optall)/np.diag(xst_matrix)
# Del_x_scaled_all = (np.diag(xst_matrix)- xst_optall)/xst_optall

# Del_xst_matrix = (xst_matrix - (xst_optall*np.ones((len(req_dv),len(req_dv)))).T)/(xst_optall*np.ones((len(req_dv),len(req_dv)))).T
# # np.fill_diagonal(Del_xst_matrix, np.nan)    
# np.fill_diagonal(Del_xst_matrix, 0)    

# Del_xst_matrix_Del_x_scaled_all =  Del_xst_matrix/Del_x_scaled_all
# sum_row = np.sum(Del_xst_matrix_Del_x_scaled_all, axis=0)
# sum_col = np.sum(Del_xst_matrix_Del_x_scaled_all, axis=1)

# Del_f_scaled_fixed = (fst_opt_fixed-fst_optall)/fst_opt_fixed
# Del_f_scaled_all = (fst_opt_fixed-fst_optall)/fst_optall
# %%
def gen_sweep_points(x0, c, n_sweeps, lb, ub): # provides x0, x0+perturbation
    start = x0
    end = x0 + c*(ub-lb)
    array = np.linspace(start, end, n_sweeps)
    return array

def opt_sweep(DCA_req_fixed,opt_filename,fname_ccdc_options):
    # run opt with all except design variables
    WTSMO.get_opt_vars(DCA_req_fixed) 
    WTSMO.run_optimization(DCA_req_fixed)
    WTSMO.save_opt_output(opt_filename)
    try:
        with open(opt_filename, 'rb') as f:
            opt_output = pkl.load(f)
    except:
        print('Unable to read optimization result file: {:}.'.format(opt_filename))
        raise Exception('Unable to read optimization result file: {:}.'.format(opt_filename))
        
        
    solver =  ccdc_options['SMB_optimization']['solver']
    
    # if solver=='SLSQP' and  opt_output['exit_flag']!=0:    #switch to COBYLA if SLSQP fails
    if opt_output['success']!=1:     
        #create copies of the analysis and modeling yaml
        name, ext = os.path.splitext(fname_ccdc_options)
        fname_ccdc_options_1 = f"{name}_1{ext}"          # Add "_1" before the file extension
        with open(fname_ccdc_options, 'r') as opt_file:
            options_dict = yaml.safe_load(opt_file)
            
        options_dict['SMB_optimization']['solver'] = 'COBYLA'
        
        with open(fname_ccdc_options_1, 'w') as opt_file:
            yaml.safe_dump(options_dict, opt_file,sort_keys=False)
            
        ccdc_options_1 = load_yaml(fname_ccdc_options_1) 
        WTSMO_1 = WindTurbineSMOpt(ccdc_options_1)
        WTSMO_1.read_sm(sm_file, modeling_options)
        WTSMO_1.get_opt_vars(DCA_req_fixed) 
        WTSMO_1.run_optimization(DCA_req_fixed)
        WTSMO_1.save_opt_output(opt_filename)
        try:
            with open(opt_filename, 'rb') as f:
                opt_output = pkl.load(f)
        except:
            print('Unable to read optimization result file: {:}.'.format(opt_filename))
            raise Exception('Unable to read optimization result file: {:}.'.format(opt_filename))
            
        # if opt_output['exit_flag']!=1:  #switch to DE if COBYLA fails
        if opt_output['success']!=1: 

            #create copies of the analysis and modeling yaml
            name, ext = os.path.splitext(fname_ccdc_options)
            fname_ccdc_options_2 = f"{name}_2{ext}"          # Add "_2" before the file extension
            with open(fname_ccdc_options, 'r') as opt_file:
                options_dict = yaml.safe_load(opt_file)
                
            # options_dict['SMB_optimization']['solver'] = 'NSGA2'
            options_dict['SMB_optimization']['solver'] = 'Nelder-Mead'
            with open(fname_ccdc_options_2, 'w') as opt_file:
                yaml.safe_dump(options_dict, opt_file,sort_keys=False)
                
            ccdc_options_2 = load_yaml(fname_ccdc_options_2) 
            WTSMO_2 = WindTurbineSMOpt(ccdc_options_2)
            WTSMO_2.read_sm(sm_file, modeling_options)
            WTSMO_2.get_opt_vars(DCA_req_fixed) 
            WTSMO_2.run_optimization(DCA_req_fixed)
            WTSMO_2.save_opt_output(opt_filename)
            try:
                with open(opt_filename, 'rb') as f:
                    opt_output = pkl.load(f)
            except:
                print('Unable to read optimization result file: {:}.'.format(opt_filename))
                raise Exception('Unable to read optimization result file: {:}.'.format(opt_filename))
            if opt_output['success']==True:
                exitflag_opt = 0
            else:
                exitflag_opt = 100
            success_opt = opt_output['success']

        if opt_output['exit_flag']==1:
            exitflag_opt = 0
        
        success_opt = opt_output['success']
    else:
        exitflag_opt = opt_output['exit_flag']
        success_opt = opt_output['success']
    
    if len(DCA_req_fixed['DesignVars']["DesignVars_keys"])==1:
        xst_opt = opt_output['dvs']['dv_values'][0]
    else:
        xst_opt = opt_output['dvs']['dv_values']
    fst_opt = opt_output['objective']['objective_values'][0]
    # print(opt_output)
    return xst_opt, fst_opt, exitflag_opt, success_opt

## #################################################################  
# Approach 1: One perturbed variable, one optimization variable, one perturbation
def jacobian_app1():
    n_sweeps = 2
    percent = 0.01
    
    x_jac = np.zeros((len(req_dv),len(req_dv)-1))
    exitflag = [] 
    constraint_violationv = []   
    first_order_optimalityv = []  
    gobj_st = np.zeros((len(req_dv),len(req_dv)-1))
    for q in range(len(req_dv)):
    
        Design_Vars = req_dv[q]          
        AllDesign_Parms = req_dv[:q] + req_dv[q+1:]
        AllDesign_Parms_x0 = DCA_problem['DesignVars']['DesignVars_x0'][:q] + DCA_problem['DesignVars']['DesignVars_x0'][q+1:] 
        xst_opt_fixed = np.zeros((n_sweeps,len(AllDesign_Parms)))
        fst_opt_fixed = np.zeros((n_sweeps,len(AllDesign_Parms)))
        grad_x_st = np.zeros((len(AllDesign_Parms),))
        grad_f_st = np.zeros((len(AllDesign_Parms),))
        x_lb = bounds_ordered[0,q]
        x_ub = bounds_ordered[1,q]
        
        for p in range(len(AllDesign_Parms)):
            # design parameters (fixed)
            Design_Parms = AllDesign_Parms[:p] + AllDesign_Parms[p+1:]
            Design_Parms_x0 = AllDesign_Parms_x0[:p] + AllDesign_Parms_x0[p+1:]
            # Sweep parameter (perturbed)
            Sweep_Parms = AllDesign_Parms[p]
            Sweep_Parms_indx = DCA_problem['DesignVars']["DesignVars_keys"].index(Sweep_Parms)
            lb = bounds_ordered[0,Sweep_Parms_indx]
            ub = bounds_ordered[1,Sweep_Parms_indx]
            x0_Sweep_Parms = DCA_problem['DesignVars']["DesignVars_x0"][Sweep_Parms_indx]
            points = gen_sweep_points(x0_Sweep_Parms, percent, n_sweeps, lb, ub)
     
            for sweep_ind in range(n_sweeps):
                
                xp = points[sweep_ind] #value of the perturbed parameter
                #create optimization problem dictionary
                DCA_req_fixed = {}
                DCA_req_fixed['constraints'] = DCA_problem["constraints"]
                DCA_req_fixed['DesignParms'] = {}
                DCA_req_fixed["DesignParms"]["DesignParms_keys"] = DCA_problem["DesignParms"]["DesignParms_keys"] + Design_Parms + [Sweep_Parms]
                DCA_req_fixed["DesignParms"]["DesignParms_values"] = np.concatenate((DCA_problem["DesignParms"]["DesignParms_values"],Design_Parms_x0, xp),None)
                DCA_req_fixed['DesignVars'] = {}
                DCA_req_fixed['DesignVars']["DesignVars_keys"] = [Design_Vars]
                DCA_req_fixed['objective'] = DCA_problem["objective"]
                   
                xst_opt_fixed[sweep_ind,p], fst_opt_fixed[sweep_ind,p], exit_flag= opt_sweep(DCA_req_fixed,opt_filename,fname_ccdc_options)
                exitflag.append(exit_flag)   
    
        
        xst_opt_fixed_scaled = (xst_opt_fixed-x_lb)/(x_ub-x_lb)
        dx_scaled = (points-points[0])/(points[-1]-points[0])
        
        
        for p in range(len(AllDesign_Parms)):
            grad_x_st[p] =  np.linalg.norm(np.gradient(xst_opt_fixed_scaled[:,p], dx_scaled.T))
            grad_f_st[p] =  np.linalg.norm(np.gradient(fst_opt_fixed[:,p]/fx0, dx_scaled.T))
            
        gobj_st[q,:]=grad_f_st.T
        x_jac[q,:]=grad_x_st.T
    
    
    dc_jacobian = np.zeros((len(req_dv), len(req_dv)), dtype=float)
    obj_jacobian = np.zeros((len(req_dv), len(req_dv)), dtype=float)
    
    # Copy the original array to the new array, skipping the diagonal
    for i in range(len(req_dv)):
        dc_jacobian[i, :i] = x_jac[i, :i]
        dc_jacobian[i, i+1:] = x_jac[i, i:]
        obj_jacobian[i, :i] = gobj_st[i, :i]
        obj_jacobian[i, i+1:] = gobj_st[i, i:]
    
    return dc_jacobian, obj_jacobian, exitflag

# dc_jacobian, obj_jacobian, exitflag_jac =  jacobian_app1()   

## #################################################################  
# Approach 2: One perturbed variable, one optimization variable, multiple sweeps
def jacobian_app2():
    n_sweeps = 100
    # n_sweeps = 4
    fst_opt_fixed = np.zeros((n_sweeps,len(req_dv)-1))
    x_jac = np.zeros((len(req_dv),len(req_dv)-1))
    exitflag = [] 
    succflag = [] 
    gobj_st = np.zeros((len(req_dv),len(req_dv)-1))
    for q in range(len(req_dv)):    
    
        Design_Vars = req_dv[q]   
        # print('design vars', Design_Vars)       
        AllDesign_Parms = req_dv[:q] + req_dv[q+1:]
        AllDesign_Parms_x0 = DCA_problem['DesignVars']['DesignVars_x0'][:q] + DCA_problem['DesignVars']['DesignVars_x0'][q+1:] 
        xst_opt_fixed = np.zeros((n_sweeps,len(AllDesign_Parms)))
        xst_opt_fixed_scaled = np.zeros((n_sweeps,len(AllDesign_Parms)))
        grad_x_st = np.zeros((len(AllDesign_Parms),))
        grad_f_st = np.zeros((len(AllDesign_Parms),))
        x_lb = bounds_ordered[0,q]
        x_ub = bounds_ordered[1,q]
        # print('dv bounds', x_lb, x_ub)
        for p in range(len(AllDesign_Parms)):
            Sweep_Parms = AllDesign_Parms[p]
            Design_Parms = AllDesign_Parms[:p] + AllDesign_Parms[p+1:]
            Design_Parms_x0 = AllDesign_Parms_x0[:p] + AllDesign_Parms_x0[p+1:]
            parms_indx = DCA_problem['DesignVars']["DesignVars_keys"].index(Sweep_Parms)
            lb = bounds_ordered[0,parms_indx]
            ub = bounds_ordered[1,parms_indx]
            x0_parms = DCA_problem['DesignVars']["DesignVars_x0"][parms_indx]
            points = np.linspace(lb, ub, n_sweeps)
            # print('sweep params', Sweep_Parms)  
            # print(' bounds', lb, ub)
            for sweep_ind in range(n_sweeps):
               
                xp = points[sweep_ind]
                    
                DCA_req_fixed = {}
                DCA_req_fixed['constraints'] = DCA_problem["constraints"]
                DCA_req_fixed['DesignParms'] = {}
                DCA_req_fixed["DesignParms"]["DesignParms_keys"] = DCA_problem["DesignParms"]["DesignParms_keys"] + Design_Parms + [Sweep_Parms]
                DCA_req_fixed["DesignParms"]["DesignParms_values"] = np.concatenate((DCA_problem["DesignParms"]["DesignParms_values"],Design_Parms_x0, xp),None)
                DCA_req_fixed['DesignVars'] = {}
                DCA_req_fixed['DesignVars']["DesignVars_keys"] = [Design_Vars]
                DCA_req_fixed['objective'] = DCA_problem["objective"]
                
                xst_opt_fixed[sweep_ind,p], fst_opt_fixed[sweep_ind,p], exit_flag, succ_flag = opt_sweep(DCA_req_fixed,opt_filename,fname_ccdc_options)
                exitflag.append(exit_flag)   
                succflag.append(succ_flag)   
                
            xst_opt_fixed_scaled[:,p] = (xst_opt_fixed[:,p]-x_lb)/(x_ub-x_lb)
            dx_scaled = (points-points[0])/(points[-1]-points[0])
            grad_x_st[p] =  np.linalg.norm(np.gradient(xst_opt_fixed_scaled[:,p], dx_scaled.T))
            grad_f_st[p] =  np.linalg.norm(np.gradient(fst_opt_fixed[:,p]/fx0, dx_scaled.T))
            
            
        x_jac[q,:]=grad_x_st.T
        gobj_st[q,:]=grad_f_st.T
    
    
    dc_jacobian = np.zeros((len(req_dv), len(req_dv)), dtype=float)
    obj_jacobian = np.zeros((len(req_dv), len(req_dv)), dtype=float)
    
    # Copy the original array to the new array, skipping the diagonal
    for i in range(len(req_dv)):
        dc_jacobian[i, :i] = x_jac[i, :i]
        dc_jacobian[i, i+1:] = x_jac[i, i:]
        obj_jacobian[i, :i] = gobj_st[i, :i]
        obj_jacobian[i, i+1:] = gobj_st[i, i:]

    return dc_jacobian, obj_jacobian, exitflag
dc_jacobian, obj_jacobian, exitflag_jac =  jacobian_app2()  

# ## #################################################################  
# # Approach 3: One perturbed variable, multiple optimization variables, one perturbation 
def jacobian_app3():
    n_sweeps = 2
    percent = 0.01
    
    fst_opt_fixed = np.zeros((n_sweeps,))
        
    x_jac = np.zeros((len(req_dv),len(req_dv)-1))
    exitflag = [] 
    gobj_st = np.zeros((len(req_dv),))
    for q in range(len(req_dv)):
    
        Sweep_Parms = req_dv[q]          
        Design_Vars = req_dv[:q] + req_dv[q+1:]
        AllDesign_Parms_x0 = DCA_problem['DesignVars']['DesignVars_x0'][:q] + DCA_problem['DesignVars']['DesignVars_x0'][q+1:] 
        xst_opt_fixed = np.zeros((n_sweeps,len(Design_Vars)))
        xst_opt_fixed_scaled = np.zeros((n_sweeps,len(Design_Vars)))
        grad_x_st = np.zeros((len(Design_Vars),))
        parms_indx = DCA_problem['DesignVars']["DesignVars_keys"].index(Sweep_Parms)
        lb = bounds_ordered[0,parms_indx]
        ub = bounds_ordered[1,parms_indx]
        x0_parms = DCA_problem['DesignVars']["DesignVars_x0"][parms_indx]  
        points = gen_sweep_points(x0_parms, percent, n_sweeps, lb, ub)
        for sweep_ind in range(n_sweeps):
            
            xp = points[sweep_ind]
                
            DCA_req_fixed = {}
            DCA_req_fixed['constraints'] = DCA_problem["constraints"]
            DCA_req_fixed['DesignParms'] = {}
            DCA_req_fixed["DesignParms"]["DesignParms_keys"] = DCA_problem["DesignParms"]["DesignParms_keys"] + [Sweep_Parms]
            DCA_req_fixed["DesignParms"]["DesignParms_values"] = np.concatenate((DCA_problem["DesignParms"]["DesignParms_values"], xp),None)
            DCA_req_fixed['DesignVars'] = {}
            DCA_req_fixed['DesignVars']["DesignVars_keys"] = Design_Vars
            DCA_req_fixed['objective'] = DCA_problem["objective"]
            xst_opt_fixed[sweep_ind,:], fst_opt_fixed[sweep_ind], exit_flag = opt_sweep(DCA_req_fixed,opt_filename,fname_ccdc_options)
            exitflag.append(exit_flag) 
        
        dx_scaled = (points-points[0])/(points[-1]-points[0])
        gobj_st[q] = np.linalg.norm(np.gradient(fst_opt_fixed/fx0,  dx_scaled.T))
        
        for p in range(len(Design_Vars)):
            x_lb = bounds_ordered[0,p]
            x_ub = bounds_ordered[1,p]
            xst_opt_fixed_scaled[:,p]  = (xst_opt_fixed[:,p]-x_lb)/(x_ub-x_lb)
            grad_x_st[p] =  np.linalg.norm(np.gradient(xst_opt_fixed_scaled[:,p], dx_scaled.T))
              
        x_jac[q,:]=grad_x_st.T
    
    dc_jacobian = np.zeros((len(req_dv), len(req_dv)), dtype=float)
    obj_jacobian = gobj_st.T
    
    # Copy the original array to the new array, skipping the diagonal
    for i in range(len(req_dv)):
        dc_jacobian[i, :i] = x_jac[i, :i]
        dc_jacobian[i, i+1:] = x_jac[i, i:]
        
    return dc_jacobian, obj_jacobian, exitflag
# dc_jacobian, obj_jacobian, exitflag_jac =  jacobian_app3()  

# ## #################################################################  
# # Approach 4: One perturbed variable, multiple optimization variables, multiple sweeps
def jacobian_app4():
    n_sweeps = 100
    
    fst_opt_fixed = np.zeros((n_sweeps,))
    
    x_jac = np.zeros((len(req_dv),len(req_dv)-1))
    exitflag = [] 
    constraint_violationv = []   
    first_order_optimalityv = []  
    gobj_st = np.zeros((len(req_dv),))
    for q in range(len(req_dv)):
        
        Sweep_Parms = req_dv[q]     
        Design_Vars = req_dv[:q] + req_dv[q+1:]
        AllDesign_Parms_x0 = DCA_problem['DesignVars']['DesignVars_x0'][:q] + DCA_problem['DesignVars']['DesignVars_x0'][q+1:] 
        xst_opt_fixed = np.zeros((n_sweeps,len(Design_Vars)))
        grad_x_st = np.zeros((len(Design_Vars),))
        xst_opt_fixed_scaled = np.zeros((n_sweeps,len(Design_Vars)))
        parms_indx = DCA_problem['DesignVars']["DesignVars_keys"].index(Sweep_Parms)
        lb = bounds_ordered[0,parms_indx]
        ub = bounds_ordered[1,parms_indx]
        x0_parms = DCA_problem['DesignVars']["DesignVars_x0"][parms_indx]
        points = np.linspace(lb, ub, n_sweeps)
        for sweep_ind in range(n_sweeps):
            
            xp = points[sweep_ind]
                
            DCA_req_fixed = {}
            DCA_req_fixed['constraints'] = DCA_problem["constraints"]
            DCA_req_fixed['DesignParms'] = {}
            DCA_req_fixed["DesignParms"]["DesignParms_keys"] = DCA_problem["DesignParms"]["DesignParms_keys"] + [Sweep_Parms]
            DCA_req_fixed["DesignParms"]["DesignParms_values"] = np.concatenate((DCA_problem["DesignParms"]["DesignParms_values"], xp),None)
            DCA_req_fixed['DesignVars'] = {}
            DCA_req_fixed['DesignVars']["DesignVars_keys"] = Design_Vars
            DCA_req_fixed['objective'] = DCA_problem["objective"]
            
            xst_opt_fixed[sweep_ind,:], fst_opt_fixed[sweep_ind], exit_flag= opt_sweep(DCA_req_fixed,opt_filename,fname_ccdc_options)
            exitflag.append(exit_flag) 
        
        dx_scaled = (points-points[0])/(points[-1]-points[0])
        gobj_st[q] = np.linalg.norm(np.gradient(fst_opt_fixed/fx0,  dx_scaled.T))
        
        for p in range(len(Design_Vars)):
            
            x_lb = bounds_ordered[0,p]
            x_ub = bounds_ordered[1,p]
            xst_opt_fixed_scaled[:,p] = (xst_opt_fixed[:,p]-x_lb)/(x_ub-x_lb)
            grad_x_st[p] =  np.linalg.norm(np.gradient(xst_opt_fixed_scaled[:,p], dx_scaled.T))
            
            
        x_jac[q,:]=grad_x_st.T
        
    dc_jacobian = np.zeros((len(req_dv), len(req_dv)), dtype=float)
    obj_jacobian = gobj_st.T
    
    # Copy the original array to the new array, skipping the diagonal
    for i in range(len(req_dv)):
        dc_jacobian[i, :i] = x_jac[i, :i]
        dc_jacobian[i, i+1:] = x_jac[i, i:] 
        
    return dc_jacobian, obj_jacobian, exitflag
# dc_jacobian, obj_jacobian, exitflag_jac =  jacobian_app4()          

# non0_exitf_ind = np.nonzero(exitflag_jac)[0]
# non0_exitf = np.zeros((len(non0_exitf_ind),))
# for i in range(len(non0_exitf_ind)):
#     non0_exitf[i] = exitflag_jac[non0_exitf_ind[i]]
    
# %%    
#Labels for the Jacobian plots
def get_plot_labels(req_dv):
    labels = []
    for k in range(len(req_dv)):
        if req_dv[k]=='floating.jointdv_2':
            labels.append(r'$c_s$')
        if req_dv[k]=='floating.jointdv_1':
            labels.append(r'$z_{\text{frbrd}}$')
        if req_dv[k]=='floating.jointdv_0':
            labels.append(r'$z_{\text{keel}}$')
        if req_dv[k]=='floating.member_Y_pontoon_upper1:outer_diameter':
            labels.append(r'$D^{\text{up}}_{\text{pnt}}$')
        if req_dv[k]=='floating.member_Y_pontoon_lower1:outer_diameter':
            labels.append(r'$D^{\text{low}}_{\text{pnt}}$')
        if req_dv[k]=='floating.member_main_column:outer_diameter':
            labels.append(r'$D_{\text{main}}$')
        if req_dv[k]=='floating.member_column1:outer_diameter':
            labels.append(r'$c_d$')
        if req_dv[k]=='tune_rosco_ivc.ps_percent':
            labels.append(r'$p_s$')

    return labels
    
labels_l = get_plot_labels(req_dv)

def plot_dxdx(dxdx_mat,labels_l):
    # plot dx*/dx (for any case)
    
    sens_log = np.log10(abs(dxdx_mat))
    masked_data = np.ma.masked_where(np.isinf(sens_log), sens_log)
    
    fig, ax = plt.subplots()
    
    # Create a custom colormap with gray for NaNs
    cmap = plt.get_cmap('BuPu')
    
    cmap = cmap(np.linspace(0, 1, 256))
    cmap = ListedColormap(cmap)
    cmap.set_bad(color='darkgrey')  # Set the color for NaNs to gray
    
    
    im = ax.imshow(masked_data,cmap=cmap, vmin=masked_data.min(), vmax=masked_data.max())
    
    ax.set_xticks(np.arange(len(labels_l)), labels=labels_l, fontsize=11)
    ax.set_yticks(np.arange(len(labels_l)), labels=labels_l, fontsize=11)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    
    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    
    ax.set_xticks(np.arange(sens_log.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(sens_log.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    ax.set_xlabel('Perturbed design variable', fontsize=12)
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Optimized design variable', fontsize=12)
    
    cbar = ax.figure.colorbar(im, ax=ax,shrink=0.8, anchor=(0, 0))
    cbar.ax.set_ylabel(r'$\log _{10}\left(\left|\frac{\partial \hat{x}^*}{\partial \hat{x}}\right|\right)$', rotation=0, labelpad=28, loc = 'top', va="bottom", fontsize=13)
    plt.show()


def plot_dfdx(dfdx_mat,labels_l):
    # plot df*/dx 
    
    sens_log = np.log10(abs(dfdx_mat))
    masked_data = np.ma.masked_where(np.isinf(sens_log), sens_log)
    
    fig, ax = plt.subplots()
    ax.set_xlabel('Perturbed design variable', fontsize=12)
    ax.xaxis.set_label_position('top')
    
    # Create a custom colormap with gray for NaNs
    cmap = plt.get_cmap('BuPu')
    
    cmap = cmap(np.linspace(0, 1, 256))
    cmap = ListedColormap(cmap)
    cmap.set_bad(color='darkgrey')  # Set the color for NaNs to gray
    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(len(labels_l)), labels=labels_l, fontsize=11)
    
    
    ax.set_yticks(np.arange(sens_log.shape[0]+1)-.5, minor=True)  
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    
    if sens_log.ndim == 2:
              
        im = ax.imshow(masked_data,cmap=cmap, vmin=masked_data.min(), vmax=masked_data.max())   
        ax.set_xticks(np.arange(sens_log.shape[1]+1)-.5, minor=True)
    
        ax.set_yticks(np.arange(len(labels_l)), labels=labels_l, fontsize=11)   
        ax.set_ylabel('Optimized design variable', fontsize=12)   
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        cbar = ax.figure.colorbar(im, ax=ax,shrink=0.8, anchor=(0, 0))
        cbar.ax.set_ylabel(r'$\log _{10}\left(\left|\frac{\partial f^*}{\partial \hat{x}}\right|\right)$', rotation=0, labelpad=28, loc = 'top', va="bottom", fontsize=13)
    else:
        im = ax.imshow([masked_data],cmap=cmap, vmin=masked_data.min(), vmax=masked_data.max())  
        ax.set_xticks(np.arange(sens_log.shape[0])-.5, minor=True)
        ax.set_yticks([])
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3, axis='x')
        ax.tick_params(which="minor", bottom=False, left=False)
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, pad=0.1)
        cbar.ax.xaxis.label.set_size(12)
        cbar.set_label(r'$\log _{10}\left(\left|\frac{\partial f^*}{\partial \hat{x}}\right|\right)$', labelpad=0, loc='right', fontsize=13)
        cbar.ax.xaxis.set_label_coords(1.28, 1.45)
        
        # Apply color limits
        im.set_clim(masked_data.min(), masked_data.max())
    plt.show()


plot_dfdx(obj_jacobian,labels_l)
plot_dxdx(dc_jacobian,labels_l)