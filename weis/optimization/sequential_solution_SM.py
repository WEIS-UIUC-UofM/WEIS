#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 12:03:23 2025

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

####################################################################
#this info is in runWEIS
run_dir                = os.path.dirname( os.path.realpath(__file__) ) + os.sep
ex_dir = "/home/elenaf3/Desktop/DC_WEIS_Feb/WEIS/examples/15_RAFT_Studies/"
# sm_file = '/home/elenaf3/Desktop/DC_studies_code/WEIS/weis/optimization/log_opt.smt'
fname_wt_input         = os.path.join(ex_dir,"..","06_IEA-15-240-RWT", "IEA-15-240-RWT_VolturnUS-S.yaml")
fname_modeling_options = ex_dir + "modeling_options_level1_doe.yaml"
# fname_analysis_options = ex_dir + "analysis_options_level1_doe_verification.yaml"
fname_analysis_options = ex_dir + "analysis_options_level1.yaml"
overridden_values = {}

wt_initial = WindTurbineOntologyPythonWEIS(fname_wt_input, fname_modeling_options, fname_analysis_options)
wt_init, modeling_options, opt_options = wt_initial.get_input_data()
# folder_output = opt_options['general']['folder_output']

# make the folder_output relative to the input, if it's a relative path
analysis_input_dir = os.path.dirname(opt_options['fname_input_analysis'])
opt_options['general']['folder_output'] = os.path.join(analysis_input_dir,opt_options['general']['folder_output'])

folder_output = opt_options['general']['folder_output']
# opt_filename = os.path.join(folder_output, 'opt_output' + '.pkl')
opt_filename ='opt_output' + '.pkl'

####################################################################

sm_file = os.path.join(folder_output, os.path.splitext(opt_options['recorder']['file_name'])[0] + '.smt')
fname_ccdc_options = ex_dir + "ccdc_options.yaml"
ccdc_options = load_yaml(fname_ccdc_options) 
WTSMO = WindTurbineSMOpt(ccdc_options)
WTSMO.read_sm(sm_file, modeling_options)
sm2 = WTSMO.sm2
model = sm2[0]  
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
                    dvs_analysis_keys.append(f"floating.memgrp{idx}.outer_diameter_in")
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

# dvs_analysis_keys = get_design_variables()

dvs_analysis_keys = ['floating.member_Y_pontoon_lower1:outer_diameter','floating.member_Y_pontoon_upper1:outer_diameter','floating.member_column1:outer_diameter',
                     'floating.member_main_column:outer_diameter', 'floating.jointdv_0', 'floating.jointdv_1', 'floating.jointdv_2', 'tune_rosco_ivc.ps_percent']


# identify available inputs and outputs provided by the sm
opt_dv_sm = model['inputs']
opt_dv_key_sm = opt_dv_sm['keys']
bounds_sm = opt_dv_sm['bounds']
sm_to_analysis_idx = [opt_dv_key_sm.index(element) for element in dvs_analysis_keys if element in opt_dv_key_sm]

# opt_dv = opt_dv_sm[sm_to_analysis_idx]
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
    
    t0 = time.time()
    # run WEIS (only simulation) to get x0 values
    wt_opt_sim, modeling_options_sim, opt_options_sim = run_weis(fname_wt_input, fname_modeling_options_sim, fname_analysis_options_sim, overridden_values)
    
    t1 = time.time()
    total = t1-t0
    print(total)
    return wt_opt_sim

# get constraints and objective
constraints_key, constraints_bounds = get_constraints(opt_options)
if opt_options['merit_figure']=='platform_mass':
    objective_key =  f'{floating_solve_component}.platform_mass'
    # objective_key = 'floatingse.platform_mass'   # use floatingse even if we are using raft
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


def opt_all_dvs():  # run opt with all design variables
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


# Reorder opt_dv_key and DesignVars_x0 according to xst_optall
key_to_x0 = dict(zip(opt_dv_key, DesignVars_x0))
bounds_to_x0 = dict(zip(opt_dv_key, bounds.T))
opt_dv_key_ordered = keys_optall
DesignVars_x0_ordered = [key_to_x0[key] for key in keys_optall]
bounds_ordered = [bounds_to_x0[key] for key in keys_optall]

DCA_problem_ord = DCA_problem
# Update the dictionary with the reordered lists, bounds need to be updated too
DCA_problem_ord['DesignVars']['DesignVars_keys'] = opt_dv_key_ordered
DCA_problem_ord['DesignVars']['DesignVars_x0'] = DesignVars_x0_ordered
DCA_problem_ord['DesignVars']['bounds'] = bounds_ordered


# run optimizations with different dvs to get sensitivity matrix
req_dv = DCA_problem_ord["DesignVars"]["DesignVars_keys"]
# req_design_parms = DCA_valid["DesignParms"]["DesignParms_keys"]

q=4
Design_Vars = req_dv[q] 

#create optimization problem dictionary
DCA_1 = {}
DCA_1['constraints'] = DCA_problem_ord["constraints"]
DCA_1['DesignParms'] = {}
DCA_1["DesignParms"]["DesignParms_keys"] = []
DCA_1['DesignVars'] = {}
DCA_1['DesignVars']["DesignVars_keys"] = [Design_Vars]
DCA_1['objective'] = DCA_problem_ord["objective"]

# run opt 
WTSMO.get_opt_vars(DCA_1) 
WTSMO.run_optimization(DCA_1)
WTSMO.save_opt_output(opt_filename)   #need to change the file name
try:
    with open(opt_filename, 'rb') as f:
        opt_output1 = pkl.load(f)
except:
    print('Unable to read optimization result file: {:}.'.format(opt_filename))
    raise Exception('Unable to read optimization result file: {:}.'.format(opt_filename))

xst_opt_1 = opt_output1['dvs']['dv_values'][0]
fst_opt_1  = opt_output1['objective']['objective_values'][0]

Design_Vars2 = req_dv[:q] + req_dv[q+1:]
Design_Parms2 = req_dv[q]

#create optimization problem dictionary
DCA_2 = {}
DCA_2['constraints'] = DCA_problem_ord["constraints"]
DCA_2['DesignParms'] = {}
DCA_2["DesignParms"]["DesignParms_keys"] = [Design_Parms2]
DCA_2["DesignParms"]["DesignParms_values"] = [xst_opt_1]
DCA_2['DesignVars'] = {}
DCA_2['DesignVars']["DesignVars_keys"] = Design_Vars2
DCA_2['objective'] = DCA_problem_ord["objective"]
# run opt 
WTSMO.get_opt_vars(DCA_2) 
WTSMO.run_optimization(DCA_2)
WTSMO.save_opt_output(opt_filename)   #need to change the file name
try:
    with open(opt_filename, 'rb') as f:
        opt_output2 = pkl.load(f)
except:
    print('Unable to read optimization result file: {:}.'.format(opt_filename))
    raise Exception('Unable to read optimization result file: {:}.'.format(opt_filename))

xst_opt_2 = opt_output2['dvs']['dv_values']
fst_opt_2  = opt_output2['objective']['objective_values'][0]
