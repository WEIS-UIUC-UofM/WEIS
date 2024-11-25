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


####################################################################
#this info is in runWEIS
run_dir                = os.path.dirname( os.path.realpath(__file__) ) + os.sep
ex_dir = "/home/elenaf3/Desktop/DC_studies_code/WEIS/examples/15_RAFT_Studies/"
# sm_file = '/home/elenaf3/Desktop/DC_studies_code/WEIS/weis/optimization/log_opt.smt'
fname_wt_input         = os.path.join(ex_dir,"..","06_IEA-15-240-RWT", "IEA-15-240-RWT_VolturnUS-S.yaml")
fname_modeling_options = ex_dir + "modeling_options_level1_doe.yaml"
fname_analysis_options = ex_dir + "analysis_options_level1_doe_verification.yaml"
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
 
# identify available inputs and outputs provided by the sm
opt_dv = model['inputs']
opt_dv_key = opt_dv['keys']
bounds = opt_dv['bounds']

# get some needed modeling info
if modeling_options['Level3']['flag']:
    floating_solve_component = 'aeroelastic'
elif modeling_options['Level1']['flag']:
    floating_solve_component = 'raft'
else:
    floating_solve_component = 'floatingse'

# identify constraints 
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
analysis_dict['driver']['surrogate_based_optimization']['flag'] = False

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

# %%
# def dict_lookup(dictionary, target_key):
    
#     for key, value in dictionary.items():
#         if key == target_key:
#             return value
#         elif isinstance(value, dict):
#             result = dict_lookup(value, target_key)
#             if result is not None:
#                 return result
#     return None

# def find_x0(target_key):
#     entry = dict_lookup(modeling_options, target_key)
#     if entry is None:
#         entry = dict_lookup(wt_init, target_key)
#     return entry

# for k in range(len(opt_dv_key)):
#     var = opt_dv_key[k]
#     if var=='floating.jointdv_0':
#         x0 = find_x0()
#     elif var=='floating.jointdv_1':
#         x0 = find_x0()
#     elif var=='floating.jointdv_2':
#         x0 = find_x0()
#     elif var=='floating.memgrp0.outer_diameter_in':
#         x0 = find_x0()
#     elif var=='floating.memgrp1.outer_diameter_in':
#         x0 = find_x0()
#     elif var=='floating.memgrp2.outer_diameter_in':
#         x0 = find_x0()
#     elif var=='floating.memgrp3.outer_diameter_in':
#         x0 = find_x0()
#     elif var=='tune_rosco_ivc.ps_percent':
#         x0 = find_x0('ps_percent')
#     elif var=='tune_rosco_ivc.omega_pc':
#         x0 = find_x0('omega_pc')
#     elif var=='tune_rosco_ivc.ptfm_freq':
#         x0 = find_x0('ptfm_freq')
#     elif var=='tune_rosco_ivc.zeta_pc':
#         x0 = find_x0('zeta_pc')
#     elif var=='configuration.rotor_diameter_user':
#         x0 = find_x0('rotor_diameter')
#     elif var=='floating.member_main_column:outer_diameter':
#         x0 = find_x0()
#     elif var=='floating.member_column1:outer_diameter':
#         x0 = find_x0()
#     elif var=='floating.member_column2:outer_diameter':
#         x0 = find_x0()
#     elif var=='floating.member_column3:outer_diameter':
#         x0 = find_x0()
#     elif var=='floating.member_Y_pontoon_upper1:outer_diameter':
#         x0 = find_x0()
#     elif var=='floating.member_Y_pontoon_upper2:outer_diameter':
#         x0 = find_x0()
#     elif var=='floating.member_Y_pontoon_upper3:outer_diameter':
#         x0 = find_x0()
#     elif var=='floating.member_Y_pontoon_lower1:outer_diameter':
#         x0 = find_x0()
#     elif var=='floating.member_Y_pontoon_lower2:outer_diameter':
#         x0 = find_x0()
#     elif var=='floating.member_Y_pontoon_lower3:outer_diameter':
#         x0 = find_x0()
#     elif var=='tune_rosco_ivc.Kp_float':
#         x0 = find_x0('Kp_float')

# design_vars_x0 = DCA_req['DesignVars']['DesignVars_x0']
# design_parms_values = DCA_req['DesignParms']['DesignParms_values']
# %%

# get constraints and objective
constraints_key, constraints_bounds = get_constraints(opt_options)
if opt_options['merit_figure']=='platform_mass':
    # objective_key =  f'{floating_solve_component}.platform_mass'
    objective_key = 'floatingse.platform_mass'
else:
    objective_key = opt_options['merit_figure']

DesignVars_x0 = np.zeros((len(opt_dv_key),))
for k in range(len(opt_dv_key)):
    var = opt_dv_key[k]
    x0 = wt_opt_sim.get_val(var)
    DesignVars_x0[k] = x0[0]
    
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


avail_output_keys = WTSMO.avail_output_keys
avail_input_keys = WTSMO.avail_input_keys


# run opt with all design variables
WTSMO.get_opt_vars(DCA_problem) 
WTSMO.run_optimization(DCA_problem)
WTSMO.save_opt_output(opt_filename)

#save optimization results
try:
    with open(opt_filename, 'rb') as f:
        opt_output_optall = pkl.load(f)
except:
    print('Unable to read optimization result file: {:}.'.format(opt_filename))
    raise Exception('Unable to read optimization result file: {:}.'.format(opt_filename))

xst_optall = opt_output_optall['dvs']['dv_values']
fst_optall = opt_output_optall['objective']['objective_values']
keys_optall = opt_output_optall['dvs']['dv_keys']

key_to_x0 = dict(zip(opt_dv_key, DesignVars_x0))

# Reorder opt_dv_key and DesignVars_x0 according to xst_optall
opt_dv_key_ordered = keys_optall
DesignVars_x0_ordered = [key_to_x0[key] for key in keys_optall]

# Update the dictionary with the reordered lists
DCA_problem['DesignVars']['DesignVars_keys'] = opt_dv_key_ordered
DCA_problem['DesignVars']['DesignVars_x0'] = DesignVars_x0_ordered


# run optimizations with different dvs to get sensitivity matrix
req_dv = DCA_problem["DesignVars"]["DesignVars_keys"]
# req_design_parms = DCA_valid["DesignParms"]["DesignParms_keys"]

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

# ## #################################################################  
# # Approach 1 


# n_sweeps = 2
# percent = 0.01
# # xst_opt_fixed = np.zexst_opt_fixedros((len(req_dv)-1,n_sweeps))
# # xst_opt_fixed = np.zeros((n_sweeps,))

# fst_opt_fixed = np.zeros((n_sweeps,))

# def gen_sweep_points(x0, c, n_sweeps, lb, ub):
#     start = x0
#     end = x0 + c*(ub-lb)
#     array = np.linspace(start, end, n_sweeps)
#     return array

# x_jac = np.zeros((len(req_dv),len(req_dv)-1))
# exitflag = [] 
# constraint_violationv = []   
# first_order_optimalityv = []  
# gobj_st = np.zeros((len(req_dv),len(req_dv)-1))
# for q in range(len(req_dv)):

#     Design_Vars = req_dv[q]          
#     AllDesign_Parms = req_dv[:q] + req_dv[q+1:]
#     AllDesign_Parms_x0 = DCA_problem['DesignVars']['DesignVars_x0'][:q] + DCA_problem['DesignVars']['DesignVars_x0'][q+1:] 
#     xst_opt_fixed = np.zeros((n_sweeps,len(AllDesign_Parms)))
#     grad_x_st = np.zeros((len(AllDesign_Parms),))
    
    
#     for p in range(len(AllDesign_Parms)):
#         Sweep_Parms = AllDesign_Parms[p]
#         Design_Parms = AllDesign_Parms[:p] + AllDesign_Parms[p+1:]
#         Design_Parms_x0 = AllDesign_Parms_x0[:p] + AllDesign_Parms_x0[p+1:]
        
#         for sweep_ind in range(n_sweeps):
            
#             parms_indx = DCA_problem['DesignVars']["DesignVars_keys"].index(Sweep_Parms)
#             lb = bounds[0,parms_indx]
#             ub = bounds[1,parms_indx]
#             x0_parms = DCA_problem['DesignVars']["DesignVars_x0"][parms_indx]
#             points = gen_sweep_points(x0_parms, percent, n_sweeps, lb, ub)
#             xp = points[sweep_ind]
                
#             DCA_req_fixed = {}
#             DCA_req_fixed['constraints'] = DCA_problem["constraints"]
#             DCA_req_fixed['DesignParms'] = {}
#             DCA_req_fixed["DesignParms"]["DesignParms_keys"] = DCA_problem["DesignParms"]["DesignParms_keys"] + Design_Parms + [Sweep_Parms]
#             DCA_req_fixed["DesignParms"]["DesignParms_values"] = np.concatenate((DCA_problem["DesignParms"]["DesignParms_values"],Design_Parms_x0, xp),None)
#             DCA_req_fixed['DesignVars'] = {}
#             DCA_req_fixed['DesignVars']["DesignVars_keys"] = [Design_Vars]
#             DCA_req_fixed['objective'] = DCA_problem["objective"]
               
#             # run opt with all except design variables
#             WTSMO.get_opt_vars(DCA_req_fixed) 
#             WTSMO.run_optimization(DCA_req_fixed)
#             WTSMO.save_opt_output(opt_filename)
#             try:
#                 with open(opt_filename, 'rb') as f:
#                     opt_output = pkl.load(f)
#             except:
#                 print('Unable to read optimization result file: {:}.'.format(opt_filename))
#                 raise Exception('Unable to read optimization result file: {:}.'.format(opt_filename))
            
#             xst_opt_fixed[sweep_ind,p] = opt_output['dvs']['dv_values']
#             fst_opt_fixed[sweep_ind]  = opt_output['objective']['objective_values']
#             # exitflag.append(prob.driver.pyopt_solution.optInform['value'])

    
#     x_lb = bounds[0,q]
#     x_ub = bounds[1,q]
#     xst_opt_fixed_scaled = (xst_opt_fixed-x_lb)/(x_ub-x_lb)
#     dx_scaled = (points-points[0])/(points[-1]-points[0])
#     gobj_st[q,p] = np.linalg.norm(np.gradient(fst_opt_fixed,  dx_scaled.T))
    
#     for p in range(len(AllDesign_Parms)):
#         grad_x_st[p] =  np.linalg.norm(np.gradient(xst_opt_fixed_scaled[:,p], dx_scaled.T))
        
        
#     x_jac[q,:]=grad_x_st.T


# dc_jacobian = np.zeros((len(req_dv), len(req_dv)), dtype=float)
# obj_jacobian = np.zeros((len(req_dv), len(req_dv)), dtype=float)

# # Copy the original array to the new array, skipping the diagonal
# for i in range(len(req_dv)):
#     dc_jacobian[i, :i] = x_jac[i, :i]
#     dc_jacobian[i, i+1:] = x_jac[i, i:]
#     obj_jacobian[i, :i] = gobj_st[i, :i]
#     obj_jacobian[i, i+1:] = gobj_st[i, i:]

    

# ## #################################################################  
# # Approach 2

# n_sweeps = 100
# percent = 0.01
# # xst_opt_fixed = np.zexst_opt_fixedros((len(req_dv)-1,n_sweeps))
# # xst_opt_fixed = np.zeros((n_sweeps,))

# fst_opt_fixed = np.zeros((n_sweeps,))

# x_jac = np.zeros((len(req_dv),len(req_dv)-1))
# exitflag = [] 
# constraint_violationv = []   
# first_order_optimalityv = []  
# gobj_st = np.zeros((len(req_dv),len(req_dv)-1))
# for q in range(len(req_dv)):

#     Design_Vars = req_dv[q]          
#     AllDesign_Parms = req_dv[:q] + req_dv[q+1:]
#     AllDesign_Parms_x0 = DCA_problem['DesignVars']['DesignVars_x0'][:q] + DCA_problem['DesignVars']['DesignVars_x0'][q+1:] 
#     xst_opt_fixed = np.zeros((n_sweeps,len(AllDesign_Parms)))
#     grad_x_st = np.zeros((len(AllDesign_Parms),))
    
    
#     for p in range(len(AllDesign_Parms)):
#         Sweep_Parms = AllDesign_Parms[p]
#         Design_Parms = AllDesign_Parms[:p] + AllDesign_Parms[p+1:]
#         Design_Parms_x0 = AllDesign_Parms_x0[:p] + AllDesign_Parms_x0[p+1:]
        
#         for sweep_ind in range(n_sweeps):
            
#             parms_indx = DCA_problem['DesignVars']["DesignVars_keys"].index(Sweep_Parms)
#             lb = bounds[0,parms_indx]
#             ub = bounds[1,parms_indx]
#             x0_parms = DCA_problem['DesignVars']["DesignVars_x0"][parms_indx]
            
#             points = np.linspace(lb, ub, n_sweeps)
#             xp = points[sweep_ind]
                
#             DCA_req_fixed = {}
#             DCA_req_fixed['constraints'] = DCA_problem["constraints"]
#             DCA_req_fixed['DesignParms'] = {}
#             DCA_req_fixed["DesignParms"]["DesignParms_keys"] = DCA_problem["DesignParms"]["DesignParms_keys"] + Design_Parms + [Sweep_Parms]
#             DCA_req_fixed["DesignParms"]["DesignParms_values"] = np.concatenate((DCA_problem["DesignParms"]["DesignParms_values"],Design_Parms_x0, xp),None)
#             DCA_req_fixed['DesignVars'] = {}
#             DCA_req_fixed['DesignVars']["DesignVars_keys"] = [Design_Vars]
#             DCA_req_fixed['objective'] = DCA_problem["objective"]
               
#             # run opt with all except design variables
#             WTSMO.get_opt_vars(DCA_req_fixed) 
#             WTSMO.run_optimization(DCA_req_fixed)
#             WTSMO.save_opt_output(opt_filename)
#             try:
#                 with open(opt_filename, 'rb') as f:
#                     opt_output = pkl.load(f)
#             except:
#                 print('Unable to read optimization result file: {:}.'.format(opt_filename))
#                 raise Exception('Unable to read optimization result file: {:}.'.format(opt_filename))
            
#             xst_opt_fixed[sweep_ind,p] = opt_output['dvs']['dv_values']
#             fst_opt_fixed[sweep_ind]  = opt_output['objective']['objective_values']
#             # exitflag.append(prob.driver.pyopt_solution.optInform['value'])

    
#     x_lb = bounds[0,q]
#     x_ub = bounds[1,q]
#     xst_opt_fixed_scaled = (xst_opt_fixed-x_lb)/(x_ub-x_lb)
#     dx_scaled = (points-points[0])/(points[-1]-points[0])
#     gobj_st[q,p] = np.linalg.norm(np.gradient(fst_opt_fixed,  dx_scaled.T))
    
#     for p in range(len(AllDesign_Parms)):
#         grad_x_st[p] =  np.linalg.norm(np.gradient(xst_opt_fixed_scaled[:,p], dx_scaled.T))
        
        
#     x_jac[q,:]=grad_x_st.T


# dc_jacobian2 = np.zeros((len(req_dv), len(req_dv)), dtype=float)
# obj_jacobian2 = np.zeros((len(req_dv), len(req_dv)), dtype=float)

# # Copy the original array to the new array, skipping the diagonal
# for i in range(len(req_dv)):
#     dc_jacobian2[i, :i] = x_jac[i, :i]
#     dc_jacobian2[i, i+1:] = x_jac[i, i:]
#     obj_jacobian2[i, :i] = gobj_st[i, :i]
#     obj_jacobian2[i, i+1:] = gobj_st[i, i:]


## #################################################################  
# Approach 3 

n_sweeps = 2
percent = 0.01

fst_opt_fixed = np.zeros((n_sweeps,))

def gen_sweep_points(x0, c, n_sweeps, lb, ub):
    start = x0
    end = x0 + c*(ub-lb)
    array = np.linspace(start, end, n_sweeps)
    return array

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
    
        
    for sweep_ind in range(n_sweeps):
        
        parms_indx = DCA_problem['DesignVars']["DesignVars_keys"].index(Sweep_Parms)
        lb = bounds[0,parms_indx]
        ub = bounds[1,parms_indx]
        x0_parms = DCA_problem['DesignVars']["DesignVars_x0"][parms_indx]
        points = gen_sweep_points(x0_parms, percent, n_sweeps, lb, ub)
        xp = points[sweep_ind]
            
        DCA_req_fixed = {}
        DCA_req_fixed['constraints'] = DCA_problem["constraints"]
        DCA_req_fixed['DesignParms'] = {}
        DCA_req_fixed["DesignParms"]["DesignParms_keys"] = DCA_problem["DesignParms"]["DesignParms_keys"] + [Sweep_Parms]
        DCA_req_fixed["DesignParms"]["DesignParms_values"] = np.concatenate((DCA_problem["DesignParms"]["DesignParms_values"], xp),None)
        DCA_req_fixed['DesignVars'] = {}
        DCA_req_fixed['DesignVars']["DesignVars_keys"] = Design_Vars
        DCA_req_fixed['objective'] = DCA_problem["objective"]
           
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
        
        xst_opt_fixed[sweep_ind,:] = opt_output['dvs']['dv_values']
        fst_opt_fixed[sweep_ind]  = opt_output['objective']['objective_values']
        # exitflag.append(prob.driver.pyopt_solution.optInform['value'])

    
    x_lb = bounds[0,q]
    x_ub = bounds[1,q]
    xst_opt_fixed_scaled = (xst_opt_fixed-x_lb)/(x_ub-x_lb)
    dx_scaled = (points-points[0])/(points[-1]-points[0])
    gobj_st[q] = np.linalg.norm(np.gradient(fst_opt_fixed,  dx_scaled.T))
    
    for p in range(len(Design_Vars)):
        grad_x_st[p] =  np.linalg.norm(np.gradient(xst_opt_fixed_scaled[:,p], dx_scaled.T))
        
        
    x_jac[q,:]=grad_x_st.T


dc_jacobian3 = np.zeros((len(req_dv), len(req_dv)), dtype=float)
obj_jacobian3 = gobj_st.T

# Copy the original array to the new array, skipping the diagonal
for i in range(len(req_dv)):
    dc_jacobian3[i, :i] = x_jac[i, :i]
    dc_jacobian3[i, i+1:] = x_jac[i, i:]

## #################################################################  
# Approach 4 

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
    
        
    for sweep_ind in range(n_sweeps):
        
        parms_indx = DCA_problem['DesignVars']["DesignVars_keys"].index(Sweep_Parms)
        lb = bounds[0,parms_indx]
        ub = bounds[1,parms_indx]
        x0_parms = DCA_problem['DesignVars']["DesignVars_x0"][parms_indx]
        points = np.linspace(lb, ub, n_sweeps)
        xp = points[sweep_ind]
            
        DCA_req_fixed = {}
        DCA_req_fixed['constraints'] = DCA_problem["constraints"]
        DCA_req_fixed['DesignParms'] = {}
        DCA_req_fixed["DesignParms"]["DesignParms_keys"] = DCA_problem["DesignParms"]["DesignParms_keys"] + [Sweep_Parms]
        DCA_req_fixed["DesignParms"]["DesignParms_values"] = np.concatenate((DCA_problem["DesignParms"]["DesignParms_values"], xp),None)
        DCA_req_fixed['DesignVars'] = {}
        DCA_req_fixed['DesignVars']["DesignVars_keys"] = Design_Vars
        DCA_req_fixed['objective'] = DCA_problem["objective"]
           
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
        
        xst_opt_fixed[sweep_ind,:] = opt_output['dvs']['dv_values']
        fst_opt_fixed[sweep_ind]  = opt_output['objective']['objective_values']
        # exitflag.append(prob.driver.pyopt_solution.optInform['value'])

    
    x_lb = bounds[0,q]
    x_ub = bounds[1,q]
    xst_opt_fixed_scaled = (xst_opt_fixed-x_lb)/(x_ub-x_lb)
    dx_scaled = (points-points[0])/(points[-1]-points[0])
    gobj_st[q] = np.linalg.norm(np.gradient(fst_opt_fixed,  dx_scaled.T))
    
    for p in range(len(Design_Vars)):
        grad_x_st[p] =  np.linalg.norm(np.gradient(xst_opt_fixed_scaled[:,p], dx_scaled.T))
        
        
    x_jac[q,:]=grad_x_st.T


dc_jacobian4 = np.zeros((len(req_dv), len(req_dv)), dtype=float)
obj_jacobian4 = gobj_st.T

# Copy the original array to the new array, skipping the diagonal
for i in range(len(req_dv)):
    dc_jacobian4[i, :i] = x_jac[i, :i]
    dc_jacobian4[i, i+1:] = x_jac[i, i:] 