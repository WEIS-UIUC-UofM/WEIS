import numpy as np
import os
import time
import re
import pickle as pkl
import openmdao.api as om
# from wisdem.commonse.mpi_tools import MPI
from smt.surrogate_models import KRG
from weis.glue_code.gc_LoadInputs     import WindTurbineOntologyPythonWEIS


class WindTurbineSMOpt():
    

    def __init__(self,ccdc_options):
        self._sm_loaded = False
        self._opt_run = False
        self.opt_options = ccdc_options
        self.scipy_methods = [
                    "SLSQP",
                    "Nelder-Mead",
                    "COBYLA",
                ]
        self.pyoptsparse_methods = [
                    "SNOPT",
                    "CONMIN",
                    "NSGA2",
                ]
        self.avail_input_keys = []
        self.opt_output_key = []    
        self.avail_outputs_keys_compact = []
        self.sm2 = []
    
    def read_sm(self, sm_file, modeling_options):
        # read sm
        try:
            sm2 = None
            with open(sm_file, "rb") as f:
               sm2 = pkl.load(f)
        
        except:
            print('Unable to read surrogate model file: {:}.'.format(sm_file))
            raise Exception('Unable to read surrogate model file: {:}.'.format(sm_file))
        
        model = sm2[0]        
        
        # identify available inputs and outputs provided by the sm
        avail_inputs = model['inputs']
        avail_outputs_keys_str = list(len(sm2)*[None])
        
        avail_outputs_keys_compact = []
        for sm_ind in range(len(sm2)):
            avail_outputs_keys_str[sm_ind]=sm2[sm_ind]['outputs']['keys']
            avail_outputs_keys_compact.append(sm2[sm_ind]['outputs']['keys'].split('+'))
            
        avail_outputs_str_keys = '+'.join(avail_outputs_keys_str)
        avail_output_keys = avail_outputs_str_keys.split('+')
        avail_input_keys = avail_inputs['keys']
               
        # Store data
        self.avail_inputs = avail_inputs
        self.avail_output_keys = avail_output_keys
        self.avail_input_keys = avail_input_keys
        self.sm2 = sm2
        self.avail_outputs_keys_compact = avail_outputs_keys_compact
        self._sm_loaded = True



    def get_opt_vars(self, DCA_req):
        req_dv = DCA_req["DesignVars"]["DesignVars_keys"]
        req_design_parms = DCA_req["DesignParms"]["DesignParms_keys"]
        req_objective = [DCA_req["objective"]]
        req_constraints = DCA_req["constraints"]["constraints_keys"]
        req_outputs = req_objective + req_constraints
        req_inputs = req_dv + req_design_parms
        
        avail_output_keys = self.avail_output_keys
        avail_input_keys = self.avail_input_keys
        opt_output_key = list(set(req_outputs).intersection(avail_output_keys))
        opt_output_key_wo_dot = [sub.replace('.', '_') for sub in opt_output_key]
        objective_key = list(set(req_objective).intersection(avail_output_keys))
        
        if len(req_objective)==0:
            raise Exception('Objective not available or missing, select a supported objective')
        if len(req_objective)>1:
            raise Exception('multiobjective optimization is not supported')
            
        objective_key_wo_dot = [sub.replace('.', '_') for sub in objective_key]
        constraints_key = list(set(req_constraints).intersection(avail_output_keys))
        constraints_key_wo_dot = [sub.replace('.', '_') for sub in constraints_key]
        opt_dv_key = list(set(req_dv).intersection(avail_input_keys))
        opt_dv_key_wo_dot = [sub.replace('.', '_') for sub in opt_dv_key]
        opt_params_key = list(set(req_design_parms).intersection(avail_input_keys))
        opt_input_key = list(set(req_inputs).intersection(avail_input_keys))
        opt_input_key_wo_dot = [sub.replace('.', '_') for sub in opt_input_key]
        
        self.opt_dv_key_wo_dot = opt_dv_key_wo_dot
        self.opt_dv_key = opt_dv_key
        self.constraints_key_wo_dot = constraints_key_wo_dot
        self.constraints_key = constraints_key
        self.objective_key_wo_dot = objective_key_wo_dot
        self.opt_output_key = opt_output_key
        self.opt_output_key_wo_dot = opt_output_key_wo_dot
        self.opt_params_key = opt_params_key
        self.objective_key = objective_key
        self.opt_input_key_wo_dot = opt_input_key_wo_dot

    
    # from WISDEM/wisdem/glue_code/gc_PoseOptimization.py
    def _set_optimizer_properties(self, prob, options_keys=[], opt_settings_keys=[], mapped_keys={}):
    
            """
            Set the optimizer properties, both the `driver.options` and
            `driver.opt_settings`. See OpenMDAO documentation on drivers
            to determine which settings are set by either options or
            opt_settings.
    
            Parameters
            ----------
            wt_opt : OpenMDAO problem object
                The wind turbine problem object.
            options_keys : list
                List of keys for driver options to be set.
            opt_settings_keys: list
                List of keys for driver opt_settings to be set.
            mapped_keys: dict
                Key pairs where the yaml name differs from what's expected
                by the driver. Specifically, the key is what's given in the yaml
                and the value is what's expected by the driver.
    
            Returns
            -------
            wt_opt : OpenMDAO problem object
                The updated wind turbine problem object with driver settings applied.
            """
    
            smb_opt_options = self.opt_options["SMB_optimization"]
    
            # Loop through all of the options provided and set them in the OM driver object
            for key in options_keys:
                if key in smb_opt_options:
                    if key in mapped_keys:
                        prob.driver.options[mapped_keys[key]] = smb_opt_options[key]
                    else:
                        prob.driver.options[key] = smb_opt_options[key]
    
            # Loop through all of the opt_settings provided and set them in the OM driver object
            for key in opt_settings_keys:
                if key in smb_opt_options:
                    if key in mapped_keys:
                        prob.driver.opt_settings[mapped_keys[key]] = smb_opt_options[key]
                    else:
                        prob.driver.opt_settings[key] = smb_opt_options[key]
    
            return prob


    def create_problem(self):
        
        prob = om.Problem()
        self.prob = prob
        model = prob.model
        model.add_subsystem('model', subsys=SM_Comp(self), promotes_inputs=self.opt_input_key_wo_dot, promotes_outputs=self.opt_output_key_wo_dot)

        self.model = model
        opt_options = self.opt_options
        solver = opt_options['SMB_optimization']['solver']
        
        # from WISDEM/wisdem/glue_code/gc_PoseOptimization.py
        # Set optimization solver and options. First, Scipy's SLSQP and COBYLA
        # if solver in self.scipy_methods:
        if solver in self.scipy_methods:

            prob.driver = om.ScipyOptimizeDriver()
            prob.driver.options["optimizer"] = solver
            
            options_keys = ["tol", "max_iter", "disp"]
            opt_settings_keys = ["rhobeg", "catol", "adaptive"]
            mapped_keys = {"max_iter": "maxiter"}
            prob = self._set_optimizer_properties(prob, options_keys, opt_settings_keys, mapped_keys)

        # The next two optimization methods require pyOptSparse.
        # elif solver in self.pyoptsparse_methods:
        elif solver in self.pyoptsparse_methods:
            try:
                from openmdao.api import pyOptSparseDriver
            except:
                raise ImportError(
                    f"You requested the optimization solver {opt_options['solver']}, but you have not installed pyOptSparse. Please do so and rerun."
                )
            prob.driver = pyOptSparseDriver()
            try:
                prob.driver.options["optimizer"] = solver
            except:
                raise ImportError(
                    f"You requested the optimization solver {solver}, but you have not installed it within pyOptSparse. Please build {solver} and rerun."
                )

            # Most of the pyOptSparse options have special syntax when setting them,
            # so here we set them by hand instead of using `_set_optimizer_properties` for SNOPT and CONMIN.
            if solver == "CONMIN":
                prob.driver.opt_settings["ITMAX"] = opt_options["max_iter"]

            if solver == "NSGA2":
                opt_settings_keys = [
                    "PopSize",
                    "maxGen",
                    "pCross_real",
                    "pMut_real",
                    "eta_c",
                    "eta_m",
                    "pCross_bin",
                    "pMut_bin",
                    "PrintOut",
                    "seed",
                    "xinit",
                ]
                prob = self._set_optimizer_properties(prob, opt_settings_keys=opt_settings_keys)

            elif solver == "SNOPT":
                prob.driver.opt_settings["Major optimality tolerance"] = float(opt_options["tol"])
                prob.driver.opt_settings["Major iterations limit"] = int(opt_options["max_major_iter"])
                prob.driver.opt_settings["Iterations limit"] = int(opt_options["max_minor_iter"])
                prob.driver.opt_settings["Major feasibility tolerance"] = float(opt_options["tol"])
                if "time_limit" in opt_options:
                    prob.driver.opt_settings["Time limit"] = int(opt_options["time_limit"])
                folder_output = opt_options["general"]["folder_output"]
                prob.driver.opt_settings["Summary file"] = os.path.join(folder_output, "SNOPT_Summary_file.txt")
                prob.driver.opt_settings["Print file"] = os.path.join(folder_output, "SNOPT_Print_file.txt")
                if "hist_file_name" in opt_options:
                    prob.driver.hist_file = opt_options["hist_file_name"]
                if "verify_level" in opt_options:
                    prob.driver.opt_settings["Verify level"] = opt_options["verify_level"]
                else:
                    prob.driver.opt_settings["Verify level"] = -1
            if "hotstart_file" in opt_options:
                prob.driver.hotstart_file = opt_options["hotstart_file"]
        
 
    def get_bounds(self, avail_inputs,dv_key):
        dv_indx = avail_inputs['keys'].index(dv_key)
        lb = avail_inputs['bounds'][0,dv_indx]
        ub = avail_inputs['bounds'][1,dv_indx]
        return lb, ub

    def add_design_vars(self):
        
        opt_dv_key = self.opt_dv_key
        avail_inputs = self.avail_inputs
        model = self.model
        # add dv
        for k in range(len(opt_dv_key)):           # change x0 to default x0
            
            if opt_dv_key[k] == 'floating.jointdv_0':
                lb, ub = self.get_bounds(avail_inputs,'floating.jointdv_0')
                model.add_design_var('floating_jointdv_0', lower=lb, upper=ub)
                model.set_input_defaults('floating_jointdv_0', val=(lb+ub)/2)
                
            elif opt_dv_key[k]  == "floating.jointdv_1":
                lb, ub = self.get_bounds(avail_inputs,'floating.jointdv_1')
                model.add_design_var('floating_jointdv_1', lower=lb, upper=ub)
                model.set_input_defaults('floating_jointdv_1', val=(lb+ub)/2)
                
            elif opt_dv_key[k]  == "floating.jointdv_2":
                lb, ub = self.get_bounds(avail_inputs,'floating.jointdv_2')
                model.add_design_var('floating_jointdv_2', lower=lb, upper=ub)
                model.set_input_defaults('floating_jointdv_2', val=(lb+ub)/2)
                
            elif opt_dv_key[k]  == "floating.memgrp0.outer_diameter_in":
                lb, ub = self.get_bounds(avail_inputs,'floating.memgrp0.outer_diameter_in')
                model.add_design_var('floating_memgrp0_outer_diameter_in', lower=lb, upper=ub)
                model.set_input_defaults('floating_memgrp0_outer_diameter_in', val=(lb+ub)/2)
                
            elif opt_dv_key[k]  == "floating.memgrp1.outer_diameter_in":
                lb, ub = self.get_bounds(avail_inputs,'floating.memgrp1.outer_diameter_in')
                model.add_design_var('floating_memgrp1_outer_diameter_in', lower=lb, upper=ub)
                model.set_input_defaults('floating_memgrp1_outer_diameter_in', val=(lb+ub)/2)
                
            elif opt_dv_key[k]  == "floating.memgrp2.outer_diameter_in":
                lb, ub = self.get_bounds(avail_inputs,'floating.memgrp2.outer_diameter_in')
                model.add_design_var('floating_memgrp2_outer_diameter_in', lower=lb, upper=ub)
                model.set_input_defaults('floating_memgrp2_outer_diameter_in', val=(lb+ub)/2)
                
            elif opt_dv_key[k]  == "floating.memgrp3.outer_diameter_in":
                lb, ub = self.get_bounds(avail_inputs,'floating.memgrp3.outer_diameter_in')
                model.add_design_var('floating_memgrp3_outer_diameter_in', lower=lb, upper=ub)
                model.set_input_defaults('floating_memgrp3_outer_diameter_in', val=(lb+ub)/2)
                
            elif opt_dv_key[k]  == "tune_rosco_ivc.ps_percent":
                lb, ub = self.get_bounds(avail_inputs,'tune_rosco_ivc.ps_percent')
                model.add_design_var('tune_rosco_ivc_ps_percent', lower=lb, upper=ub)
                model.set_input_defaults('tune_rosco_ivc_ps_percent', val=(lb+ub)/2)
                
            elif opt_dv_key[k]  == "tune_rosco_ivc.omega_pc":
                lb, ub = self.get_bounds(avail_inputs,'tune_rosco_ivc.omega_pc')
                model.add_design_var('tune_rosco_ivc_omega_pc', lower=lb, upper=ub)
                model.set_input_defaults('tune_rosco_ivc_omega_pc', val=(lb+ub)/2)
                
            elif opt_dv_key[k]  == "tune_rosco_ivc.ptfm_freq":
                lb, ub = self.get_bounds(avail_inputs,'tune_rosco_ivc.ptfm_freq')
                model.add_design_var('tune_rosco_ivc_ptfm_freq', lower=lb, upper=ub)
                model.set_input_defaults('tune_rosco_ivc_ptfm_freq', val=(lb+ub)/2)
                
            elif opt_dv_key[k]  == "tune_rosco_ivc.zeta_pc":
                lb, ub = self.get_bounds(avail_inputs,'tune_rosco_ivc.zeta_pc')
                model.add_design_var('tune_rosco_ivc_zeta_pc', lower=lb, upper=ub)
                model.set_input_defaults('tune_rosco_ivc_zeta_pc', val=(lb+ub)/2)
                
            elif opt_dv_key[k]  == "configuration.rotor_diameter_user":
                lb, ub = self.get_bounds(avail_inputs,'configuration.rotor_diameter_user')
                model.add_design_var('configuration_rotor_diameter_user', lower=lb, upper=ub)
                model.set_input_defaults('configuration_rotor_diameter_user', val=(lb+ub)/2)
            
            elif opt_dv_key[k]  == "floating.member_main_column:outer_diameter":
                lb, ub = self.get_bounds(avail_inputs,'floating.member_main_column:outer_diameter')
                model.add_design_var('floating_member_main_column:outer_diameter', lower=lb, upper=ub)
                model.set_input_defaults('floating_member_main_column:outer_diameter', val=(lb+ub)/2)
                
            elif opt_dv_key[k]  == "floating.member_column1:outer_diameter":
                lb, ub = self.get_bounds(avail_inputs,'floating.member_column1:outer_diameter')
                model.add_design_var('floating_member_column1:outer_diameter', lower=lb, upper=ub)
                model.set_input_defaults('floating_member_column1:outer_diameter', val=(lb+ub)/2)
                
            elif opt_dv_key[k]  == "floating.member_column2:outer_diameter":
                lb, ub = self.get_bounds(avail_inputs,'floating.member_column2:outer_diameter')
                model.add_design_var('floating_member_column2:outer_diameter', lower=lb, upper=ub)
                model.set_input_defaults('floating_member_column2:outer_diameter', val=(lb+ub)/2)
                
            elif opt_dv_key[k]  == "floating.member_column3:outer_diameter":
                lb, ub = self.get_bounds(avail_inputs,'floating.member_column3:outer_diameter')
                model.add_design_var('floating_member_column3:outer_diameter', lower=lb, upper=ub)
                model.set_input_defaults('floating_member_column3:outer_diameter', val=(lb+ub)/2)
                
            elif opt_dv_key[k]  == "floating.member_Y_pontoon_upper1:outer_diameter":
                lb, ub = self.get_bounds(avail_inputs,'floating.member_Y_pontoon_upper1:outer_diameter')
                model.add_design_var('floating_member_Y_pontoon_upper1:outer_diameter', lower=lb, upper=ub)
                model.set_input_defaults('floating_member_Y_pontoon_upper1:outer_diameter', val=(lb+ub)/2)
                
            elif opt_dv_key[k]  == "floating.member_Y_pontoon_upper2:outer_diameter":
                lb, ub = self.get_bounds(avail_inputs,'floating.member_Y_pontoon_upper2:outer_diameter')
                model.add_design_var('floating_member_Y_pontoon_upper2:outer_diameter', lower=lb, upper=ub)
                model.set_input_defaults('floating_member_Y_pontoon_upper2:outer_diameter', val=(lb+ub)/2)
                
            elif opt_dv_key[k]  == "floating.member_Y_pontoon_upper3:outer_diameter":
                lb, ub = self.get_bounds(avail_inputs,'floating.member_Y_pontoon_upper3:outer_diameter')
                model.add_design_var('floating_member_Y_pontoon_upper3:outer_diameter', lower=lb, upper=ub)
                model.set_input_defaults('floating_member_Y_pontoon_upper3:outer_diameter', val=(lb+ub)/2)
                
            elif opt_dv_key[k]  == "floating.member_Y_pontoon_lower1:outer_diameter":
                lb, ub = self.get_bounds(avail_inputs,'floating.member_Y_pontoon_lower1:outer_diameter')
                model.add_design_var('floating_member_Y_pontoon_lower1:outer_diameter', lower=lb, upper=ub)
                model.set_input_defaults('floating_member_Y_pontoon_lower1:outer_diameter', val=(lb+ub)/2)
                
            elif opt_dv_key[k]  == "floating.member_Y_pontoon_lower2:outer_diameter":
                lb, ub = self.get_bounds(avail_inputs,'floating.member_Y_pontoon_lower2:outer_diameter')
                model.add_design_var('floating_member_Y_pontoon_lower2:outer_diameter', lower=lb, upper=ub)
                model.set_input_defaults('floating_member_Y_pontoon_lower2:outer_diameter', val=(lb+ub)/2)
                
            elif opt_dv_key[k]  == "floating.member_Y_pontoon_lower3:outer_diameter":
                lb, ub = self.get_bounds(avail_inputs,'floating.member_Y_pontoon_lower3:outer_diameter')
                model.add_design_var('floating_member_Y_pontoon_lower3:outer_diameter', lower=lb, upper=ub)
                model.set_input_defaults('floating_member_Y_pontoon_lower3:outer_diameter', val=(lb+ub)/2)
                
            elif opt_dv_key[k]  == "tune_rosco_ivc.Kp_float":
                lb, ub = self.get_bounds(avail_inputs,'tune_rosco_ivc.Kp_float')
                model.add_design_var('tune_rosco_ivc_Kp_float', lower=lb, upper=ub)
                model.set_input_defaults('tune_rosco_ivc_Kp_float', val=(lb+ub)/2)
            else:
                print("Please choose correct design variable")
     
    def get_parameter_val(self, DCA_req,opt_param_key):
        par_indx = DCA_req['DesignParms']['DesignParms_keys'].index(opt_param_key)
        val = DCA_req['DesignParms']['DesignParms_values'][par_indx]
        return val
    
    def add_parameters(self, DCA_req):
        opt_params_key = self.opt_params_key
        model = self.model
        
        for k in range(len(opt_params_key)):  #need to get the values from Saeid's code, the sm inputs that are not dvs or opt parameters also need vals
            
            if opt_params_key[k] == 'floating.jointdv_0':
                model.set_input_defaults('floating_jointdv_0', val=self.get_parameter_val(DCA_req,'floating.jointdv_0'))
                
            elif opt_params_key[k]  == "floating.jointdv_1":
                model.set_input_defaults('floating_jointdv_1', val=self.get_parameter_val(DCA_req,'floating.jointdv_1'))
                
            elif opt_params_key[k]  == "floating.jointdv_2":
                model.set_input_defaults('floating_jointdv_2', val=self.get_parameter_val(DCA_req,'floating.jointdv_2'))
                
            elif opt_params_key[k]  == "floating.memgrp0.outer_diameter_in":
                model.set_input_defaults('floating_memgrp0_outer_diameter_in', val=self.get_parameter_val(DCA_req,'floating.memgrp0.outer_diameter_in'))
                
            elif opt_params_key[k]  == "floating.memgrp1.outer_diameter_in":
                model.set_input_defaults('floating_memgrp1_outer_diameter_in', val=self.get_parameter_val(DCA_req,'floating.memgrp1.outer_diameter_in'))
                
            elif opt_params_key[k]  == "floating.memgrp2.outer_diameter_in":
                model.set_input_defaults('floating_memgrp2_outer_diameter_in', val=self.get_parameter_val(DCA_req,'floating.memgrp2.outer_diameter_in'))
                
            elif opt_params_key[k]  == "floating.memgrp3.outer_diameter_in":
                model.set_input_defaults('floating_memgrp3_outer_diameter_in', val=self.get_parameter_val(DCA_req,'floating.memgrp3.outer_diameter_in'))
                
            elif opt_params_key[k]  == "tune_rosco_ivc.ps_percent":
                model.set_input_defaults('tune_rosco_ivc_ps_percent', val=self.get_parameter_val(DCA_req,'tune_rosco_ivc.ps_percent'))
                
            elif opt_params_key[k]  == "tune_rosco_ivc.omega_pc":
                model.set_input_defaults('tune_rosco_ivc_omega_pc', val=self.get_parameter_val(DCA_req,'tune_rosco_ivc.omega_pc'))
                
            elif opt_params_key[k]  == "tune_rosco_ivc.ptfm_freq":
                model.set_input_defaults('tune_rosco_ivc_ptfm_freq', val=self.get_parameter_val(DCA_req,'tune_rosco_ivc.ptfm_freq'))
                
            elif opt_params_key[k]  == "tune_rosco_ivc.zeta_pc":
                model.set_input_defaults('tune_rosco_ivc_zeta_pc', val=self.get_parameter_val(DCA_req,'tune_rosco_ivc.zeta_pc'))
                
            elif opt_params_key[k]  == "configuration.rotor_diameter_user":
                model.set_input_defaults('configuration_rotor_diameter_user', val=self.get_parameter_val(DCA_req,'configuration.rotor_diameter_user'))
            
            elif opt_params_key[k]  == "floating.member_main_column:outer_diameter":
                model.set_input_defaults('floating_member_main_column:outer_diameter', val=self.get_parameter_val(DCA_req,'floating.member_main_column:outer_diameter'))
                
            elif opt_params_key[k]  == "floating.member_column1:outer_diameter":
                model.set_input_defaults('floating_member_column1:outer_diameter', val=self.get_parameter_val(DCA_req,'floating.member_column1:outer_diameter'))
                
            elif opt_params_key[k]  == "floating.member_column2:outer_diameter":
                model.set_input_defaults('floating_member_column2:outer_diameter', val=self.get_parameter_val(DCA_req,'floating.member_column2:outer_diameter'))
                
            elif opt_params_key[k]  == "floating.member_column3:outer_diameter":
                model.set_input_defaults('floating_member_column3:outer_diameter', val=self.get_parameter_val(DCA_req,'floating.member_column3:outer_diameter'))
                
            elif opt_params_key[k]  == "floating.member_Y_pontoon_upper1:outer_diameter":
                model.set_input_defaults('floating_member_Y_pontoon_upper1:outer_diameter', val=self.get_parameter_val(DCA_req,'floating.member_Y_pontoon_upper1:outer_diameter'))
                
            elif opt_params_key[k]  == "floating.member_Y_pontoon_upper2:outer_diameter":
                model.set_input_defaults('floating_member_Y_pontoon_upper2:outer_diameter', val=self.get_parameter_val(DCA_req,'floating.member_Y_pontoon_upper2:outer_diameter'))
                
            elif opt_params_key[k]  == "floating.member_Y_pontoon_upper3:outer_diameter":
                model.set_input_defaults('floating_member_Y_pontoon_upper3:outer_diameter', val=self.get_parameter_val(DCA_req,'floating.member_Y_pontoon_upper3:outer_diameter'))
                
            elif opt_params_key[k]  == "floating.member_Y_pontoon_lower1:outer_diameter":
                model.set_input_defaults('floating_member_Y_pontoon_lower1:outer_diameter', val=self.get_parameter_val(DCA_req,'floating.member_Y_pontoon_lower1:outer_diameter'))
                
            elif opt_params_key[k]  == "floating.member_Y_pontoon_lower2:outer_diameter":
                model.set_input_defaults('floating_member_Y_pontoon_lower2:outer_diameter', val=self.get_parameter_val(DCA_req,'floating.member_Y_pontoon_lower2:outer_diameter'))
                
            elif opt_params_key[k]  == "floating.member_Y_pontoon_lower3:outer_diameter":
                model.set_input_defaults('floating_member_Y_pontoon_lower3:outer_diameter', val=self.get_parameter_val(DCA_req,'floating.member_Y_pontoon_lower3:outer_diameter'))
                
            elif opt_params_key[k]  == "tune_rosco_ivc.Kp_float":
                model.set_input_defaults('tune_rosco_ivc_Kp_float', val=self.get_parameter_val(DCA_req,'tune_rosco_ivc.Kp_float'))
            else:
                print("Please choose correct parameter")
                
    def get_constraint_bounds(self, DCA_req,constraint_key):
        cons_indx = DCA_req['constraints']['constraints_keys'].index(constraint_key)
        lb = DCA_req['constraints']['constraints_bounds'][0,cons_indx]
        ub = DCA_req['constraints']['constraints_bounds'][1,cons_indx]
        return lb, ub
       
    def add_constraints(self, DCA_req):
        
        constraints_key = self.constraints_key
        model = self.model
        for k in range(len(constraints_key)):  
            
            if constraints_key[k] == 'raft.Max_PtfmPitch':
                lb, ub = self.get_constraint_bounds(DCA_req, 'raft.Max_PtfmPitch')
                model.add_constraint('raft_Max_PtfmPitch', lower = lb, upper = ub)
                
            elif constraints_key[k]  == "raft.Std_PtfmPitch":
                lb, ub = self.get_constraint_bounds(DCA_req, "raft.Std_PtfmPitch")
                model.add_constraint('raft_Std_PtfmPitch', lower = lb, upper = ub)
                
            elif constraints_key[k]  == "raft.max_nac_accel":
                lb, ub = self.get_constraint_bounds(DCA_req, "raft.max_nac_accel")
                model.add_constraint('raft_max_nac_accel', lower = lb, upper = ub)
                
            # elif constraints_key[k]  == "floatingse.structural_frequencies":
            #     lb, ub = self.get_constraint_bounds(DCA_req, "floatingse.structural_frequencies")
            #     model.add_constraint('floatingse_structural_frequencies', lower = lb, upper= ub)
                
            elif constraints_key[k]  == "floatingse.structural_frequencies_0":
                lb, ub = self.get_constraint_bounds(DCA_req, "floatingse.structural_frequencies_0")
                model.add_constraint('floatingse_structural_frequencies_0', lower = lb, upper= ub)
        
            else:
                print("Please choose correct constraint")


    def run_optimization(self, DCA_req):
        
        self.create_problem()
        self.add_design_vars()
        self.add_parameters(DCA_req)
        self.add_constraints(DCA_req)
        
        prob = self.prob
        prob.model.add_objective(self.objective_key_wo_dot[0])

        # Setup the problem
        prob.setup()
        # Execute the model with the given inputs
        prob.run_driver()
        self._opt_run = True
        # prob.model.list_inputs() #used for debugging
        # prob.model.list_outputs()
        # prob.list_driver_vars(print_arrays=True,
        #               desvar_opts=['lower', 'upper', 'ref', 'ref0',
        #                             'indices', 'adder', 'scaler',
        #                             'parallel_deriv_color', 'min', 'max'],
        #               cons_opts=['lower', 'upper', 'equals', 'ref', 'ref0',
        #                           'indices', 'adder', 'scaler', 'linear', 'min', 'max'],
        #               objs_opts=['ref', 'ref0',
        #                           'indices', 'adder', 'scaler',
        #                           'parallel_deriv_color',
        #                           'cache_linear_solution'])
        
        
    def save_opt_output(self, opt_filename):
        
        if not self._sm_loaded:
                raise Exception('SM data needs to be loaded first.')
        if not self._opt_run:
                raise Exception('Optimization needs to be run before saving to file.')
                
        prob = self.prob
        objective_st = prob.get_val(self.objective_key_wo_dot[0])  
        
        # output dictionary
        opt_output = {
          "dvs": {"dv_keys": [], "dv_values":np.zeros((len(self.opt_dv_key),))},
          "objective": {"objective_keys": self.objective_key, "objective_values":objective_st},
          "constraints":{"constraints_keys": [], "constraints_values":np.zeros((len(self.constraints_key),))}, 
          "success": prob.driver.result.success,
        }
           
        print(prob.driver.result)
        for k in range(len(self.constraints_key)):
            cons_val = prob.get_val(self.constraints_key_wo_dot[k]) 
            opt_output['constraints']['constraints_keys'].append(self.constraints_key[k])
            opt_output['constraints']['constraints_values'][k] = cons_val
            
        for k in range(len(self.opt_dv_key)):
            dv_val = prob.get_val(self.opt_dv_key_wo_dot[k]) 
            opt_output['dvs']['dv_keys'].append(self.opt_dv_key[k])
            opt_output['dvs']['dv_values'][k] = dv_val
            
        try:
            with open(opt_filename, 'wb') as fid:
                pkl.dump(opt_output, fid, protocol=5)
        except:
            print('Unable to write optimization result file: {:}.'.format(opt_filename))
            raise Exception('Unable to write optimization result file: {:}.'.format(opt_filename))
                
        # return opt_output



class SM_Comp(om.ExplicitComponent):
    
    def __init__(self, WindTurbineSMOpt):
        super().__init__()
        self.avail_input_keys = WindTurbineSMOpt.avail_input_keys
        self.opt_output_key = WindTurbineSMOpt.opt_output_key
        self.avail_outputs_keys_compact = WindTurbineSMOpt.avail_outputs_keys_compact
        self.sm2 = WindTurbineSMOpt.sm2
        
   

    def setup(self):
        # WindTurbineSMOpt()
        # WindTurbineSMOpt().get_opt_vars(DCA_req)
        # avail_input_keys = WindTurbineSMOpt.avail_input_keys
        # opt_output_key = WindTurbineSMOpt.opt_output_key
        
        avail_input_keys = self.avail_input_keys
        opt_output_key = self.opt_output_key
        
        for k in range(len(avail_input_keys)):   #add all sm inputs, add their values
            
            if avail_input_keys[k] == 'floating.jointdv_0':
                self.add_input('floating_jointdv_0')
                
            elif avail_input_keys[k]  == "floating.jointdv_1":
                self.add_input('floating_jointdv_1')
                
            elif avail_input_keys[k]  == "floating.jointdv_2":
                self.add_input('floating_jointdv_2')
                
            elif avail_input_keys[k]  == "floating.memgrp0.outer_diameter_in":
                self.add_input('floating_memgrp0_outer_diameter_in')
                
            elif avail_input_keys[k]  == "floating.memgrp1.outer_diameter_in":
                self.add_input('floating_memgrp1_outer_diameter_in')
                
            elif avail_input_keys[k]  == "floating.memgrp2.outer_diameter_in":
                self.add_input('floating_memgrp2_outer_diameter_in')
                
            elif avail_input_keys[k]  == "floating.memgrp3.outer_diameter_in":
                self.add_input('floating_memgrp3_outer_diameter_in')
                
            elif avail_input_keys[k]  == "tune_rosco_ivc.ps_percent":
                self.add_input('tune_rosco_ivc_ps_percent')
                
            elif avail_input_keys[k]  == "tune_rosco_ivc.omega_pc":
                self.add_input('tune_rosco_ivc_omega_pc')
                
            elif avail_input_keys[k]  == "tune_rosco_ivc.ptfm_freq":
                self.add_input('tune_rosco_ivc_ptfm_freq')
                
            elif avail_input_keys[k]  == "tune_rosco_ivc.zeta_pc":
                self.add_input('tune_rosco_ivc_zeta_pc')
                
            elif avail_input_keys[k]  == "configuration.rotor_diameter_user":
                self.add_input('configuration_rotor_diameter_user')
                
            elif avail_input_keys[k]  == "floating.member_main_column:outer_diameter":
                self.add_input('floating_member_main_column:outer_diameter')
                
            elif avail_input_keys[k]  == "floating.member_column1:outer_diameter":
                self.add_input('floating_member_column1:outer_diameter')
                
            elif avail_input_keys[k]  == "floating.member_column2:outer_diameter":
                self.add_input('floating_member_column2:outer_diameter')
                
            elif avail_input_keys[k]  == "floating.member_column3:outer_diameter":
                self.add_input('floating_member_column3:outer_diameter')
                
            elif avail_input_keys[k]  == "floating.member_Y_pontoon_upper1:outer_diameter":
                self.add_input('floating_member_Y_pontoon_upper1:outer_diameter')
                
            elif avail_input_keys[k]  == "floating.member_Y_pontoon_upper2:outer_diameter":
                self.add_input('floating_member_Y_pontoon_upper2:outer_diameter')
                
            elif avail_input_keys[k]  == "floating.member_Y_pontoon_upper3:outer_diameter":
                self.add_input('floating_member_Y_pontoon_upper3:outer_diameter')
                
            elif avail_input_keys[k]  == "floating.member_Y_pontoon_lower1:outer_diameter":
                self.add_input('floating_member_Y_pontoon_lower1:outer_diameter')
                
            elif avail_input_keys[k]  == "floating.member_Y_pontoon_lower2:outer_diameter":
                self.add_input('floating_member_Y_pontoon_lower2:outer_diameter')
                
            elif avail_input_keys[k]  == "floating.member_Y_pontoon_lower3:outer_diameter":
                self.add_input('floating_member_Y_pontoon_lower3:outer_diameter')
                
            elif avail_input_keys[k]  == "tune_rosco_ivc.Kp_float":
                self.add_input('tune_rosco_ivc_Kp_float')
            else:
                print("Please choose correct input")


        for k in range(len(opt_output_key)):
            
            if opt_output_key[k] == 'raft.Max_PtfmPitch':
                self.add_output('raft_Max_PtfmPitch')
                
            elif opt_output_key[k]  == "raft.Std_PtfmPitch":
                self.add_output('raft_Std_PtfmPitch')
                
            elif opt_output_key[k]  == "raft.max_nac_accel":
                self.add_output('raft_max_nac_accel')
                
            # elif opt_output_key[k]  == "floatingse.structural_frequencies":
            #     self.add_output('floatingse_structural_frequencies')
                
            elif opt_output_key[k]  == "floatingse.structural_frequencies_0":
                self.add_output('floatingse_structural_frequencies_0')
                
            elif opt_output_key[k]  == "floatingse.platform_mass":
                self.add_output('floatingse_platform_mass')
                
            elif opt_output_key[k]  == "raft.platform_mass":
                self.add_output('raft_platform_mass')

            else:
                print("Please choose correct constraint")
        

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')
        
    def run_predict(self, predict_output, predict_input):
        # avail_outputs_keys_compact = WindTurbineSMOpt.avail_outputs_keys_compact
        avail_outputs_keys_compact = self.avail_outputs_keys_compact
        # sm2 = WindTurbineSMOpt.sm2
        sm2 = self.sm2
        
        output_indx = avail_outputs_keys_compact.index(predict_output)
        output_value, variance = sm2[output_indx]['surrogate'].predict(np.array([predict_input]))
        return output_value

    def compute(self, inputs, outputs):
        # avail_input_keys = WindTurbineSMOpt.avail_input_keys
        # opt_output_key = WindTurbineSMOpt.opt_output_key
        
        avail_input_keys = self.avail_input_keys
        opt_output_key = self.opt_output_key

        predict_input = []
        for k in range(len(avail_input_keys)):   #add all sm inputs
            
            if avail_input_keys[k] == 'floating.jointdv_0':
                floating_jointdv_0 = inputs['floating_jointdv_0']
                predict_input.append(floating_jointdv_0)
                
            elif avail_input_keys[k]  == "floating.jointdv_1":
                floating_jointdv_1 = inputs['floating_jointdv_1']
                predict_input.append(floating_jointdv_1)
                
            elif avail_input_keys[k]  == "floating.jointdv_2":
                floating_jointdv_2 = inputs['floating_jointdv_2']
                predict_input.append(floating_jointdv_2)
                
            elif avail_input_keys[k]  == "floating.memgrp0.outer_diameter_in":
                floating_memgrp0_outer_diameter_in = inputs['floating_memgrp0_outer_diameter_in']
                predict_input.append(floating_memgrp0_outer_diameter_in)
                
            elif avail_input_keys[k]  == "floating.memgrp1.outer_diameter_in":
                floating_memgrp1_outer_diameter_in = inputs['floating_memgrp1_outer_diameter_in']
                predict_input.append(floating_memgrp1_outer_diameter_in)
                
            elif avail_input_keys[k]  == "floating.memgrp2.outer_diameter_in":
                floating_memgrp2_outer_diameter_in = inputs['floating_memgrp2_outer_diameter_in']
                predict_input.append(floating_memgrp2_outer_diameter_in)
                
            elif avail_input_keys[k]  == "floating.memgrp3.outer_diameter_in":
                floating_memgrp3_outer_diameter_in = inputs['floating_memgrp3_outer_diameter_in']
                predict_input.append(floating_memgrp3_outer_diameter_in)
                
            elif avail_input_keys[k]  == "tune_rosco_ivc.ps_percent":
                tune_rosco_ivc_ps_percent = inputs['tune_rosco_ivc_ps_percent']
                predict_input.append(tune_rosco_ivc_ps_percent)
                
            elif avail_input_keys[k]  == "tune_rosco_ivc.omega_pc":
                tune_rosco_ivc_omega_pc = inputs['tune_rosco_ivc_omega_pc']
                predict_input.append(tune_rosco_ivc_omega_pc)
                
            elif avail_input_keys[k]  == "tune_rosco_ivc.ptfm_freq":
                tune_rosco_ivc_ptfm_freq = inputs['tune_rosco_ivc_ptfm_freq']
                predict_input.append(tune_rosco_ivc_ptfm_freq)
                
            elif avail_input_keys[k]  == "tune_rosco_ivc.zeta_pc":
                tune_rosco_ivc_zeta_pc = inputs['tune_rosco_ivc_zeta_pc']
                predict_input.append(tune_rosco_ivc_zeta_pc)
                
            elif avail_input_keys[k]  == "configuration.rotor_diameter_user":
                configuration_rotor_diameter_user = inputs['configuration_rotor_diameter_user']
                predict_input.append(configuration_rotor_diameter_user)
            
            elif avail_input_keys[k]  == "floating.member_main_column:outer_diameter":
                floating_member_main_column_outer_diameter = inputs['floating_member_main_column:outer_diameter']
                predict_input.append(floating_member_main_column_outer_diameter)
                
            elif avail_input_keys[k]  == "floating.member_column1:outer_diameter":
                floating_member_column1_outer_diameter = inputs['floating_member_column1:outer_diameter']
                predict_input.append(floating_member_column1_outer_diameter)
                
            elif avail_input_keys[k]  == "floating.member_column2:outer_diameter":
                floating_member_column2_outer_diameter = inputs['floating_member_column2:outer_diameter']
                predict_input.append(floating_member_column2_outer_diameter)
                
            elif avail_input_keys[k]  == "floating.member_column3:outer_diameter":
                floating_member_column3_outer_diameter = inputs['floating_member_column3:outer_diameter']
                predict_input.append(floating_member_column3_outer_diameter)
                
            elif avail_input_keys[k]  == "floating.member_Y_pontoon_upper1:outer_diameter":
                floating_member_Y_pontoon_upper1_outer_diameter = inputs['floating_member_Y_pontoon_upper1:outer_diameter']
                predict_input.append(floating_member_Y_pontoon_upper1_outer_diameter)
                
            elif avail_input_keys[k]  == "floating.member_Y_pontoon_upper2:outer_diameter":
                floating_member_Y_pontoon_upper2_outer_diameter = inputs['floating_member_Y_pontoon_upper2:outer_diameter']
                predict_input.append(floating_member_Y_pontoon_upper2_outer_diameter)
                
            elif avail_input_keys[k]  == "floating.member_Y_pontoon_upper3:outer_diameter":
                floating_member_Y_pontoon_upper3_outer_diameter = inputs['floating_member_Y_pontoon_upper3:outer_diameter']
                predict_input.append(floating_member_Y_pontoon_upper3_outer_diameter)
                
            elif avail_input_keys[k]  == "floating.member_Y_pontoon_lower1:outer_diameter":
                floating_member_Y_pontoon_lower1_outer_diameter = inputs['floating_member_Y_pontoon_lower1:outer_diameter']
                predict_input.append(floating_member_Y_pontoon_lower1_outer_diameter)
                
            elif avail_input_keys[k]  == "floating.member_Y_pontoon_lower2:outer_diameter":
                floating_member_Y_pontoon_lower2_outer_diameter = inputs['floating_member_Y_pontoon_lower2:outer_diameter']
                predict_input.append(floating_member_Y_pontoon_lower2_outer_diameter)
                
            elif avail_input_keys[k]  == "floating.member_Y_pontoon_lower3:outer_diameter":
                floating_member_Y_pontoon_lower3_outer_diameter = inputs['floating_member_Y_pontoon_lower3:outer_diameter']
                predict_input.append(floating_member_Y_pontoon_lower3_outer_diameter)
                
            elif avail_input_keys[k]  == "tune_rosco_ivc.Kp_float":
                tune_rosco_ivc_Kp_float = inputs['tune_rosco_ivc_Kp_float']
                predict_input.append(tune_rosco_ivc_Kp_float)
            else:
                print("Please choose correct input")
                        
        predict_input_flat = [arr.flatten()[0] for arr in predict_input]
        for k in range(len(opt_output_key)):
            
            if opt_output_key[k] == 'raft.Max_PtfmPitch':
                outputs['raft_Max_PtfmPitch'] = self.run_predict(['raft.Max_PtfmPitch'], predict_input_flat) 
                
            elif opt_output_key[k]  == "raft.Std_PtfmPitch":
                outputs['raft_Std_PtfmPitch'] = self.run_predict(['raft.Std_PtfmPitch'], predict_input_flat)
                
            elif opt_output_key[k]  == "raft.max_nac_accel":
                outputs['raft_max_nac_accel'] = self.run_predict(['raft.max_nac_accel'], predict_input_flat)
                
            elif opt_output_key[k]  == "floatingse.structural_frequencies":
                outputs['floatingse_structural_frequencies'] = self.run_predict(['floatingse.structural_frequencies'], predict_input_flat)
                
            elif opt_output_key[k]  == "floatingse.structural_frequencies_0":
                outputs['floatingse_structural_frequencies_0'] = self.run_predict(['floatingse.structural_frequencies_0'], predict_input_flat)
            
            elif opt_output_key[k]  == "floatingse.platform_mass":
                outputs['floatingse_platform_mass'] = self.run_predict(['floatingse.platform_mass'], predict_input_flat)
            
            elif opt_output_key[k]  == "raft.platform_mass":
                outputs['raft_platform_mass'] = self.run_predict(['raft.platform_mass'], predict_input_flat)
                
            else:
                print("Please choose correct output")
                

# ####################################################################
# #this info is in runWEIS
# run_dir                = os.path.dirname( os.path.realpath(__file__) ) + os.sep
# ex_dir = "/home/elenaf3/Desktop/DC_studies_code/WEIS/examples/15_RAFT_Studies/"
# sm_file = '/home/elenaf3/Desktop/DC_studies_code/WEIS/weis/optimization/log_opt.smt'
# fname_wt_input         = os.path.join(ex_dir,"..","06_IEA-15-240-RWT", "IEA-15-240-RWT_VolturnUS-S.yaml")
# fname_modeling_options = ex_dir + "modeling_options_level1_doe.yaml"
# fname_analysis_options = ex_dir + "analysis_options_level1_doe.yaml"
# wt_initial = WindTurbineOntologyPythonWEIS(fname_wt_input, fname_modeling_options, fname_analysis_options)
# wt_init, modeling_options, opt_options = wt_initial.get_input_data()
# folder_output = opt_options['general']['folder_output']
# # opt_filename = os.path.join(folder_output, 'opt_output' + '.pkl')
# opt_filename ='opt_output' + '.pkl'

# # requested dv and outputs for optimization (this info comes from Saeid)
# DCA_req = {
#   "DesignVars": ["floating.member_column1:outer_diameter", "tune_rosco_ivc.Kp_float", "tune_rosco_ivc.omega_pc"],
#   "DesignParms": {"DesignParms_keys": ["floating.memgrp0.outer_diameter_in", "tune_rosco_ivc.ps_percent"], "DesignParms_values": np.array([6.5, 5])},
#   "objective": "floatingse.platform_mass",
#   "constraints":{"constraints_keys": ["raft.Max_PtfmPitch"], "constraints_bounds": np.array([[-5], [5]])}, 
# }

# ####################################################################

# ####################################################################                
# # this part will be moved to runWEIS
# WTSMO = WindTurbineSMOpt()
# WTSMO.read_sm(sm_file, modeling_options, opt_options) 
# WTSMO.get_opt_vars(DCA_req) 
# WTSMO.run_optimization()
# WTSMO.save_opt_output(opt_filename)

# # opt_output = WTSMO.save_opt_output()
# ####################################################################