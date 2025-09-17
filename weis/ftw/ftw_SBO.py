import numpy as np
from weis.glue_code.gc_LoadInputs     import WindTurbineOntologyPythonWEIS
from wisdem.glue_code.gc_PoseOptimization import PoseOptimization
import openmdao.api as om
import pickle as pkl

class WindTurbineSMOpt():
    
    def __init__(self):
        self._sm_loaded = False
        self._opt_run = False
        self.avail_input_keys = []
        self.opt_output_key = []    
        self.avail_outputs_keys = []
        self.sm = []
        self.scipy_methods = [
                    "SLSQP",
                    "Nelder-Mead",
                    "COBYLA",
                ]
        self.pyoptsparse_methods = [
                    "SNOPT",
                    "CONMIN",
                    "NSGA2",
                    "IPOPT"
                ]

    
    def get_outputs(self, doedata):
        doe_outputs = doedata['output']
        n_outputs = len(doe_outputs)
        avail_output_keys = []
        for k in range(n_outputs):
            avail_output_keys.append(doe_outputs[k]['name'])
               
        return avail_output_keys
      
    def get_inputs(self, doedata):
        doe_inputs = doedata['input']
        n_inputs = len(doe_inputs)
        avail_input_keys = []
        for k in range(n_inputs):
            avail_input_keys.append(doe_inputs[k]['name'])
               
        return avail_input_keys
    
    def map_keys(self, keys):
        clean_keys = [sub.replace('.', '_') for sub in keys]
        keys_map = dict(zip(keys, clean_keys))      
        
        return clean_keys, keys_map
    
    def read_sm(self, WTSM, input_bounds, doedata):
        
        # identify available inputs and outputs provided by the sm
        avail_input_keys = self.get_inputs(doedata)
        avail_output_keys = self.get_outputs(doedata)
        bounds_inputs_map = dict(zip(avail_input_keys, input_bounds.T))
               
        # Store data
        self.avail_output_keys = avail_output_keys
        self.avail_input_keys = avail_input_keys
        self.sm = WTSM
        self.bounds_inputs_map = bounds_inputs_map
        self._sm_loaded = True
        self.doedata = doedata
 
    
    def get_opt_vars(self, DCA_req):
        req_dv = DCA_req["DesignVars"]["DesignVars_keys"]
        req_design_parms = DCA_req["DesignParms"]["DesignParms_keys"]
        req_objective = [DCA_req["objective"]]
        req_constraints = DCA_req["constraints"]["constraints_keys"]
        if type(req_constraints)==str:
            req_constraints = [req_constraints]
        req_outputs = req_objective + req_constraints
        req_inputs = req_dv + req_design_parms
        
        avail_output_keys = self.avail_output_keys
        avail_input_keys = self.avail_input_keys
        
        opt_output_key = list(set(req_outputs).intersection(avail_output_keys))
        objective_key = list(set(req_objective).intersection(avail_output_keys))
        
        if len(req_objective)==0:
            raise Exception('Objective not available or missing, select a supported objective')
        if len(req_objective)>1:
            raise Exception('multiobjective optimization is not supported')
            
        constraints_key = list(set(req_constraints).intersection(avail_output_keys))
        opt_dv_key = list(set(req_dv).intersection(avail_input_keys))
        opt_params_key = list(set(req_design_parms).intersection(avail_input_keys))
        opt_input_key = list(set(req_inputs).intersection(avail_input_keys))
        
        if not len(constraints_key)==len(req_constraints):
            raise Exception('some constraint is not supported')
        if not len(req_dv)==len(opt_dv_key):
            raise Exception('some design variable is not supported')
            
            
        self.opt_dv_key = opt_dv_key
        self.constraints_key = constraints_key
        self.opt_output_key = opt_output_key
        self.opt_params_key = opt_params_key
        self.objective_key = objective_key
        self.opt_input_key = opt_input_key
    
    
    
    def get_opt_options(self, fname_wt_input, fname_modeling_options, fname_analysis_options, modeling_override): #need to decide if we use the optimization options from the analysis yaml or if we have specific sbo options
        opt_override = {}
        opt_override['driver'] = {}
        opt_override['driver']['optimization'] = {}
        opt_override['driver']['optimization']['flag'] = True
        
        # Load all yaml inputs and validate (also fills in defaults)
        wt_initial = WindTurbineOntologyPythonWEIS(
            fname_wt_input,
            fname_modeling_options,
            fname_analysis_options,
            modeling_override=modeling_override,
            analysis_override=opt_override
            )
        wt_init, modeling_options, opt_options = wt_initial.get_input_data()
        
        
        smb_options = opt_options
        smb_options['driver']['optimization']['flag'] = True
        smb_options['driver']['optimization']['solver'] = 'COBYLA'
        smb_options['driver']['optimization']['max_iter'] = 500
        smb_options['driver']['optimization']['tol'] = 1e-5
        self.smb_options = smb_options
        self.wt_init = wt_init
        self.modeling_options = modeling_options
        return smb_options, wt_init, modeling_options
    
    def create_problem(self, fname_wt_input, fname_modeling_options, fname_analysis_options, modeling_override):
        self.get_opt_options(fname_wt_input, fname_modeling_options, fname_analysis_options, modeling_override)
        smb_options = self.smb_options
        wt_init = self.wt_init
        modeling_options = self.modeling_options
        
        prob = om.Problem(reports=False)
        
        model = prob.model
        clean_input_keys, inputs_key_map = self.map_keys(self.avail_input_keys)
        clean_output_keys, outputs_key_map = self.map_keys(self.avail_output_keys)
        model.add_subsystem('model', subsys=SM_Comp(self), promotes_inputs=clean_input_keys, promotes_outputs=clean_output_keys)        
        
        # from WISDEM/wisdem/glue_code/gc_PoseOptimization.py  sets driver and opt options
        poseoptimization = PoseOptimization(wt_init, modeling_options, smb_options)
        prob = poseoptimization.set_driver(prob)
        
        self.inputs_key_map = inputs_key_map
        self.outputs_key_map = outputs_key_map
        self.prob = prob
        self.model = model
        
    
    def get_bounds(self, dv_key):
        bounds_inputs_map = self.bounds_inputs_map
        bounds = bounds_inputs_map[dv_key]
        lb = bounds[0]
        ub = bounds[1]
        return lb, ub
    
    def add_design_vars(self, DCA_req):
        
        opt_dv_key = self.opt_dv_key
        model = self.model
        
        # add dv
        opt_dv_key_clean, _ = self.map_keys(opt_dv_key)
        for k in range(len(opt_dv_key_clean)):           
            dv_indx = DCA_req['DesignVars']['DesignVars_keys'].index(opt_dv_key[k])
            x0 = DCA_req['DesignVars']['DesignVars_x0'][dv_indx]
            lb, ub = self.get_bounds(opt_dv_key[k])
            model.add_design_var(opt_dv_key_clean[k], lower=lb, upper=ub)
            model.set_input_defaults(opt_dv_key_clean[k], val=x0)
        self.opt_dv_key_clean = opt_dv_key_clean
            
    def get_parameter_val(self, DCA_req, opt_param_key):
        par_indx = DCA_req['DesignParms']['DesignParms_keys'].index(opt_param_key)
        val = DCA_req['DesignParms']['DesignParms_values'][par_indx]
        return val
    
    def add_parameters(self, DCA_req): #need to check that params+dvs=inputs, if not add params w default vals
        opt_params_key = self.opt_params_key
        model = self.model
        opt_params_key_clean, _ = self.map_keys(opt_params_key)
        for k in range(len(opt_params_key)):  
            
                model.set_input_defaults(opt_params_key_clean[k], val=self.get_parameter_val(DCA_req,opt_params_key[k]))
                        
                
    def get_constraint_bounds(self, DCA_req,constraint_key):
        cons_indx = DCA_req['constraints']['constraints_keys'].index(constraint_key)
        if type(DCA_req['constraints']['constraints_keys'])==str:
            lb = DCA_req['constraints']['constraints_bounds'][0]
            ub = DCA_req['constraints']['constraints_bounds'][1]
        else:
            
            lb = DCA_req['constraints']['constraints_bounds'][0,cons_indx]
            ub = DCA_req['constraints']['constraints_bounds'][1,cons_indx]
        return lb, ub
       
    def add_constraints(self, DCA_req):
        
        constraints_key = self.constraints_key
        model = self.model
        constraints_key_clean, _ = self.map_keys(constraints_key)
    
        for k in range(len(constraints_key)):  
                lb, ub = self.get_constraint_bounds(DCA_req, constraints_key[k])
                model.add_constraint(constraints_key_clean[k], lower = lb, upper = ub)
        self.constraints_key_clean = constraints_key_clean
                
        
    def run_optimization(self, DCA_req):
        objective_key = self.objective_key
        prob = self.prob
        # self.create_problem()
        self.add_design_vars(DCA_req)
        self.add_parameters(DCA_req)
        self.add_constraints(DCA_req)
        objective_key_clean, _ = self.map_keys(objective_key)
        prob.model.add_objective(objective_key_clean[0], ref=1e3) #need to change scaling
        
        # Setup the problem
        prob.setup()
        om.n2(prob)
        prob.final_setup()
        om.view_connections(prob, outfile= "sb_opt_connections.html", show_browser=False)
        # Execute the model with the given inputs
        # prob.driver.options['disp'] = True
        prob.run_driver()
        self._opt_run = True
        
        # prob.model.list_inputs() #used for debugging
        # prob.model.list_outputs()
        # prob.list_driver_vars(print_arrays=False,
        #               desvar_opts=['lower', 'upper', 'ref', 'ref0',
        #                             'indices', 'adder', 'scaler',
        #                             'parallel_deriv_color', 'min', 'max'],
        #               cons_opts=['lower', 'upper', 'equals', 'ref', 'ref0',
        #                           'indices', 'adder', 'scaler', 'linear', 'min', 'max'],
        #               objs_opts=['ref', 'ref0',
        #                           'indices', 'adder', 'scaler',
        #                           'parallel_deriv_color',
        #                           'cache_linear_solution'])
        self.objective_key_clean=objective_key_clean
        
       
    def save_opt_output(self, opt_filename):
        # inputs_key_map = self.inputs_key_map 
        # outputs_key_map = self.outputs_key_map 
        # inputs_key_map_inv = {v: k for k, v in inputs_key_map.items()} 
        # outputs_key_map_inv = {v: k for k, v in outputs_key_map.items()} 
        
        if not self._sm_loaded:
                raise Exception('SM data needs to be loaded first.')
        if not self._opt_run:
                raise Exception('Optimization needs to be run before saving to file.')
                
        prob = self.prob
        objective_st = prob.get_val(self.objective_key_clean[0])  
    
        # solver = self.smb_options['SMB_optimization']['solver']
        solver = self.smb_options['driver']['optimization']['solver']
        if solver in self.scipy_methods:
            exit_flag = prob.driver._scipy_optimize_result['status']
        elif solver in self.pyoptsparse_methods:
            exit_flag = prob.driver.pyopt_solution.optInform['value']
        else:
            exit_flag=[]
        # output dictionary and map keys back
        opt_output = {
          "dvs": {"dv_keys": [], "dv_values":np.zeros((len(self.opt_dv_key),))},
          "objective": {"objective_keys": self.objective_key[0], "objective_values":objective_st},
          "constraints":{"constraints_keys": [], "constraints_values":np.zeros((len(self.constraints_key),))}, 
          "success": prob.driver.result.success,
          "exit_flag": exit_flag,
          # 'driver_out': prob.driver.result
          
        }
        # print(prob.driver.opt_settings) 
        # print(prob.driver.options) 
        # print(prob.driver.opt_settings['tol']) 
        # print(prob.driver.opt_settings['acceptable_tol'])
        # print(dir(prob.driver.result))
        # print(prob.driver.result)
        # print(prob.driver.msginfo)
        # print(prob.driver.get_exit_status)
        # print(prob.msginfo)
        # print('raft.heave_avg', prob.get_val('raft_heave_avg'))
        # print('raft.Std_PtfmPitch', prob.get_val('raft_Std_PtfmPitch'))
        # print('raft.heave_period', prob.get_val('raft_heave_period'))
        # print('raft.pitch_period', prob.get_val('raft_pitch_period'))
        # print(prob.driver.result)
        for k in range(len(self.constraints_key)):
            cons_val = prob.get_val(self.constraints_key_clean[k]) 
            opt_output['constraints']['constraints_keys'].append(self.constraints_key[k])
            opt_output['constraints']['constraints_values'][k] = cons_val
            
        for k in range(len(self.opt_dv_key)):
            dv_val = prob.get_val(self.opt_dv_key_clean[k]) 
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
        self.avail_output_keys =  WindTurbineSMOpt.avail_output_keys
        self.avail_input_keys = WindTurbineSMOpt.avail_input_keys
        self.sm = WindTurbineSMOpt.sm
        self.clean_input_keys, _ = WindTurbineSMOpt.map_keys(self.avail_input_keys)
        self.clean_output_keys, _ = WindTurbineSMOpt.map_keys(self.avail_output_keys)
        self.constraints_key = WindTurbineSMOpt.constraints_key
        self.doedata = WindTurbineSMOpt.doedata
        
    def setup(self):
        doedata = self.doedata
        clean_output_keys = self.clean_output_keys
        clean_input_keys = self.clean_input_keys
        for k in range(len(clean_input_keys)):   #add all sm inputs, add their values. Need to get x0 in a better way
            
                self.add_input(clean_input_keys[k], 0.)
        
        out = doedata['output']
        for k in range(len(clean_output_keys)):   #add all sm outputs
                out_key = out[k]['name']
                out_size = out[k]['len']
                self.add_output(clean_output_keys[k], shape=(out_size,) )     

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')
        
    def map_outputs(self, doedata, pred_out):
        out = doedata['output']
        out_dict = {}
        c = 0
        for k in range(len(out)):
            out_key = out[k]['name']
            out_size = out[k]['len']
            out_dict[out_key] = pred_out[0, c:c+out_size]
            c = c+ out_size
        return out_dict
        
    def run_predict(self, predict_output_idx, predict_input_vals):
        doedata = self.doedata
        clean_output_keys = self.clean_output_keys
        avail_output_keys = self.avail_output_keys
        sm = self.sm
        output_values, variance = sm.predict(np.array([predict_input_vals]))
        output_value_dict = self.map_outputs(doedata, output_values)
        output_value = output_value_dict[avail_output_keys[predict_output_idx]]
        # output_value = output_values[0,predict_output_idx]
        
        return output_value

    def compute(self, inputs, outputs):
        
        avail_input_keys = self.avail_input_keys
        avail_output_keys = self.avail_output_keys
        clean_output_keys = self.clean_output_keys
        clean_input_keys = self.clean_input_keys
        predict_input = []
        
        # Automatically collect all input values in order
        predict_input = [inputs[input_name] for input_name in clean_input_keys]
        
        # Flatten if needed
        predict_input_flat = [arr.flatten()[0] for arr in predict_input]
            
        for k in range(len(clean_output_keys)):
            outputs[clean_output_keys[k]] = self.run_predict(k, predict_input_flat) 
            # if avail_output_keys[k]=='floatingse.system_structural_mass':
            #     print(outputs[clean_output_keys[k]])