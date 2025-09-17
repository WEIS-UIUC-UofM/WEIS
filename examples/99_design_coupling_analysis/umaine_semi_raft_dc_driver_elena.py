import os
import numpy as np
from weis.ftw.weis_wrapper import ftw_doe
from weis.ftw.surrogate    import ftw_surrogate_modeling
import yaml
from weis.glue_code.runWEIS     import run_weis
from weis.ftw.ftw_SBO import WindTurbineSMOpt
# TEST_RUN will reduce the number and duration of simulations
TEST_RUN = False

## File management
run_dir = os.path.realpath(os.curdir)
#run_dir = os.path.dirname( os.path.realpath(__file__) )
fname_wt_input = os.path.join(run_dir, "..", "00_setup", "ref_turbines", "IEA-15-240-RWT_VolturnUS-S_rectangular.yaml")
fname_modeling_options = os.path.join(run_dir, "umaine_semi_raft_dc_modeling.yaml")
fname_analysis_options = os.path.join(run_dir, "umaine_semi_raft_dc_analysis.yaml")
geometry_override = {}
modeling_override = {}
analysis_override = {}

# Run DOE to prepare for the surrogate model training
doedata, fname_doedata, fname_smt, skip_training_if_sm_exist = ftw_doe(
    fname_wt_input, fname_modeling_options, fname_analysis_options,
    geometry_override, modeling_override, analysis_override, TEST_RUN)

# Train WTSM
WTSM = ftw_surrogate_modeling(fname_doedata=fname_doedata, fname_smt=fname_smt,
    doedata=doedata, WTSM=None, skip_training_if_sm_exist=skip_training_if_sm_exist)

# Usage Example (Temporary code --- to be removed)
input_bounds = WTSM.get_input_bounds()
input_lower = input_bounds[0,:].reshape(1,-1)
input_upper = input_bounds[1,:].reshape(1,-1)
x_normalized = np.random.rand(1,input_lower.size)
x = input_lower + (input_upper - input_lower)*x_normalized
y, v = WTSM.predict(x)
print(y)
print(v)

# %%

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
    wt_opt_sim, modeling_options_sim, opt_options_sim = run_weis(fname_wt_input, fname_modeling_options_sim, fname_analysis_options_sim, {})
    
    # t1 = time.time()
    # total = t1-t0
    # print(total)
    return wt_opt_sim

WTSMO = WindTurbineSMOpt()

# %%
avail_input_keys = WTSMO.get_inputs(doedata)
#get x0 values and f(x0)
DesignVars_x0 = np.zeros((len(avail_input_keys),))
wt_opt_sim = run_model()
for k in range(len(avail_input_keys)):
    var = avail_input_keys[k]
    x0 = wt_opt_sim.get_val(var)
    DesignVars_x0[k] = x0[0]  

DCA_req = {
    'DesignVars': {
        'DesignVars_keys': avail_input_keys,
        'DesignVars_x0': DesignVars_x0,
        'bounds': input_bounds
    },
    'DesignParms': {
        'DesignParms_keys': [],
        'DesignParms_values': []
    },
    'objective': 'floatingse.system_structural_mass',
    # 'objective': 'financese.lcoe',

    'constraints': {
        'constraints_keys': 'raft.max_nac_accel',
        'constraints_bounds': [-1000, 0.7]
    }
}

opt_filename = 'opt_res'
WTSMO = WindTurbineSMOpt()
WTSMO.read_sm(WTSM, input_bounds,doedata)
WTSMO.get_opt_vars(DCA_req)
WTSMO.create_problem(fname_wt_input, fname_modeling_options, fname_analysis_options, modeling_override)
WTSMO.run_optimization(DCA_req)
WTSMO.save_opt_output(opt_filename)