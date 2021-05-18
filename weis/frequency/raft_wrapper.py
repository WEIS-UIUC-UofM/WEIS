import openmdao.api as om
import numpy as np
import wisdem.commonse.utilities as util
from wisdem.commonse.wind_wave_drag import cylinderDrag
from raft.omdao_raft import RAFT_OMDAO

class RAFT_WEIS(om.Group):

    def initialize(self):
        self.options.declare('modeling_options')
    
    def setup(self):
        # Stuff WEIS options into RAFT options structure
        weis_opt = self.options['modeling_options']

        raft_opt = {}
        raft_opt['nfreq'] = len(weis_opt['Level1']['frequencies'])
        raft_opt['potModMaster'] = weis_opt['Level1']['potential_model_override']
        raft_opt['XiStart'] = weis_opt['Level1']['xi_start']
        raft_opt['nIter'] = weis_opt['Level1']['nIter']
        raft_opt['dlsMax'] = weis_opt['Level1']['dls_max']

        turbine_opt = {}
        turbine_opt['npts'] = weis_opt['WISDEM']['TowerSE']['n_height_tower']
        turbine_opt['scalar_thicknesses'] = turbine_opt['scalar_diameters'] = turbine_opt['scalar_coefficients'] = False
        turbine_opt['shape'] = 'circ'

        members_opt = {}
        members_opt['nmembers'] = len(weis_opt["floating"]["members"]["name"])
        members_opt['npts'] = weis_opt["floating"]["members"]["n_height"]
        members_opt['npts_lfill'] = members_opt['npts_rho_fill'] = [m-1 for m in members_opt['npts']]
        members_opt['ncaps'] = weis_opt["floating"]["members"]["n_bulkheads"]
        members_opt['nreps'] = [0]*members_opt['nmembers']
        members_opt['shape'] = ['circ']*members_opt['nmembers']
        members_opt['scalar_thicknesses'] = members_opt['scalar_diameters'] = [False]*members_opt['nmembers']
        members_opt['scalar_coefficients'] = [False]*members_opt['nmembers']

        mooring_opt = {}
        mooring_opt['nlines'] = weis_opt['mooring']['n_lines']
        mooring_opt['nline_types'] = weis_opt['mooring']['n_line_types']
        mooring_opt['nconnections'] = weis_opt['mooring']['n_nodes']
        

        self.add_subsystem('pre', RAFT_WEIS_Prep(modeling_options=weis_opt), promotes=['*'])
        self.add_subsystem('raft', RAFT_OMDAO(modeling_options=raft_opt,
                                              turbine_options=turbine_opt,
                                              mooring_options=mooring_opt,
                                              member_options=members_opt), promotes=['*'])

class RAFT_WEIS_Prep(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('modeling_options')
    
    def setup(self):
        opt = self.options['modeling_options']
        nfreq = len(opt['Level1']['frequencies'])
        
        n_nodes = opt['mooring']['n_nodes']
        n_lines = opt['mooring']['n_lines']
        n_line_types = opt['mooring']['n_line_types']

        # Process tower layer sections
        n_height_tow = opt["WISDEM"]["TowerSE"]["n_height_tower"]
        n_layers_tow = opt["WISDEM"]["TowerSE"]["n_layers_tower"]
        self.add_input("tower_layer_thickness", val=np.zeros((n_layers_tow, n_height_tow)), units="m")
        self.add_input("tower_rho", val=np.zeros(n_height_tow-1), units="kg/m**3")
        self.add_input("tower_section_height", np.zeros(n_height_tow - 1), units="m", desc="sectional height")
        self.add_input("tower_torsional_stiffness", np.zeros(n_height_tow - 1), units="N*m**2", desc="sectional torsional stiffness")
        self.add_output("turbine_tower_t", val=np.zeros(n_height_tow), units="m")
        self.add_output("turbine_tower_rho_shell", val=0.0, units="kg/m**3")
        self.add_output("turbine_yaw_stiffness", 0.0, units="N*m", desc="sectional torsional stiffness")
        self.add_output('turbine_tower_gamma', val=0.0, units='deg', desc='Twist angle about z-axis')

        # Tower drag
        self.add_input("rho_air", 0.0, units="kg/m**3")
        self.add_input("mu_air", 0.0, units="kg/(m*s)")
        self.add_input('tower_U', val=np.zeros(n_height_tow), units='m/s', desc='Wind speed on the tower')
        self.add_input('turbine_tower_d', val=np.zeros(n_height_tow), units='m', desc='Diameters if circular or side lengths if rectangular')
        self.add_output('turbine_tower_Cd', val=np.zeros(n_height_tow), desc='Transverse drag coefficient')
        self.add_output('turbine_tower_Ca', val=np.zeros(n_height_tow), desc='Transverse added mass coefficient')
        self.add_output('turbine_tower_CdEnd', val=np.zeros(n_height_tow), desc='End axial drag coefficient')
        self.add_output('turbine_tower_CaEnd', val=np.zeros(n_height_tow), desc='End axial added mass coefficient')
        
        # RNA mass properties
        self.add_input("rna_I_TT", np.zeros(6), units="kg*m**2", desc='Moment of inertia at tower top [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]')
        self.add_input('turbine_mRNA', val=0.0, units='kg', desc='RNA mass')
        self.add_input('drive_height', val=0.0, units='m', desc='Distance from tower top to rotor apex (hub height)')
        self.add_input("rna_cm", np.zeros(3), units="m", desc='RNA center of mass from tower top')
        self.add_output('turbine_IxRNA', val=0.0, units='kg*m**2', desc='RNA moment of inertia about local x axis')
        self.add_output('turbine_IrRNA', val=0.0, units='kg*m**2', desc='RNA moment of inertia about local y or z axes')
        self.add_output('turbine_xCG_RNA', val=0.0, units='m', desc='x location of RNA center of mass')

        n_member = len(opt["floating"]["members"]["name"])
        self.add_input("member_variable_height", val=np.zeros(n_member))
        for k in range(n_member):
            n_height = opt["floating"]["members"]["n_height"][k]
            n_layers = opt["floating"]["members"]["n_layers"][k]
            n_ball   = opt["floating"]["members"]["n_ballasts"][k]
            n_bulk   = opt["floating"]["members"]["n_bulkheads"][k]

            self.add_input(f"member{k}:height", val=0.0, units="m")
            self.add_input(f"member{k}:layer_thickness", val=np.zeros((n_layers, n_height)), units="m")
            self.add_input(f"member{k}:rho", val=np.zeros(n_height-1), units="kg/m**3")
            self.add_input(f"member{k}:ballast_grid", val=np.zeros((n_ball,2)))
            self.add_input(f"member{k}:ballast_height", val=np.zeros(n_ball))
            self.add_input(f"member{k}:ballast_density", val=np.zeros(n_ball), units="kg/m**3")
            self.add_input(f"member{k}:ring_stiffener_web_height", 0.0, units="m")
            self.add_input(f"member{k}:ring_stiffener_web_thickness", 0.0, units="m")
            self.add_input(f"member{k}:ring_stiffener_flange_width", 1e-6, units="m")
            self.add_input(f"member{k}:ring_stiffener_flange_thickness", 0.0, units="m")
            self.add_input(f"member{k}:ring_stiffener_spacing", 1000.0, units="m")
            self.add_input(f"platform_member{k+1}_stations", val=np.zeros(n_height))
            
            self.add_output(f"platform_member{k+1}_heading", val=np.zeros(0), units='deg')
            self.add_output(f"platform_member{k+1}_gamma", val=0.0, units='deg')
            self.add_output(f"platform_member{k+1}_t", val=np.zeros(n_height), units="m")
            self.add_output(f"platform_member{k+1}_rho_shell", val=0.0, units="kg/m**3")
            self.add_output(f"platform_member{k+1}_l_fill", val=np.zeros(n_height-1), units="m")
            self.add_output(f"platform_member{k+1}_rho_fill", val=np.zeros(n_height-1), units="kg/m**3")
            self.add_output(f"platform_member{k+1}_cap_d_in", val=np.zeros(n_bulk), units='m')
            self.add_output(f"platform_member{k+1}_ring_spacing", val=0.0)
            self.add_output(f"platform_member{k+1}_ring_t", val=0.0, units="m")
            self.add_output(f"platform_member{k+1}_ring_h", val=0.0, units="m")
            self.add_output(f"platform_member{k+1}_Cd", val=np.zeros(n_height))
            self.add_output(f"platform_member{k+1}_Ca", val=np.zeros(n_height))
            self.add_output(f"platform_member{k+1}_CdEnd", val=np.zeros(n_height))
            self.add_output(f"platform_member{k+1}_CaEnd", val=np.zeros(n_height))
            self.add_discrete_output(f"platform_member{k+1}_potMod", val=False)

        # Mooring inputs
        self.add_input('mooring_nodes', val=np.zeros((n_nodes, 3)), units='m', desc='Mooring node locations in global xyz')
        for k in range(n_nodes):
            self.add_discrete_output(f'mooring_point{k+1}_name', val=f'line{k+1}', desc='Mooring point identifier')
            self.add_discrete_output(f'mooring_point{k+1}_type', val='fixed', desc='Mooring connection type')
            self.add_output(f'mooring_point{k+1}_location', val=np.zeros(3), units='m', desc='Mooring node location')

        for k in range(n_lines):
            self.add_discrete_output(f'mooring_line{k+1}_endA', val='default', desc='End A coordinates')
            self.add_discrete_output(f'mooring_line{k+1}_endB', val='default', desc='End B coordinates')
            self.add_discrete_output(f'mooring_line{k+1}_type', val='mooring_line_type1', desc='Mooring line type')
            self.add_output(f'mooring_line{k+1}_length', val=0.0, units='m', desc='Length of line')

        self.add_input("unstretched_length", val=np.zeros(n_lines), units="m")
        self.add_input("line_diameter", val=np.zeros(n_lines), units="m")
        self.add_input("line_mass_density", val=np.zeros(n_lines), units="kg/m")
        self.add_input("line_stiffness", val=np.zeros(n_lines), units="N")
        self.add_input("line_breaking_load", val=np.zeros(n_lines), units="N")
        self.add_input("line_cost_rate", val=np.zeros(n_lines), units="USD/m")
        self.add_input("line_transverse_added_mass", val=np.zeros(n_lines), units="kg/m")
        self.add_input("line_tangential_added_mass", val=np.zeros(n_lines), units="kg/m")
        self.add_input("line_transverse_drag", val=np.zeros(n_lines))
        self.add_input("line_tangential_drag", val=np.zeros(n_lines))

        for k in range(n_line_types):
            self.add_discrete_output(f'mooring_line_type{k+1}_name', val='default', desc='Name of line type')
            self.add_output(f'mooring_line_type{k+1}_diameter', val=0.0, units='m', desc='Diameter of mooring line type')
            self.add_output(f'mooring_line_type{k+1}_mass_density', val=0.0, units='kg/m**3', desc='Mass density of line type')
            self.add_output(f'mooring_line_type{k+1}_stiffness', val=0.0, desc='Stiffness of line type')
            self.add_output(f'mooring_line_type{k+1}_breaking_load', val=0.0, desc='Breaking load of line type')
            self.add_output(f'mooring_line_type{k+1}_cost', val=0.0, units='USD', desc='Cost of mooring line type')
            self.add_output(f'mooring_line_type{k+1}_transverse_added_mass', val=0.0, desc='Transverse added mass')
            self.add_output(f'mooring_line_type{k+1}_tangential_added_mass', val=0.0, desc='Tangential added mass')
            self.add_output(f'mooring_line_type{k+1}_transverse_drag', val=0.0, desc='Transverse drag')
            self.add_output(f'mooring_line_type{k+1}_tangential_drag', val=0.0, desc='Tangential drag') 

        # Frequencies to calculate
        self.add_output('frequency_range', val=np.zeros(nfreq), units='Hz', desc='Frequency range to compute response over')
    
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        opt = self.options['modeling_options']

        # Frequencies to calculate
        outputs['frequency_range'] = np.array( opt['Level1']['frequencies'] )

        # Tower layer sections
        outputs['turbine_tower_t'] = inputs['tower_layer_thickness'].sum(axis=0)
        outputs['turbine_tower_rho_shell'] = inputs['tower_rho'].mean()
        k_tow_tor = inputs['tower_torsional_stiffness'] / inputs['tower_section_height']
        outputs['turbine_yaw_stiffness'] = 1.0/np.sum(1.0/k_tow_tor)

        # Tower drag
        Re = float(inputs['rho_air'])*inputs['tower_U']*inputs['turbine_tower_d']/float(inputs['mu_air'])
        cd, _ = cylinderDrag(Re)
        outputs['turbine_tower_Cd'] = cd
        
        # Move tower-top MoI to hub height
        m_rna = float(inputs['turbine_mRNA'])
        I_rna = util.assembleI( inputs['rna_I_TT'] )
        # Vector from WISDEM tower-top c.s. to raft tower center-line at hub height c.s.
        r = np.r_[0.0, 0.0, float(inputs['drive_height'])]
        outputs['turbine_xCG_RNA'] = (inputs['rna_cm'] - r)[0]
        I_rna_raft = util.unassembleI( I_rna + m_rna * (np.dot(r, r) * np.eye(3) - np.outer(r, r)) )
        outputs['turbine_IxRNA'] = I_rna_raft[0]
        outputs['turbine_IrRNA'] = 0.5*(I_rna_raft[1] + I_rna_raft[2])

        # Floating member data structure transfer
        n_member = len(opt["floating"]["members"]["name"])
        var_height = inputs['member_variable_height']
        for k in range(n_member):
            discrete_outputs[f"platform_member{k+1}_potMod"] = opt["Level1"]["model_potential"][k]
            
            # Member thickness
            outputs[f"platform_member{k+1}_t"] = inputs[f"member{k}:layer_thickness"].sum(axis=0)
            outputs[f"platform_member{k+1}_rho_shell"] = inputs[f"member{k}:rho"].mean()

            # Ring stiffener discretization conversion
            if float(inputs[f"member{k}:ring_stiffener_spacing"]) > 0.0:
                outputs[f"platform_member{k+1}_ring_spacing"] = (inputs[f"member{k}:ring_stiffener_spacing"] /
                                                                 inputs[f"member{k}:height"])
                h_web = inputs[f"member{k}:ring_stiffener_web_height"]
                t_web = inputs[f"member{k}:ring_stiffener_web_thickness"]
                t_flange = inputs[f"member{k}:ring_stiffener_flange_thickness"]
                w_flange = inputs[f"member{k}:ring_stiffener_flange_width"]
                outputs[f"platform_member{k+1}_ring_h"] = h_web + t_flange
                outputs[f"platform_member{k+1}_ring_t"] = (t_web*h_web + w_flange*t_flange) / (h_web+t_flange)

            # Ballast discretization conversion
            s_grid = inputs[f"platform_member{k+1}_stations"]
            s_ballast = inputs[f"member{k}:ballast_grid"]
            rho_ballast = inputs[f"member{k}:ballast_density"]
            h_ballast = inputs[f"member{k}:ballast_height"]
            l_fill = np.zeros(s_grid.size-1)
            rho_fill = np.zeros(s_grid.size-1)
            for ii in range(s_ballast.shape[0]):
                iball = np.where(s_ballast[ii,0] >= s_grid)[0][0]
                rho_fill[iball] = rho_ballast[ii]
                l_fill[iball] = h_ballast[ii] if rho_ballast[ii] < 1100.0 else var_height[k]
            outputs[f"platform_member{k+1}_l_fill"] = l_fill
            outputs[f"platform_member{k+1}_rho_fill"] = rho_fill
            
        # Mooring
        for k in range(opt['mooring']['n_nodes']):
            discrete_outputs[f'mooring_point{k+1}_name'] = opt['mooring']['node_names'][k]
            discrete_outputs[f'mooring_point{k+1}_type'] = opt['mooring']['node_type'][k]
            outputs[f'mooring_point{k+1}_location'] = inputs['mooring_nodes'][k,:]

        for k in range(opt['mooring']['n_lines']):
            discrete_outputs[f'mooring_line{k+1}_endA'] = opt['mooring']["node1"][k]
            discrete_outputs[f'mooring_line{k+1}_endB'] = opt['mooring']["node2"][k]
            discrete_outputs[f'mooring_line{k+1}_type'] = opt['mooring']["line_type"][k]
            outputs[f'mooring_line{k+1}_length'] = inputs['unstretched_length'][k]

        for k in range(opt['mooring']['n_line_types']):
            discrete_outputs[f'mooring_line_type{k+1}_name'] = opt['mooring']["line_type_name"][k]
            outputs[f'mooring_line_type{k+1}_cost'] = inputs['line_cost_rate'][k]
            
            for var in ['diameter','mass_density','stiffness','breaking_load',
                        'transverse_added_mass','tangential_added_mass','transverse_drag','tangential_drag']:
                outputs[f'mooring_line_type{k+1}_{var}'] = inputs[f'line_{var}'][k]
                    
