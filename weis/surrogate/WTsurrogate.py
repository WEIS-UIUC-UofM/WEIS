import numpy as np
import csv
import os
import openmdao.api as om
from wisdem.commonse.mpi_tools import MPI
from smt.surrogate_models import KRG

class WindTurbineDOE2SM():

    def __init__(self):
        pass

    def read_doe(self, sql_file, modeling_options, opt_options):

        if MPI:
            rank = MPI.COMM_WORLD.Get_rank()
        else:
            rank = 0

        cr = om.CaseReader(sql_file)
        cases = cr.list_cases('driver')

        if (not MPI) or (MPI and rank == 0):
            case = cases[0]
            inputs = cr.get_case(case).inputs
            outputs = cr.get_case(case).outputs
            input_keys_ref = list(set(inputs.keys()).intersection([
                'floating.member_main_column:outer_diameter',
                'floating.member_column1:outer_diameter',
                'floating.member_column2:outer_diameter',
                'floating.member_column3:outer_diameter',
                'floating.member_Y_pontoon_upper1:outer_diameter',
                'floating.member_Y_pontoon_upper2:outer_diameter',
                'floating.member_Y_pontoon_upper3:outer_diameter',
                'floating.member_Y_pontoon_lower1:outer_diameter',
                'floating.member_Y_pontoon_lower2:outer_diameter',
                'floating.member_Y_pontoon_lower3:outer_diameter',
            ]))
            input_keys_ref.sort()
            input_keys_dv = self._identify_dv(input_keys_ref, opt_options, inputs, outputs)
            output_keys_ref = list(set(outputs.keys()).intersection([
                'tune_rosco_ivc.ps_percent',
                'tune_rosco_ivc.omega_pc',
                'tune_rosco_ivc.zeta_pc',
                'tune_rosco_ivc.Kp_float',
                'tune_rosco_ivc.ptfm_freq',
                'tune_rosco_ivc.omega_vs',
                'tune_rosco_ivc.zeta_vs',
                'configuration.rotor_diameter_user',
                'towerse.tower.fore_aft_freqs', # 123
                'towerse.tower.side_side_freqs', # 123
                'towerse.tower.torsion_freqs', # 123
                'towerse.tower.top_deflection',
                'floatingse.platform_base_F', # xyz
                'floatingse.platform_base_M', # xyz
                'floating.member_main_column:joint1', # xyz
                'floating.member_main_column:joint2', # xyz
                'floating.member_column1:joint1', # xyz
                'floating.member_column1:joint2', # xyz
                'floating.member_column2:joint1', # xyz
                'floating.member_column2:joint2', # xyz
                'floating.member_column3:joint1', # xyz
                'floating.member_column3:joint2', # xyz
                'floating.jointdv_0', # keel z-location
                'floating.jointdv_1', # freeboard z-location
                'floating.jointdv_2', # column123 r-location
                'raft.Max_Offset', # Maximum distance in surge/sway direction [m]
                'raft.heave_avg', # Average heave over all cases [m]
                'raft.Max_PtfmPitch', # Maximum platform pitch over all cases [deg]
                'raft.Std_PtfmPitch', # Average platform pitch std. over all cases [deg]
                'rigid_body_periods', # Rigid body natural period [s]
                'raft.heave_period', # Heave natural period [s]
                'raft.pitch_period', # Pitch natural period [s]
                'raft.roll_period', # Roll natural period [s]
                'raft.surge_period', # Surge natural period [s]
                'raft.sway_period', # Sway natural period [s]
                'raft.yaw_period', # Yaw natural period [s]
                'raft.max_nac_accel', # Maximum nacelle accelleration over all cases [m/s**2]
                'raft.max_tower_base', # Maximum tower base moment over all cases [N*m]
                'raft.platform_total_center_of_mass', # xyz
                'raft.platform_displacement',
                'raft.platform_mass', # Platform mass
                'tcons.tip_deflection_ratio', # Blade tip deflection ratio (constrained to be <=1.0)
                'financese.lcoe', # WEIS LCOE from FinanceSE
                'rotorse.rp.AEP', # WISDEM AEP from RotorSE
                'rotorse.blade_mass', # Blade mass
                #'towerse.tower_mass', # Tower mass
                'fixedse.structural_mass', # System structural mass for fixed foundation turbines
                'floatingse.system_structural_mass', # System structural mass for floating turbines
                'floatingse.platform_mass', # Platform mass from FloatingSE
                'floatingse.platform_cost', # Platform cost
                #'floatingse.mooring_mass', # Mooring mass
                #'floatingse.mooring_cost', # Mooring cost
                'floatingse.structural_frequencies', 
            ]))
            output_keys_ref.sort()
            output_keys_dv = self._identify_dv(output_keys_ref, opt_options, inputs, outputs)
        else:
            input_keys_ref = None
            input_keys_dv = None
            output_keys_ref = None
            output_keys_dv = None
        if MPI:
            input_keys_ref = MPI.COMM_WORLD.bcast(input_keys_ref, root=0)
            input_keys_dv = MPI.COMM_WORLD.bcast(input_keys_dv, root=0)
            output_keys_ref = MPI.COMM_WORLD.bcast(output_keys_ref, root=0)
            output_keys_dv = MPI.COMM_WORLD.bcast(output_keys_dv, root=0)

        # Retrieve values to construct dataset list
        dataset = []

        for case in cases:
            inputs = cr.get_case(case).inputs
            outputs = cr.get_case(case).outputs
            var_keys = []
            var_dv = []
            var_values = []
            for key in input_keys_ref:
                if len(inputs[key]) == 1:
                    var_keys.append(key)
                    try:
                        dvidx = input_keys_dv[input_keys_ref.index(key)]
                        if ((type(dvidx) == bool) and (dvidx == False)) or \
                                ((type(dvidx) == list) and (len(dvidx) == 0)):
                            var_dv.append(False)
                        elif ((type(dvidx) == list) and (len(dvidx) == 1)):
                            var_dv.append(True)
                        else:
                            raise Exception
                    except:
                        var_dv.append(False)
                    var_values.append(inputs[key][0])
                else:
                    for idx in range(len(np.squeeze(inputs[key]))):
                        var_keys.append(key + '_' + str(idx))
                        try:
                            dvidx = input_keys_dv[input_keys_ref.index(key)]
                            if ((type(dvidx) == bool) and (dvidx == False)) or \
                                    ((type(dvidx) == list) and (len(dvidx) == 0)):
                                var_dv.append(False)
                            elif ((type(dvidx) == list) and (len(dvidx) > 0)):
                                for jdx in range(len(dvidx)):
                                    var_dv_append = False
                                    if idx == dvidx[jdx]:
                                        var_dv_append = True
                                var_dv.append(var_dv_append)
                            else:
                                raise Exception
                        except:
                            var_dv.append(False)
                        var_values.append(np.squeeze(inputs[key])[idx])
            for key in output_keys_ref:
                if len(outputs[key]) == 1:
                    var_keys.append(key)
                    try:
                        dvidx = output_keys_dv[output_keys_ref.index(key)]
                        if ((type(dvidx) == bool) and (dvidx == False)) or \
                                ((type(dvidx) == list) and (len(dvidx) == 0)):
                            var_dv.append(False)
                        elif ((type(dvidx) == list) and (len(dvidx) == 1)):
                            var_dv.append(True)
                        else:
                            raise Exception
                    except:
                        var_dv.append(False)
                    var_values.append(outputs[key][0])
                else:
                    for idx in range(len(np.squeeze(outputs[key]))):
                        var_keys.append(key + '_' + str(idx))
                        try:
                            dvidx = output_keys_dv[output_keys_ref.index(key)]
                            if ((type(dvidx) == bool) and (dvidx == False)) or \
                                    ((type(dvidx) == list) and (len(dvidx) == 0)):
                                var_dv.append(False)
                            elif ((type(dvidx) == list) and (len(dvidx) > 0)):
                                for jdx in range(len(dvidx)):
                                    var_dv_append = False
                                    if idx == dvidx[jdx]:
                                        var_dv_append = True
                                var_dv.append(var_dv_append)
                            else:
                                raise Exception
                        except:
                            var_dv.append(False)
                        var_values.append(np.squeeze(outputs[key])[idx])
            dataset.append(var_values)

        # Gather data values
        if MPI:
            dataset = MPI.COMM_WORLD.gather(dataset, root=0)
        if rank == 0:
            dataset = np.array([dp for proc in dataset for dp in proc])
            # Remove duplicated columns
            flag = dataset.shape[1]*[True]
            for idx in range(1,dataset.shape[1]):
                for jdx in range(idx):
                    if (np.array_equal(dataset[:,jdx], dataset[:,idx])):
                        flag[idx] = False
                        if flag[jdx] == True:
                            var_keys[jdx] += ('+' + var_keys[idx])
            data_vals = np.zeros(shape=(dataset.shape[0],0))
            data_dv = []
            data_keys = []
            for idx in range(dataset.shape[1]):
                if flag[idx]:
                    data_vals = np.concatenate(
                            (data_vals, dataset[:,idx].reshape(len(dataset[:,idx]),1)),
                            axis=1)
                    data_dv.append(var_dv[idx])
                    data_keys.append(var_keys[idx])
        else:
            dataset = None
            data_vals = None
            data_dv = None
            data_keys = None

        # Store data
        self.data_vals = data_vals
        self.data_dv = data_dv
        self.data_keys = data_keys

        if rank==0:
            with open(os.path.join(opt_options['general']['folder_output'],
                    opt_options['general']['fname_output']+'_doedata.csv'), 'wt', newline='') as fid:
                dwriter = csv.writer(fid, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                dwriter.writerow(data_keys)
                dwriter.writerow(data_dv)
                for idx in range(data_vals.shape[0]):
                    dwriter.writerow(data_vals[idx,:])


    def _identify_dv(self, keys, opt_options, inputs, outputs):

        dvflag = len(keys)*[False]

        if opt_options['design_variables']['control']['ps_percent']['flag']:
            for key in ['tune_rosco_ivc.ps_percent']:
                try:
                    idx = keys.index(key)
                    if not dvflag[idx]:
                        dvflag[idx] = [0]
                    else:
                        dvflag[idx].append(0)
                except:
                    pass

        if opt_options['design_variables']['control']['servo']['pitch_control']['omega']['flag']:
            for key in ['tune_rosco_ivc.omega_pc']:
                try:
                    idx = keys.index(key)
                    if not dvflag[idx]:
                        dvflag[idx] = [0]
                    else:
                        dvflag[idx].append(0)
                except:
                    pass

        if opt_options['design_variables']['control']['servo']['pitch_control']['zeta']['flag']:
            for key in ['tune_rosco_ivc.zeta_pc']:
                try:
                    idx = keys.index(key)
                    if not dvflag[idx]:
                        dvflag[idx] = [0]
                    else:
                        dvflag[idx].append(0)
                except:
                    pass

        if opt_options['design_variables']['control']['servo']['pitch_control']['Kp_float']['flag']:
            for key in ['tune_rosco_ivc.Kp_float']:
                try:
                    idx = keys.index(key)
                    if not dvflag[idx]:
                        dvflag[idx] = [0]
                    else:
                        dvflag[idx].append(0)
                except:
                    pass

        if opt_options['design_variables']['control']['servo']['pitch_control']['ptfm_freq']['flag']:
            for key in ['tune_rosco_ivc.ptfm_freq']:
                try:
                    idx = keys.index(key)
                    if not dvflag[idx]:
                        dvflag[idx] = [0]
                    else:
                        dvflag[idx].append(0)
                except:
                    pass

        if opt_options['design_variables']['control']['servo']['torque_control']['omega']['flag']:
            for key in ['tune_rosco_ivc.omega_vs']:
                try:
                    idx = keys.index(key)
                    if not dvflag[idx]:
                        dvflag[idx] = [0]
                    else:
                        dvflag[idx].append(0)
                except:
                    pass

        if opt_options['design_variables']['control']['servo']['torque_control']['zeta']['flag']:
            for key in ['tune_rosco_ivc.zeta_vs']:
                try:
                    idx = keys.index(key)
                    if not dvflag[idx]:
                        dvflag[idx] = [0]
                    else:
                        dvflag[idx].append(0)
                except:
                    pass

        if opt_options['design_variables']['floating']['members']['flag']:
            groups = opt_options['design_variables']['floating']['members']['groups']
            for group in groups:
                names = group['names']
                for key in keys:
                    for name in names:
                        txt = name + ':outer_diameter'
                        ltxt = len(txt)
                        if key[-ltxt:] == txt:
                            try:
                                idx = keys.index(key)
                                if group['diameter']['constant']:
                                    jdx = 0
                                else:
                                    jdx = True
                                if not dvflag[idx]:
                                    dvflag[idx] = [jdx]
                                else:
                                    dvflag[idx].append(jdx)
                            except:
                                pass

        if opt_options['design_variables']['floating']['joints']['flag']:
            key_prefix = 'floating.jointdv_'
            for key in keys:
                if key[:len(key_prefix)] == key_prefix:
                    try:
                        idx = keys.index(key)
                        if not dvflag[idx]:
                            dvflag[idx] = [0]
                        else:
                            dvflag[idx].append(0)
                    except:
                        pass

        if opt_options['design_variables']['rotor_diameter']['flag']:
            for key in ['configuration.rotor_diameter_user']:
                try:
                    idx = keys.index(key)
                    if not dvflag[idx]:
                        dvflag[idx] = [0]
                    else:
                        dvflag[idx].append(0)
                except:
                    pass

        return dvflag


    def train_SM(self):

        if MPI:
            rank = MPI.COMM_WORLD.Get_rank()
            n_cores = MPI.COMM_WORLD.Get_size()
        else:
            rank = 0
            n_cores = 1

        # Prepare dataset columns individually (and split if MPI)
        if (not MPI) or (MPI and rank == 0):
            data_vals = self.data_vals
            data_dv = self.data_dv
            data_keys = self.data_keys

            dataset_list = []
            for idx in range(len(data_keys)):
                data_entry = {
                    'key': data_keys[idx],
                    'dv': data_dv[idx],
                    'val': data_vals[:, idx],
                }
                dataset_list.append(data_entry)

            dataset_lists = list(self._split_list_chunks(dataset_list, n_cores))
            if len(dataset_lists) < n_cores:
                for _ in range(0, n_cores - len(dataset_lists)):
                    dataset_lists.append([])
        else:
            dataset_list = []
            dataset_lists = []

        # Scatter data to train
        if MPI:
            MPI.COMM_WORLD.barrier()
            dataset_list = MPI.COMM_WORLD.scatter(dataset_lists, root=0)

        




    def _split_list_chunks(self, fulllist, max_n_chunk=1, item_count=None):
        item_count = item_count or len(fulllist)
        n_chunks = min(item_count, max_n_chunk)
        fulllist = iter(fulllist)
        floor = item_count // n_chunks
        ceiling = floor + 1
        stepdown = item_count % n_chunks
        for x_i in range(n_chunks):
            length = ceiling if x_i < stepdown else floor
            yield [next(fulllist) for _ in range(length)]

