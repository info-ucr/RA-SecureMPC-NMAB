import cvxpy as cp
import numpy as np
import os
import random
import torch
import time

start_time_global = time.time()

class RealizationGenerator:
    def __init__(self, N, M, K_Values, r_values, T, D):
        assert len(K_Values) == N and len(r_values) == N, "K_Values and r_values must match N in length"
        self.N = N
        self.M = M
        self.K_Values = K_Values
        self.r_values = r_values
        self.T = T
        self.D = D
        self.valid_rows = [self._generate_valid_rows(k, r) for k, r in zip(K_Values, r_values)]
        self.current_indices = [0] * N  # Initialize with the first combination index for each row

    def _generate_valid_rows(self, K, r):
        # Generate all possible rows of length M with binary values
        all_rows = np.array([list(np.binary_repr(i, width=self.M)) for i in range(2**self.M)], dtype=int)
        # Filter rows based on the given condition for specific K and r
        valid_rows = [row for row in all_rows if np.sum(row) == r * (K + self.T - 1) + 1 + self.D]
        return valid_rows

    def get_one_realization(self):
        # Check if any set of valid rows is empty or if any current index exceeds its corresponding valid_rows length
        if any(len(vr) == 0 for vr in self.valid_rows) or any(self.current_indices[i] >= len(self.valid_rows[i]) for i in range(self.N)):
            return None  # Not possible to generate more realizations or criteria cannot be met
        # Construct the matrix a from the current combination
        a = np.zeros((self.N, self.M), dtype=int)
        for i in range(self.N):
            a[i, :] = self.valid_rows[i][self.current_indices[i]]
        # Update the indices for the next call
        for i in range(self.N - 1, -1, -1):
            if self.current_indices[i] < len(self.valid_rows[i]) - 1:
                self.current_indices[i] += 1
                break
            else:
                self.current_indices[i] = 0  # Reset this index
                if i == 0:  # If this is the first row, all combinations are exhausted
                    self.current_indices = [len(vr) for vr in self.valid_rows]  # Signal exhaustion
        return a
    
def update_npz_file(npz_filename, new_arrays):
    if not os.path.exists(npz_filename):
        np.savez(npz_filename)
    # Load the existing .npz file
    with np.load(npz_filename, allow_pickle=True) as data:
        # Extract existing arrays into a dict, allowing updates
        existing_arrays = {key: data[key] for key in data.files}
    # Track whether any updates are made
    updates_made = False
    # Check and update or add new arrays
    for array_name, array_data in new_arrays.items():
        if array_name in existing_arrays:
            # Update the array
            existing_arrays[array_name] = array_data
            updates_made = True
        else:
            # Add the new array if it doesn't exist
            existing_arrays[array_name] = array_data
            updates_made = True
    # Save all arrays back into the .npz file if any updates were made
    if updates_made:
        np.savez(npz_filename, **existing_arrays)

def find_nearest_workers_binary(Distance, num_selection):
    worker_selection_binary = np.zeros_like(Distance, dtype=int)
    nearest_indices = np.argsort(Distance)[:num_selection]
    worker_selection_binary[nearest_indices] = 1
    return worker_selection_binary

def find_best_channel_workers_binary(Channel, num_selection):
    worker_selection_binary = np.zeros_like(Channel, dtype=int)
    best_channel_indices = np.argsort(Channel)[::-1][:num_selection]
    worker_selection_binary[best_channel_indices] = 1
    return worker_selection_binary

def find_powerful_workers_binary(CPU_frequency_worker, num_selection):
    worker_selection_binary = np.zeros_like(CPU_frequency_worker, dtype=int)
    powerul_worker_indices = np.argsort(CPU_frequency_worker)[::-1][:num_selection]
    worker_selection_binary[powerul_worker_indices] = 1
    return worker_selection_binary

def get_latency(num_worker, worker_selection_binary, eta_W, channel_gain, comp_overhead, comm_overhead, P_U_Watt, P_W_Watt, B):
    # latency
    t_Encode = np.zeros((num_worker))
    t_Comp = np.zeros((num_worker))
    t_Upload = np.zeros((num_worker))
    t_MAP_1 = comp_overhead['Map_1'] / f_C_user
    t_Mask = comp_overhead['Mask'] / f_C_user
    t_Decode = comp_overhead['Decode'] / f_C_user
    t_Map_2 = comp_overhead['Map_2'] / f_C_user
    t_final = comp_overhead['Final'] / f_C_user
    R_Multicast = B * np.log2(1 + np.ma.min(np.ma.masked_where(worker_selection_binary == 0, channel_gain)) * P_U_Watt / Noise_Watt)
    t_Multicast = comm_overhead['Multicast'] / R_Multicast
    for j in range(num_worker):
        t_Encode[j] = worker_selection_binary[j] * comp_overhead['Encode'] / f_C_worker[j]
        t_Comp[j] = worker_selection_binary[j] * comp_overhead['Comp'] / f_C_worker[j]
        if worker_selection_binary[j] == 1:
            R_Upload = eta_W[j] * B * np.log2(1 + channel_gain[j] * P_W_Watt[j] / Noise_Watt)
            t_Upload[j] = comm_overhead['Upload'] / R_Upload
    objective = np.max(np.multiply(worker_selection_binary, t_Multicast + t_Encode + t_Comp + t_Upload)) + t_Decode
    system_latency = t_MAP_1 + t_Mask + objective + t_Map_2 + t_final
    # computation latency
    comp_latency = {
        'Map_1': t_MAP_1,
        'Mask': t_Mask,
        'Encode': t_Encode,
        'Comp': t_Comp,
        'Decode': t_Decode,
        'Map_2': t_Map_2,
        'Final': t_final,
    }
    # communication latency
    comm_latency = {
        'Multicast': t_Multicast,
        'Upload': t_Upload,
    }
    return objective, system_latency, comp_latency, comm_latency

def initialize_solution_storage(num_episode, num_epoch, num_K, num_worker, mode):
    if mode == 1:
        solution = {
            'a': np.zeros((num_episode, num_epoch, num_K, num_worker), dtype=int),
            'eta_W': np.zeros((num_episode, num_epoch, num_K, num_worker))
        }
    elif mode == 2:
        solution = {
            'a': np.zeros((num_episode, num_epoch, num_worker), dtype=int),
            'eta_W': np.zeros((num_episode, num_epoch, num_worker))
        }
    else:
        raise ValueError("Invalid mode. Mode should be 1 or 2.")
    return solution

def initialize_latency_storage(num_episode, num_epoch, num_K, num_worker, mode=1):
    if mode == 1:
        latency = {
            'Objective': np.full((num_episode, num_epoch, num_K), np.inf),
            'System_Latency': np.full((num_episode, num_epoch, num_K+1), np.inf),
            'Map_1': np.zeros((num_episode, num_epoch, num_K)),
            'Mask': np.zeros((num_episode, num_epoch, num_K)),
            'Encode': np.zeros((num_episode, num_epoch, num_K, num_worker)),
            'Comp': np.zeros((num_episode, num_epoch, num_K, num_worker)),
            'Decode': np.zeros((num_episode, num_epoch, num_K)),
            'Map_2': np.zeros((num_episode, num_epoch, num_K)),
            'Final': np.zeros((num_episode, num_epoch, num_K)),
            'Multicast': np.zeros((num_episode, num_epoch, num_K, num_worker)),
            'Upload': np.zeros((num_episode, num_epoch, num_K, num_worker))
        }
    elif mode == 2:
        latency = {
            'Objective': np.full((num_episode, num_epoch), np.inf),
            'System_Latency': np.full((num_episode, num_epoch), np.inf),
            'Map_1': np.zeros((num_episode, num_epoch)),
            'Mask': np.zeros((num_episode, num_epoch)),
            'Encode': np.zeros((num_episode, num_epoch, num_worker)),
            'Comp': np.zeros((num_episode, num_epoch, num_worker)),
            'Decode': np.zeros((num_episode, num_epoch)),
            'Map_2': np.zeros((num_episode, num_epoch)),
            'Final': np.zeros((num_episode, num_epoch)),
            'Multicast': np.zeros((num_episode, num_epoch, num_worker)),
            'Upload': np.zeros((num_episode, num_epoch, num_worker))
        }
    else:
        raise ValueError("Invalid mode. Mode should be 1 or 2.")
    return latency

def update_solution(solution, worker_selection, bandwidth_allocation, episode, epoch, k, mode=1):
    if mode == 1:
        solution['a'][episode][epoch][k] = worker_selection
        solution['eta_W'][episode][epoch][k] = bandwidth_allocation
    elif mode == 2:
        solution['a'][episode][epoch] = worker_selection
        solution['eta_W'][episode][epoch] = bandwidth_allocation
    else:
        raise ValueError("Invalid mode. Mode should be 1 or 2.")
    return solution

def update_latency(latency, objective, system_latency, comp_latency, comm_latency, episode, epoch, k, mode=1):
    if mode == 1:
        latency['Objective'][episode][epoch][k] = objective
        latency['System_Latency'][episode][epoch][k] = system_latency
        latency['Map_1'][episode][epoch][k] = comp_latency['Map_1']
        latency['Mask'][episode][epoch][k] = comp_latency['Mask']
        latency['Encode'][episode][epoch][k] = comp_latency['Encode']
        latency['Comp'][episode][epoch][k] = comp_latency['Comp']
        latency['Decode'][episode][epoch][k] = comp_latency['Decode']
        latency['Map_2'][episode][epoch][k] = comp_latency['Map_2']
        latency['Final'][episode][epoch][k] = comp_latency['Final']
        latency['Multicast'][episode][epoch][k] = comm_latency['Multicast']
        latency['Upload'][episode][epoch][k] = comm_latency['Upload']
    elif mode == 2:
        latency['Objective'][episode][epoch] = objective
        latency['System_Latency'][episode][epoch] = system_latency
        latency['Map_1'][episode][epoch] = comp_latency['Map_1']
        latency['Mask'][episode][epoch] = comp_latency['Mask']
        latency['Encode'][episode][epoch] = comp_latency['Encode']
        latency['Comp'][episode][epoch] = comp_latency['Comp']
        latency['Decode'][episode][epoch] = comp_latency['Decode']
        latency['Map_2'][episode][epoch] = comp_latency['Map_2']
        latency['Final'][episode][epoch] = comp_latency['Final']
        latency['Multicast'][episode][epoch] = comm_latency['Multicast']
        latency['Upload'][episode][epoch] = comm_latency['Upload']
    else:
        raise ValueError("Invalid mode. Mode should be 1 or 2.")
    return latency

def update_name(dictionary, new_name):
    updated_dictionary = {}
    for key in dictionary:
        updated_key = key + new_name
        updated_dictionary[updated_key] = dictionary[key]
    return updated_dictionary

def num_cycle(field='Real', opera='Add', dtype='Float'):
    num_Cycle_Modulo = 40
    cycle_times = {
        'Real': {
            'Float': {'Add': 3, 'Sub': 3, 'Mul': 5, 'Div': 15, 'Cmp': 3, 'Round': 21},
            'Int': {'Add': 1, 'Sub': 1, 'Mul': 3, 'Div': 40, 'Cmp': 1}
        },
        'FF': {
            'Add': None,  # Placeholder
            'Sub': None,  # Placeholder
            'Mul': None   # Placeholder
        }
    }

    if field == 'FF':
        # Use Int values from Real and add num_Cycle_Modulo
        int_cycles = cycle_times['Real']['Int']
        return int_cycles.get(opera, None) + num_Cycle_Modulo if opera in int_cycles else None

    # Default behavior for 'Real'
    return cycle_times.get(field, {}).get(dtype, {}).get(opera, None)

def get_overhead_Local(l):

    # Computing
    Lambda_Comp_PiNet = l * (
        # Convolution Layer 1
        16 * 16 * 64 * 147 * num_cycle(field='Real', opera='Mul', dtype='Float') +
        16 * 16 * 64 * 146 * num_cycle(field='Real', opera='Add', dtype='Float') +
        # Convolution Layer 2
        16 * 16 * 64 * 576 * num_cycle(field='Real', opera='Mul', dtype='Float') +
        16 * 16 * 64 * 575 * num_cycle(field='Real', opera='Add', dtype='Float') +
        # Convolution Layer 3
        16 * 16 * 64 * 576 * num_cycle(field='Real', opera='Mul', dtype='Float') +
        16 * 16 * 64 * 575 * num_cycle(field='Real', opera='Add', dtype='Float') +
        # Hadamard Product
        16 * 16 * 64 * num_cycle(field='Real', opera='Mul', dtype='Float') +
        # Summation
        16 * 16 * 64 * 2 * num_cycle(field='Real', opera='Add', dtype='Float') +
        # Fully Connected Layer
        10 * 16384 * num_cycle(field='Real', opera='Mul', dtype='Float') +
        (10 * (16384 - 1) + 10) * num_cycle(field='Real', opera='Add', dtype='Float')
    )
    # Classification
    Lambda_final = (10 - 1) * l * num_cycle(field='Real', opera='Cmp', dtype='Float')
    Lambda_Comp = Lambda_Comp_PiNet + Lambda_final
    
    return Lambda_Comp

Dataset = np.load('Dataset.npz')

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
gpu = 0
device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')

# environment
# number of workers
M = Dataset['M'].item()
# Privacy degree
T = Dataset['T'].item()
# polynomial degree
r = Dataset['r'].item()
# local dataset size
l = Dataset['l'].item()
# number of dropout workers
D = Dataset['D'].item()
# transmit power for broadcasting of users (dBm)
P_U_dBm = Dataset['P_U_dBm'].item()
P_U_Watt = Dataset['P_U_Watt'].item()
# maximum transmit power for worker (dBm)
P_W_dBm = Dataset['P_W_dBm']
P_W_Watt = Dataset['P_W_Watt']
# bandwidth (MHz)
B = Dataset['B'].item()
# noise power (dBm)
Noise_dBm = Dataset['Noise_dBm'].item()
Noise_Watt = Dataset['Noise_Watt'].item()
# computation capacity of worker (MHz)
f_C_user = Dataset['f_C_user'].item()
f_C_worker = Dataset['f_C_worker']

K_possible = Dataset['K_possible']

name_txt = 'Baselines.txt'

# train

name_npz_Train = 'Baselines.npz'

Num_Episode = Dataset['Num_Episode_Train'].item()
# Num_Epoch = Dataset['Num_Epoch_Train'].item()
Num_Epoch = 1

Distance = Dataset['Distance_Train']
Channel_Gain = Dataset['Channel_Gain_Train']

# computation overhead
# data preprocessing
Lambda_Map_1 = Dataset['Lambda_Map_1_Train']
# masking
Lambda_Mask = Dataset['Lambda_Mask_Train']
# encode
Lambda_Encode = Dataset['Lambda_Encode_Train']
# computing
Lambda_Comp = Dataset['Lambda_Comp_Train']
# decode
Lambda_Decode = Dataset['Lambda_Decode_Train']
# result postprocessing
Lambda_Map_2 = Dataset['Lambda_Map_2_Train']
# final result
Lambda_Final = Dataset['Lambda_Final_Train']

# communication overhead
# multicast
Lambda_Multicast = Dataset['Lambda_Multicast_Train']
# upload
Lambda_Upload = Dataset['Lambda_Upload_Train']

# Exhaustive Search
Solution_ES_Each_K_1 = initialize_solution_storage(Num_Episode, Num_Epoch, K_possible.shape[0], M, mode=1)
Latency_ES_Each_K_1 = initialize_latency_storage(Num_Episode, Num_Epoch, K_possible.shape[0], M, mode=1)

# Shamir's Secret Sharing, nearest
Solution_SSS_1_1 = initialize_solution_storage(Num_Episode, Num_Epoch, K_possible.shape[0], M, mode=2)
Latency_SSS_1_1 = initialize_latency_storage(Num_Episode, Num_Epoch, K_possible.shape[0], M, mode=2)

# Shamir's Secret Sharing, best channel
Solution_SSS_2_1 = initialize_solution_storage(Num_Episode, Num_Epoch, K_possible.shape[0], M, mode=2)
Latency_SSS_2_1 = initialize_latency_storage(Num_Episode, Num_Epoch, K_possible.shape[0], M, mode=2)

# Baseline 4 (Each K, best channel)
Solution_Baseline4_1 = initialize_solution_storage(Num_Episode, Num_Epoch, K_possible.shape[0], M, mode=1)
Latency_Baseline4_1 = initialize_latency_storage(Num_Episode, Num_Epoch, K_possible.shape[0], M, mode=1)

# Local
Latency_Local = {
    'System_Latency': np.zeros((Num_Episode, Num_Epoch))
    }

start_time = time.time()

for episode in range(Num_Episode):
    
    start_time_episode = time.time()
    
    distance = Distance[episode].flatten()
    
    for epoch in range(Num_Epoch):
        
        start_time_epoch = time.time()

        channel_gain = Channel_Gain[episode][epoch].flatten()

        # Local
        Latency_Local['System_Latency'][episode][epoch] = get_overhead_Local(l) / f_C_user

        # Exhaustive Search
        for k in range(K_possible.shape[0]):
            K = K_possible[k]
            generator = RealizationGenerator(1, M, [K], [r], T, D)
            # computation overhead
            comp_overhead = {
                'Map_1': Lambda_Map_1[episode][epoch][k],
                'Mask': Lambda_Mask[episode][epoch][k],
                'Encode': Lambda_Encode[episode][epoch][k],
                'Comp': Lambda_Comp[episode][epoch][k],
                'Decode': Lambda_Decode[episode][epoch][k],
                'Map_2': Lambda_Map_2[episode][epoch][k],
                'Final': Lambda_Final[episode][epoch][k]
            }
            # communication overhead
            comm_overhead = {
                'Multicast': Lambda_Multicast[episode][epoch][k],
                'Upload': Lambda_Upload[episode][epoch][k]
            }
            while True:
                a = generator.get_one_realization()
                if a is not None:
                    a = a.flatten()
                    eta_W = a / np.sum(a)
                    objective_ES_1, system_latency_ES_1, comp_latency_ES_1, comm_latency_ES_1 = get_latency(M, a, eta_W, channel_gain, comp_overhead, comm_overhead, P_U_Watt, P_W_Watt, B)
                    if system_latency_ES_1 < Latency_ES_Each_K_1['System_Latency'][episode][epoch][k]: 
                        Solution_ES_Each_K_1 = update_solution(Solution_ES_Each_K_1, a, eta_W, episode, epoch, k, mode=1)
                        Latency_ES_Each_K_1 = update_latency(Latency_ES_Each_K_1, objective_ES_1, system_latency_ES_1, comp_latency_ES_1, comm_latency_ES_1, episode, epoch, k, mode=1)
                else:
                    break
        
        K = 1
        k = 0
        # computation overhead
        comp_overhead = {
            'Map_1': Lambda_Map_1[episode][epoch][K-1],
            'Mask': Lambda_Mask[episode][epoch][K-1],
            'Encode': Lambda_Encode[episode][epoch][K-1],
            'Comp': Lambda_Comp[episode][epoch][K-1],
            'Decode': Lambda_Decode[episode][epoch][K-1],
            'Map_2': Lambda_Map_2[episode][epoch][K-1],
            'Final': Lambda_Final[episode][epoch][K-1]
        }
        # communication overhead
        comm_overhead = {
            'Multicast': Lambda_Multicast[episode][epoch][K-1],
            'Upload': Lambda_Upload[episode][epoch][K-1]
        }

        a = find_best_channel_workers_binary(channel_gain, r*(K+T-1)+1)
        # Shamir's Secret Sharing, best channel, equal resource allocation
        eta_W = a / np.sum(a)
        objective_SSS_2_1, system_latency_SSS_2_1, comp_latency_SSS_2_1, comm_latency_SSS_2_1 = get_latency(M, a, eta_W, channel_gain, comp_overhead, comm_overhead, P_U_Watt, P_W_Watt, B)
        Solution_SSS_2_1 = update_solution(Solution_SSS_2_1, a, eta_W, episode, epoch, k, mode=2)
        Latency_SSS_2_1 = update_latency(Latency_SSS_2_1, objective_SSS_2_1, system_latency_SSS_2_1, comp_latency_SSS_2_1, comm_latency_SSS_2_1, episode, epoch, k, mode=2)
        
        # Baseline 4 (Each K, best channel)
        for k in range(K_possible.shape[0]):
            K = K_possible[k]
            a = find_best_channel_workers_binary(channel_gain, r*(K+T-1)+1)
            eta_W = a / np.sum(a)
            # computation overhead
            comp_overhead = {
                'Map_1': Lambda_Map_1[episode][epoch][k],
                'Mask': Lambda_Mask[episode][epoch][k],
                'Encode': Lambda_Encode[episode][epoch][k],
                'Comp': Lambda_Comp[episode][epoch][k],
                'Decode': Lambda_Decode[episode][epoch][k],
                'Map_2': Lambda_Map_2[episode][epoch][k],
                'Final': Lambda_Final[episode][epoch][k]
            }
            # communication overhead
            comm_overhead = {
                'Multicast': Lambda_Multicast[episode][epoch][k],
                'Upload': Lambda_Upload[episode][epoch][k]
            }
            objective_Baseline4_1, system_latency_Baseline4_1, comp_latency_Baseline4_1, comm_latency_Baseline4_1 = get_latency(M, a, eta_W, channel_gain, comp_overhead, comm_overhead, P_U_Watt, P_W_Watt, B)
            Solution_Baseline4_1 = update_solution(Solution_Baseline4_1, a, eta_W, episode, epoch, k, mode=1)
            Latency_Baseline4_1 = update_latency(Latency_Baseline4_1, objective_Baseline4_1, system_latency_Baseline4_1, comp_latency_Baseline4_1, comm_latency_Baseline4_1, episode, epoch, k, mode=1)
        
        end_time_epoch = time.time()
    
    end_time_episode = time.time()
    print(f"Episode: {episode+1},")
    print(f"Time cost: {end_time_episode - start_time_episode:.5f},")
    print('-'*30)
    with open(name_txt, "a") as file:
        file.write(f"Episode: {episode+1},\n")
        file.write(f"Time cost: {end_time_episode - start_time_episode:.5f},\n")
        file.write("-" * 30 + "\n")

    if (episode+1) % 1000 == 0:

        # Local
        update_npz_file(name_npz_Train, update_name(Latency_Local, 'Local'))

        # Exhaustive Search
        update_npz_file(name_npz_Train, update_name(Solution_ES_Each_K_1, '_ES_Each_K_1'))
        update_npz_file(name_npz_Train, update_name(Latency_ES_Each_K_1, '_ES_Each_K_1'))
                         
        # Shamir's Secret Sharing, best channel
        update_npz_file(name_npz_Train, update_name(Solution_SSS_2_1, '_SSS_2_1'))
        update_npz_file(name_npz_Train, update_name(Latency_SSS_2_1, '_SSS_2_1'))
                         
        # Baseline 4 (Each K, best channel)
        update_npz_file(name_npz_Train, update_name(Solution_Baseline4_1, '_Baseline4_1'))
        update_npz_file(name_npz_Train, update_name(Latency_Baseline4_1, '_Baseline4_1'))

end_time = time.time()
print(f"{'Train time cost'}: {end_time - start_time:.5f},")
print('-'*30)
with open(name_txt, "a") as file:
    file.write(f"{'Train time cost'}: {end_time - start_time:.5f},\n")
    file.write("-" * 30 + "\n")

end_time_global = time.time()
print(f"{'Total time cost'}: {end_time_global - start_time_global:.5f},")
with open(name_txt, "a") as file:
    file.write(f"{'Total time cost'}: {end_time_global - start_time_global:.5f},\n")
    file.write("-" * 30 + "\n")