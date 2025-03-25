import numpy as np
import os
import random
import time
import itertools

start_time_global = time.time()

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

def get_overhead(l):

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

def get_worker_latency(comp_latency, comm_latency):
    return comm_latency['Multicast'] + comp_latency['Encode'] + comp_latency['Comp'] + comm_latency['Upload']

def get_reward(objective):
    return np.exp(- objective/1e8 * 0.75e-2)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_storage(num_episode, num_epoch, num_worker):
    solution = {
        'K': np.zeros((num_episode, num_epoch), dtype=int),
        'a': np.zeros((num_episode, num_epoch, num_worker), dtype=int),
        'eta_W': np.zeros((num_episode, num_epoch, num_worker))
    }
    latency = {
        'Objective': np.zeros((num_episode, num_epoch)),
        'System_Latency': np.zeros((num_episode, num_epoch)),
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
    return solution, latency

def update_solution(solution, K, worker_selection, bandwidth_allocation, episode, epoch):
    solution['K'][episode][epoch] = K
    solution['a'][episode][epoch] = worker_selection
    solution['eta_W'][episode][epoch] = bandwidth_allocation
    return solution

def update_latency(latency, objective, system_latency, comp_latency, comm_latency, episode, epoch):
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
    return latency

def update_name(dictionary, new_name_1, new_name_2=None):
    if new_name_2 is None:
        updated_dictionary = {}
        for key in dictionary:
            updated_key = key + new_name_1
            updated_dictionary[updated_key] = dictionary[key]
    else:
        updated_dictionary = {}
        for key in dictionary:
            if key == 'Objective' or key == 'System_Latency':
                updated_key = key + new_name_2
            else:
                updated_key = new_name_1 + key + new_name_2
            updated_dictionary[updated_key] = dictionary[key]
    return updated_dictionary

def z_score_normalize_3d(data):
    for i in range(data.shape[0]):
        mean = np.mean(data[i])
        std_dev = np.std(data[i])
        data[i] = (data[i] - mean) / std_dev
    return data

def generate_valid_combinations(M, K, r, T):
    """
    Generate valid combinations of one-hot vectors for values in K and binary vectors of length M.
    Filter out combinations where the sum of the binary vector is less than r*(K+T-1)+1.
    Return the combinations with an additional dimension for easy batching in further processing.
    Parameters:
    - M (int): Length of the binary vector a.
    - K (list): List of possible values for K (one-hot encoded).
    - r (int): Constant multiplier for the condition.
    - T (int): Constant added to K in the condition.
    Returns:
    - np.array: An array of valid combinations, each combination in its own subarray,
                shape (num_valid_combinations, 1, len(K) + M)
    """
    # Generate all possible values of a
    a_values = list(itertools.product([0, 1], repeat=M))
    # Generate one-hot vectors for K
    one_hot_vectors = np.eye(len(K))
    # Filter combinations
    valid_combinations = []
    for k_index, k in enumerate(K):
        one_hot = one_hot_vectors[k_index]
        for a in a_values:
            if sum(a) == r * (k + T - 1) + 1:
                combined_vector = np.concatenate((one_hot, a))
                valid_combinations.append(combined_vector)
    # Convert list to NumPy array and reshape
    valid_combinations_array = np.array(valid_combinations).reshape(-1, 1, len(K) + M)

    return valid_combinations_array

Dataset = np.load('Dataset.npz')

seed_Train = 0
seed_Test = 0

random.seed(seed_Train)
np.random.seed(seed_Train)

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

Method = ['MAB', 'LMAB']

num_state = 1 + M + M + K_possible.shape[0]
num_action = 1

name_txt = 'Bandit.txt'

# train

Num_Episode_Train = Dataset['Num_Episode_Train'].item()
# Num_Epoch_Train = Dataset['Num_Epoch_Train'].item()
# Num_Episode_Train = 10000
Num_Epoch_Train = 1
Episode_Train = np.arange(Num_Episode_Train,dtype=int)
Epoch_Train = np.arange(Num_Epoch_Train,dtype=int)

Channel_Gain_Train = Dataset['Channel_Gain_Train']

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

# MAB
epsilon = 0.1
N_Train_MAB = np.zeros((1014), dtype=int)
Q_Train_MAB = np.zeros((1014))
Solution_Train_MAB, Latency_Train_MAB = initialize_storage(Num_Episode_Train, Num_Epoch_Train, M)

# LMAB
lambda_Train_LMAB = 2.5
V_Train_LMAB = np.eye(num_state)
b_Train_LMAB = np.zeros((num_state,1))
Solution_Train_LMAB, Latency_Train_LMAB = initialize_storage(Num_Episode_Train, Num_Epoch_Train, M)

start_time_train = time.time()
for episode in range(Num_Episode_Train):
    
    start_time_episode = time.time()
    
    epoch_cur = 0
    for epoch in range(Num_Epoch_Train):

        epoch = 0
        
        start_time_epoch = time.time()

        channel_gain = Channel_Gain_Train[episode][epoch].flatten()

        channel_gain_state = channel_gain / channel_gain.max()

        combinations = generate_valid_combinations(M, K_possible, r, T)
        state = np.concatenate((np.tile(channel_gain_state, (combinations.shape[0], 1, 1)), combinations), axis=2)
        state = np.concatenate((np.tile(np.array([1]), (state.shape[0], 1, 1)), state), axis=2)
        state = np.vstack((np.concatenate((np.array([0]), channel_gain_state, np.zeros((M+K_possible.shape[0])))).reshape(1, 1, -1), state))

        # MAB
        # action
        if np.random.random() < epsilon:
            action = np.random.choice(np.arange(1014), size=1, replace=False).item()
        else:
            action = np.argmax(Q_Train_MAB)
        if action != 0:
            k = np.argmax(state[action].flatten()[1+M:1+M+K_possible.shape[0]])
            K = K_possible[k]
            a = state[action].flatten()[1+M+K_possible.shape[0]:]
            if not np.all(np.sum(a) >= r*(K+T-1)+1):
                raise ValueError("Insufficient number of selected workers!")
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
            eta_W = a / np.sum(a)
            objective_MAB, system_latency_MAB, comp_latency_MAB, comm_latency_MAB = get_latency(M, a, eta_W, channel_gain, comp_overhead, comm_overhead, P_U_Watt, P_W_Watt, B)
            # save results
            Solution_Train_MAB = update_solution(Solution_Train_MAB, K, a, eta_W, episode, epoch_cur)
            Latency_Train_MAB = update_latency(Latency_Train_MAB, objective_MAB, system_latency_MAB, comp_latency_MAB, comm_latency_MAB, episode, epoch_cur)
        elif action == 0:
            objective_MAB = system_latency_MAB = get_overhead(l) / f_C_user
        # reward
        reward_MAB = get_reward(objective_MAB)
        # update model
        N_Train_MAB[action] += 1
        Q_Train_MAB[action] += (reward_MAB - Q_Train_MAB[action]) / N_Train_MAB[action]
        # if epsilon > 0.05:
        #     epsilon /= 1.001

        # LMAB
        # action
        theta = np.linalg.inv(V_Train_LMAB) @ b_Train_LMAB
        score_list = []
        for idx in range(1014):
            score = theta.T @ state[idx].T + lambda_Train_LMAB * np.sqrt(state[idx] @ np.linalg.inv(V_Train_LMAB) @ state[idx].T)
            score_list.append(score.item())
        action = np.argmax(score_list)
        if action != 0:
            k = np.argmax(state[action].flatten()[1+M:1+M+K_possible.shape[0]])
            K = K_possible[k]
            a = state[action].flatten()[1+M+K_possible.shape[0]:]
            if not np.all(np.sum(a) >= r*(K+T-1)+1):
                raise ValueError("Insufficient number of selected workers!")
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
            eta_W = a / np.sum(a)
            objective_LMAB, system_latency_LMAB, comp_latency_LMAB, comm_latency_LMAB = get_latency(M, a, eta_W, channel_gain, comp_overhead, comm_overhead, P_U_Watt, P_W_Watt, B)
            # save results
            Solution_Train_LMAB = update_solution(Solution_Train_LMAB, K, a, eta_W, episode, epoch_cur)
            Latency_Train_LMAB = update_latency(Latency_Train_LMAB, objective_LMAB, system_latency_LMAB, comp_latency_LMAB, comm_latency_LMAB, episode, epoch_cur)
        elif action == 0:
            objective_LMAB = system_latency_LMAB = get_overhead(l) / f_C_user
        # reward
        reward_LMAB = get_reward(objective_LMAB)
        # update model
        V_Train_LMAB += np.outer(state[action],state[action])
        b_Train_LMAB += reward_LMAB * state[action].T

        end_time_epoch = time.time()
        print(f"Episode: {episode+1}, Epoch: {epoch_cur+1},")
        print(f"MAB: {Latency_Train_MAB['System_Latency'][episode][epoch_cur]:.5f}, {reward_MAB:.5f}, K = {Solution_Train_MAB['K'][episode][epoch_cur]}, a = {Solution_Train_MAB['a'][episode][epoch_cur]},")
        print(f"LMAB: {Latency_Train_LMAB['System_Latency'][episode][epoch_cur]:.5f}, {reward_LMAB:.5f}, K = {Solution_Train_LMAB['K'][episode][epoch_cur]}, a = {Solution_Train_LMAB['a'][episode][epoch_cur]},")
        print(f"Time cost: {end_time_epoch - start_time_epoch:.5f},")
        print('-'*30)
        with open(name_txt, "a") as file:
            file.write(f"Episode: {episode+1}, Epoch: {epoch_cur+1},\n")
            file.write(f"MAB: {Latency_Train_MAB['System_Latency'][episode][epoch_cur]:.5f}, {reward_MAB:.5f}, K = {Solution_Train_MAB['K'][episode][epoch_cur]}, a = {Solution_Train_MAB['a'][episode][epoch_cur]},\n")
            file.write(f"LMAB: {Latency_Train_LMAB['System_Latency'][episode][epoch_cur]:.5f}, {reward_LMAB:.5f}, K = {Solution_Train_LMAB['K'][episode][epoch_cur]}, a = {Solution_Train_LMAB['a'][episode][epoch_cur]},\n")
            file.write(f"Time cost: {end_time_epoch - start_time_epoch:.5f},\n")
            file.write("-" * 30 + "\n")
    
    end_time_episode = time.time()

    if (episode+1) % 500 == 0:
        for name in Method:
            update_npz_file(name,
                            {'Num_Episode_Train':Num_Episode_Train, 'Num_Epoch_Train':Num_Epoch_Train})
        # MAB
        update_npz_file('MAB',
                        {'Q_Train':Q_Train_MAB, 'N_Train': N_Train_MAB})
        update_npz_file('MAB', update_name(Solution_Train_MAB, '_Train'))
        update_npz_file('MAB', update_name(Latency_Train_MAB, 't_', '_Train'))
        # LMAB
        update_npz_file('LMAB',
                        {'V_Train': V_Train_LMAB, 'b_Train': b_Train_LMAB,
                         'lambda_Train':lambda_Train_LMAB})
        update_npz_file('LMAB', update_name(Solution_Train_LMAB, '_Train'))
        update_npz_file('LMAB', update_name(Latency_Train_LMAB, 't_', '_Train'))
end_time_train = time.time()
print(f"{'Train time cost'}: {end_time_train - start_time_train:.5f},")
print('-'*30)

end_time_global = time.time()
print(f"{'Total time cost'}: {end_time_global - start_time_global:.5f},")
with open(name_txt, "a") as file:
    file.write(f"{'Total time cost'}: {end_time_global - start_time_global:.5f},\n")
    file.write("-" * 30 + "\n")