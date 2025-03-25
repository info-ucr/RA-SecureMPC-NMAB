import numpy as np
import os
import random
import torch
import time

start_time_global = time.time()

def initialize_positions(N, M, width):
    """
    Initialize positions for N users and M workers within a square area of given width.
    Ensures minimum distance constraints between users (>25m), between users and workers (>25m),
    and between workers (>25m).
    Parameters: N (int): Number of users.
                M (int): Number of workers.
                width (float): Width of the square area.
    Returns: Position_User (numpy.ndarray): Positions of users with shape (N, 2).
             Position_Worker (numpy.ndarray): Positions of workers with shape (M, 2).
    """
    # Initialize empty arrays for positions
    Position_User = np.zeros((N, 2))
    Position_Worker = np.zeros((M, 2))
    # Helper function to check minimum distance constraint
    def check_distance(new_position, existing_positions, min_distance):
        if existing_positions.size == 0:
            return True
        distances = np.sqrt(np.sum((existing_positions - new_position) ** 2, axis=1))
        return np.all(distances > min_distance)
    # Place users ensuring distance > 25 meters between them
    for i in range(N):
        valid_position = False
        while not valid_position:
            new_position = np.random.rand(2) * width
            if check_distance(new_position, Position_User[:i], 25):
                Position_User[i] = new_position
                valid_position = True
    # Place workers ensuring distance > 25 meters from all users and other workers
    for i in range(M):
        valid_position = False
        while not valid_position:
            new_position = np.random.rand(2) * width
            if check_distance(new_position, Position_User, 25) and check_distance(new_position, Position_Worker[:i], 25):
                Position_Worker[i] = new_position
                valid_position = True         
    return Position_User, Position_Worker

def get_distance(Position_User, Position_Worker):
    # compute distance in km
    Distance = np.zeros((Position_User.shape[0],Position_Worker.shape[0]))
    for i in range(Position_User.shape[0]):
        for j in range(Position_Worker.shape[0]):
            Distance[i,j] = np.linalg.norm(Position_User[i,:]-Position_Worker[j,:]) / 1e3
    return Distance

def get_channel_gain_LS(Distance, sigma2_shadow):
    # Path loss in dB
    PL_dB = 128.1 + 37.6 * np.log10(Distance)
    # Shadowing in dB, Gauss-distributed with mean 0 and variance sigma2_shadow
    zeta_dB = np.random.normal(0, np.sqrt(sigma2_shadow), Distance.shape)
    # Large-scale fading propagation coefficient in linear scale
    phi_linear = 10 ** ((-(PL_dB - zeta_dB)) / 10)
    return phi_linear

def get_channel_gain_SS(Distance):
    # Small-scale fading propagation coefficient (Rayleigh fading, |h|^2 ~ Exponential with mean 1)
    # Since h ~ CN(0, 1), |h|^2 follows an exponential distribution with lambda = 1 (mean = 1/lambda = 1)
    h_magnitude_squared = np.random.exponential(1, Distance.shape)
    return h_magnitude_squared

def get_channel_gain(phi_linear, h_magnitude_squared):
    return phi_linear * h_magnitude_squared

def get_random_images(imageset, size):
    # Randomly select indices without replacement
    random_indices = np.random.choice(imageset, size, replace=False)
    return random_indices

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

def get_overhead(l, K, T, r, size, count_1, count_2, count_3):

    d_1 = size[0]
    d_2 = size[1]
    
    # computation overhead

    # Data preprocessing
    Lambda_MAP_1 = (
        l * d_1 * (
            num_cycle(field='Real', opera='Mul', dtype='Float') +
            num_cycle(field='Real', opera='Round', dtype='Float') +
            num_cycle(field='Real', opera='Sub', dtype='Float') +
            num_cycle(field='Real', opera='Cmp', dtype='Float') +
            num_cycle(field='Real', opera='Cmp', dtype='Int')
        )
        + count_1 * num_cycle(field='Real', opera='Add', dtype='Int')
        + count_2 * num_cycle(field='Real', opera='Add', dtype='Int')
    )

    # Masking
    Lambda_Mask = l * d_1 * num_cycle(field='FF', opera='Sub')

    # Encoding
    Lambda_Encode = (
        K * l/K * d_1 * num_cycle(field='FF', opera='Mul') +
        l/K * d_1 * (K - 1) * num_cycle(field='FF', opera='Add') +
        l/K * d_1 * num_cycle(field='FF', opera='Add')
    )

    # Computing
    Lambda_Comp_PiNet = l/K * (
        # Convolution Layer 1
        16 * 16 * 64 * 147 * num_cycle(field='FF', opera='Mul') +
        16 * 16 * 64 * 146 * num_cycle(field='FF', opera='Add') +
        # Convolution Layer 2
        16 * 16 * 64 * 576 * num_cycle(field='FF', opera='Mul') +
        16 * 16 * 64 * 575 * num_cycle(field='FF', opera='Add') +
        # Convolution Layer 3
        16 * 16 * 64 * 576 * num_cycle(field='FF', opera='Mul') +
        16 * 16 * 64 * 575 * num_cycle(field='FF', opera='Add') +
        # Hadamard Product
        16 * 16 * 64 * num_cycle(field='FF', opera='Mul') +
        # Summation
        16 * 16 * 64 * 2 * num_cycle(field='FF', opera='Add') +
        # Fully Connected Layer
        10 * 16384 * num_cycle(field='FF', opera='Mul') +
        (10 * (16384 - 1) + 10) * num_cycle(field='FF', opera='Add')
    )
    Lambda_Comp_Mask = l/K * d_2 * num_cycle(field='FF', opera='Sub')
    Lambda_Comp = Lambda_Comp_PiNet + Lambda_Comp_Mask

    # Decoding
    Lambda_LI = (
        K * l/K * d_2 * (r * (K + T - 1) + 1) * num_cycle(field='FF', opera='Mul') +
        K * r * (K + T - 1) * l/K * d_2 * num_cycle(field='FF', opera='Add')
    )
    lambda_Unmask = K * l/K * d_2 * num_cycle(field='FF', opera='Add')
    Lambda_Decode = Lambda_LI + lambda_Unmask

    # Result postprocessing
    Lambda_Map_2 = (
        K * l/K * d_2 * num_cycle(field='Real', opera='Cmp', dtype='Int') +
        count_3 * num_cycle(field='Real', opera='Sub', dtype='Int') +
        K * l/K * d_2 * num_cycle(field='Real', opera='Mul', dtype='Float')
    )

    Lambda_final = (10 - 1) * l * num_cycle(field='Real', opera='Cmp', dtype='Float')

    # communication overhead
    # multicast
    Lambda_Multicast = l*d_1 * 26
    # upload
    Lambda_Upload = l/K*d_2 * 26
    
    # computation overhead
    comp_overhead = {
        'Map_1': Lambda_MAP_1,
        'Mask': Lambda_Mask,
        'Encode': Lambda_Encode,
        'Comp': Lambda_Comp,
        'Decode': Lambda_Decode,
        'Map_2': Lambda_Map_2,
        'Final': Lambda_final,
    }
    # communication overhead
    comm_overhead = {
        'Multicast': Lambda_Multicast,
        'Upload': Lambda_Upload,
    }
    
    return comp_overhead, comm_overhead

def initialize_overhead_storage(num_episode, num_epoch, num_K):
    overhead = {
        'Map_1': np.zeros((num_episode, num_epoch, num_K)),
        'Mask': np.zeros((num_episode, num_epoch, num_K)),
        'Encode': np.zeros((num_episode, num_epoch, num_K)),
        'Comp': np.zeros((num_episode, num_epoch, num_K)),
        'Decode': np.zeros((num_episode, num_epoch, num_K)),
        'Map_2': np.zeros((num_episode, num_epoch, num_K)),
        'Final': np.zeros((num_episode, num_epoch, num_K)),
        'Multicast': np.zeros((num_episode, num_epoch, num_K)),
        'Upload': np.zeros((num_episode, num_epoch, num_K))
    }
    return overhead

def update_overhead(overhead, comp_overhead, comm_overhead, episode, epoch, k):
    overhead['Map_1'][episode][epoch][k] = comp_overhead['Map_1']
    overhead['Mask'][episode][epoch][k] = comp_overhead['Mask']
    overhead['Encode'][episode][epoch][k] = comp_overhead['Encode']
    overhead['Comp'][episode][epoch][k] = comp_overhead['Comp']
    overhead['Decode'][episode][epoch][k] = comp_overhead['Decode']
    overhead['Map_2'][episode][epoch][k] = comp_overhead['Map_2']
    overhead['Final'][episode][epoch][k] = comp_overhead['Final']
    overhead['Multicast'][episode][epoch][k] = comm_overhead['Multicast']
    overhead['Upload'][episode][epoch][k] = comm_overhead['Upload']
    return overhead

def update_name(dictionary, new_name_1, new_name_2):
    updated_dictionary = {}
    for key in dictionary:
        updated_key = new_name_1 + key + new_name_2
        updated_dictionary[updated_key] = dictionary[key]
    return updated_dictionary

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
M = 11
# number of collude workers
T = 1
# polynomial degree
r = 2
# local dataset size
l = 1200
# number of dropout workers
D = 0
# area (m)
width = 500
# transmit power for broadcasting of users (dBm)
P_U_dBm = 23
P_U_Watt = 10**((P_U_dBm-30)/10) * 1e3
# maximum transmit power for worker (dBm)
P_W_dBm = np.ones((M)) * 23
P_W_Watt = 10**((P_W_dBm-30)/10) * 1e3
# bandwidth (MHz)
B = 2
# noise power (dBm)
Noise_dBm = -96
Noise_Watt = 10**((Noise_dBm-30)/10) * 1e3
# shadowing effect (dB)
shadow_variance = 8
# CPU frequency (MHz)
f_C_user = 10

# CPU_mode = 'Homo'
CPU_mode = 'Heter'

if CPU_mode == 'Homo':
    CPU = 1
    if CPU == 1:
        f_C_worker = np.ones((M), dtype=int) * 100
    elif CPU == 2:
        f_C_worker = np.ones((M), dtype=int) * 500
    elif CPU == 3:
        f_C_worker = np.ones((M), dtype=int) * 1000
    name_npz = 'Dataset ' + str(seed) + ' ' + str(l) + ' ' + CPU_mode + '_' + str(CPU) + '.npz'
    name_txt = 'Dataset ' + str(seed) + ' ' + str(l) + ' ' + CPU_mode + '_' + str(CPU) + '.txt'
elif CPU_mode == 'Heter':
    # beta
    # Parameters for the beta distribution
    # Scale the frequencies to fit within the range 100 to 1000
    CPU = 1
    if CPU == 1:
        f_C_worker = 100 + np.random.beta(a=0.5, b=0.5, size=M) * (1000 - 100)
    elif CPU == 2:
        f_C_worker = 100 + np.random.beta(a=0.6, b=0.4, size=M) * (1000 - 100)
    elif CPU == 3:
        f_C_worker = 100 + np.random.beta(a=0.7, b=0.3, size=M) * (1000 - 100)
    name_npz = 'Dataset ' + str(seed) + ' ' + str(l) + ' ' + CPU_mode + '_' + str(CPU) + '.npz'
    name_txt = 'Dataset ' + str(seed) + ' ' + str(l) + ' ' + CPU_mode + '_' + str(CPU) + '.txt'

K_possible = np.arange(1,np.floor(np.divide(M-1-D,r)-T+1).astype(int)+1)

data = np.load('Cifar10.npz')

# train

Num_Episode_Train = 15000
Num_Epoch_Train = 1

data_1_Train = data['data_1_Train']
data_2_FF_Train = data['data_2_FF_Train']
count_1_Train = data['count_1_Train']
count_2_Train = data['count_2_Train']
count_3_Train = data['count_3_Train']

Position_User_Train = np.zeros((Num_Episode_Train, 1, 2))
Position_Worker_Train = np.zeros((Num_Episode_Train, M, 2))
Distance_Train = np.zeros((Num_Episode_Train, 1, M))
Channel_Gain_Train = np.zeros((Num_Episode_Train, Num_Epoch_Train, 1, M))
Channel_Gain_LS_Train = np.zeros((Num_Episode_Train, 1, M))
Channel_Gain_SS_Train = np.zeros((Num_Episode_Train, Num_Epoch_Train, 1, M))

Image_Index_Train = np.zeros((Num_Episode_Train, l), dtype=int)

Overhead_Train = initialize_overhead_storage(Num_Episode_Train, Num_Epoch_Train, K_possible.shape[0])

start_time = time.time()

for episode in range(Num_Episode_Train):
    
    start_time_episode = time.time()
    
    # initialize positions
    position_user, position_worker = initialize_positions(1, M, width)
    Position_User_Train[episode] = position_user
    Position_Worker_Train[episode] = position_worker
    # compute distance
    distance = get_distance(position_user, position_worker)
    Distance_Train[episode] = distance

    channel_gain_LS = get_channel_gain_LS(distance, shadow_variance)
    Channel_Gain_LS_Train[episode] = channel_gain_LS

    Image_Index_Train[episode] = get_random_images(10000, l)

    # data
    data_1 = data_1_Train[Image_Index_Train[episode]]
    data_2_FF = data_2_FF_Train[Image_Index_Train[episode]]
    count_1 = np.sum(count_1_Train[Image_Index_Train[episode]])
    count_2 = np.sum(count_2_Train[Image_Index_Train[episode]])
    count_3 = np.sum(count_3_Train[Image_Index_Train[episode]])

    # size = d_1, d_2
    size = [data_1.flatten().shape[0]/l, data_2_FF.flatten().shape[0]/l]
    
    for epoch in range(Num_Epoch_Train):

        start_time_epoch = time.time()

        channel_gain_SS = get_channel_gain_SS(distance)
        channel_gain = get_channel_gain(channel_gain_LS, channel_gain_SS)
        Channel_Gain_SS_Train[episode][epoch] = channel_gain_SS
        Channel_Gain_Train[episode][epoch] = channel_gain

        for k in range(K_possible.shape[0]):
        
            comp_overhead, comm_overhead = get_overhead(l, K_possible[k], T, r, size, count_1, count_2, count_3)
            
            Overhead_Train = update_overhead(Overhead_Train, comp_overhead, comm_overhead, episode, epoch, k)
        
        end_time_epoch = time.time()
    
    end_time_episode = time.time()
    print(f"Episode: {episode+1},")
    print(f"Time cost: {end_time_episode - start_time_episode:.5f},")
    print('-'*30)
    with open(name_txt, "a") as file:
        file.write(f"Episode: {episode+1},\n")
        file.write(f"Time cost: {end_time_episode - start_time_episode:.5f},\n")
        file.write("-" * 30 + "\n")

    if (episode+1) % Num_Episode_Train == 0:
        update_npz_file(name_npz,
                        {'seed':seed,
                         'M':M, 'T':T, 'r':r, 'l':l, 'D':D, 'width':width, 'K_possible':K_possible,
                         'P_U_dBm':P_U_dBm, 'P_U_Watt':P_U_Watt, 'P_W_dBm':P_W_dBm, 'P_W_Watt':P_W_Watt,
                         'B':B, 'Noise_dBm':Noise_dBm, 'Noise_Watt':Noise_Watt, 'shadow_variance':shadow_variance,
                         'f_C_user':f_C_user, 'f_C_worker':f_C_worker,
                         'Num_Episode_Train':Num_Episode_Train, 'Num_Epoch_Train':Num_Epoch_Train,
                         'Position_User_Train':Position_User_Train, 'Position_Worker_Train':Position_Worker_Train,
                         'Distance_Train':Distance_Train, 'Channel_Gain_LS_Train':Channel_Gain_LS_Train,
                         'Channel_Gain_SS_Train':Channel_Gain_SS_Train, 'Channel_Gain_Train':Channel_Gain_Train,
                         'Image_Index_Train':Image_Index_Train})
        update_npz_file(name_npz, update_name(Overhead_Train, 'Lambda_', '_Train'))

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