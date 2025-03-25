import numpy as np
from numpy import ndarray
import os
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

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
        
def get_random_images(imageset, size):
    # Randomly select indices without replacement
    random_indices = np.random.choice(len(imageset), size, replace=False)
    # Extract images and labels based on the selected indices
    images = [imageset[i][0] for i in random_indices]
    labels = [imageset[i][1] for i in random_indices]
    # Stack the images into a single tensor
    images = torch.stack(images)
    # Convert labels to a tensor
    labels = torch.tensor(labels)
    # Return images, labels, and indices
    return images, labels, random_indices

def to_finite_field_domain(real: ndarray, quantization_bit: int, prime: int) -> ndarray:
    scaled_real = real * (2 ** quantization_bit)
    int_domain = np.around(scaled_real)
    finite_field = np.zeros(int_domain.shape, dtype=np.int64)
    negative_mask = int_domain < 0
    finite_field[~negative_mask] = int_domain[~negative_mask]
    finite_field[negative_mask] = prime - (int_domain[negative_mask] * -1).astype(np.int64)
    return finite_field

def to_int_domain(real: ndarray, quantization_bit: int) -> ndarray:
    scaled_real = real * (2 ** quantization_bit)
    int_domain = np.around(scaled_real).astype(np.int64)
    return int_domain

def to_real_domain(finite_field: ndarray, quantization_bit: int, prime: int) -> ndarray:
    threshold = (prime - 1) / 2
    negative_mask = finite_field > threshold
    real_domain = np.zeros(finite_field.shape, dtype=np.float64)
    real_domain[~negative_mask] = finite_field[~negative_mask]
    real_domain[negative_mask] = -1 * (prime - finite_field[negative_mask]).astype(np.float64)
    real_domain = real_domain / (2 ** quantization_bit)
    return real_domain

def int_truncation(int_domain: ndarray, scale_down: int) -> ndarray:
    return int_domain >> scale_down

def from_finite_field_to_int_domain(finite_field: ndarray, prime: int) -> ndarray:
    int_domain = np.zeros(finite_field.shape, dtype=np.int64)
    threshold = (prime - 1) / 2
    negative_mask = finite_field > threshold
    int_domain[~negative_mask] = finite_field[~negative_mask]
    int_domain[negative_mask] = -1 * (prime - finite_field[negative_mask]).astype(np.int64)
    return int_domain

def from_int_to_finite_field_domain(int_domain: ndarray, prime: int) -> ndarray:
    finite_field = np.zeros(int_domain.shape, dtype=np.int64)
    negative_mask = int_domain < 0
    finite_field[~negative_mask] = int_domain[~negative_mask]
    finite_field[negative_mask] = int_domain[negative_mask] + prime
    return finite_field

def finite_field_truncation(finite_field: ndarray, scale_down: int, prime: int) -> ndarray:
    int_domain = from_finite_field_to_int_domain(finite_field, prime)
    int_domain = int_truncation(int_domain, scale_down)
    finite_field_domain = from_int_to_finite_field_domain(int_domain, prime)
    return finite_field_domain

def multiply_finite_field(data_1: ndarray, data_2: ndarray, quantization_bit: int, prime: int) -> ndarray:
    product = np.mod(np.multiply(data_1,data_2), prime)
    return finite_field_truncation(product, quantization_bit, prime)

class PiNet(nn.Module):
    def __init__(self, num_classes=10):
        super(PiNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.fc = nn.Linear(16384, num_classes)

    def forward(self, x, mode=1):
        if mode == 1:
            x = self.conv1(x)
            first = self.conv3(self.conv2(x))
            second = x * first
            x = x + first + second
            x = torch.flatten(x, 1)
            x_intermediate_1 = self.fc(x)
            x = F.softmax(x_intermediate_1, dim=1)
            return x_intermediate_1, x
        elif mode == 2:
            x = F.softmax(x, dim=1)
            return x
        
def generate_patches(Input, kernel, stride=1, padding=0):
    patches = []
    num_of_samples, num_channels, image_height, image_width = Input.shape
    out_channels, _, kernel_height, kernel_width = kernel.shape
    
    stride_over_height, stride_over_width = stride, stride
    padding_top, padding_bottom, padding_left, padding_right = padding, padding, padding, padding
    
    padded_height = image_height + padding_top + padding_bottom
    padded_width = image_width + padding_left + padding_right
    
    curr_input_data = np.zeros((num_of_samples, num_channels, padded_height, padded_width), dtype=np.int64)
    curr_input_data[:, :, padding_top:padding_top + image_height, padding_left:padding_left + image_width] = Input
    
    for output_width_idx in range(0, padded_width - kernel_width + 1, stride_over_width):
        for output_height_idx in range(0, padded_height - kernel_height + 1, stride_over_height):
            patches.append((
                output_height_idx, output_width_idx,
                curr_input_data[:, :, output_height_idx:output_height_idx + kernel_height, output_width_idx:output_width_idx + kernel_width]
            ))
    
    return patches

def conv2d_numpy_finite_field_2(Input, kernel, quantization_bit, prime, stride=1, padding=0):
    patches = generate_patches(Input, kernel, stride=stride, padding=padding)
    num_of_samples, num_channels, image_height, image_width = Input.shape
    out_channels, _, kernel_height, kernel_width = kernel.shape
    
    output_height = (image_height + 2 * padding - kernel_height) // stride + 1
    output_width = (image_width + 2 * padding - kernel_width) // stride + 1
    
    output_data = np.zeros((num_of_samples, out_channels, output_height, output_width), dtype=np.int64)
    
    for patch in patches:
        output_height_idx, output_width_idx, patch_data = patch
        unscaled_out = np.tensordot(patch_data, kernel, axes=([1, 2, 3], [1, 2, 3]))
        output_data[:, :, output_height_idx // stride, output_width_idx // stride] = unscaled_out
    
    output_data = finite_field_truncation(np.mod(output_data, prime), quantization_bit, prime)
    
    return output_data

def fully_connected_finite_field(data, weight, bias, quantization_bit, prime):
    temp = finite_field_truncation(np.mod(np.dot(data, weight.T), prime), quantization_bit, prime)
    return np.mod(temp + bias, prime)

def PiNet_numpy_finite_field(Input, model_torch, quantization_bit, prime):
    Input = to_finite_field_domain(Input, quantization_bit, prime)
    conv1_weight = to_finite_field_domain(model_torch.state_dict()['conv1.weight'].numpy(), quantization_bit, prime)
    conv2_weight = to_finite_field_domain(model_torch.state_dict()['conv2.weight'].numpy(), quantization_bit, prime)
    conv3_weight = to_finite_field_domain(model_torch.state_dict()['conv3.weight'].numpy(), quantization_bit, prime)
    fc_weight = to_finite_field_domain(model_torch.state_dict()['fc.weight'].numpy(), quantization_bit, prime)
    fc_bias = to_finite_field_domain(model_torch.state_dict()['fc.bias'].numpy(), quantization_bit, prime)
    Input = conv2d_numpy_finite_field_2(Input, conv1_weight, quantization_bit, prime, stride=2, padding=3)
    first = conv2d_numpy_finite_field_2(Input, conv2_weight, quantization_bit, prime, stride=1, padding=1)
    first = conv2d_numpy_finite_field_2(first, conv3_weight, quantization_bit, prime, stride=1, padding=1)
    second = multiply_finite_field(Input, first, quantization_bit, prime)
    out_2 = np.mod(Input + first + second, prime)
    out = out_2.reshape(out_2.shape[0],np.prod(out_2.shape[1:]))
    out = fully_connected_finite_field(out, fc_weight, fc_bias, quantization_bit, prime)
    return out, to_real_domain(out, quantization_bit, prime)

def data_prepare(loader, torch_model, device, quantization_bit, prime):
    data_1 = np.zeros((0,3,32,32))
    data_2_FF = np.zeros((0,10))
    correct_torch, correct_numpy = torch.zeros((0),dtype=int,device=device), torch.zeros((0),dtype=int,device=device)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            # torch model
            torch_model.to(device)
            _, output_torch = torch_model(images, mode=1)
            _, predicted = output_torch.max(1)
            correct_torch = torch.cat((correct_torch, predicted.eq(labels)))
            # paritial numpy model
            out_numpy_2_FF, out_numpy_2 = PiNet_numpy_finite_field(images.cpu().numpy(), torch_model.cpu(), quantization_bit, prime)
            torch_model.to(device)
            outputs = torch_model(torch.tensor(out_numpy_2, dtype=torch.float32).to(device), mode=2)
            _, predicted = outputs.max(1)
            correct_numpy = torch.cat((correct_numpy, predicted.eq(labels)))
        data_1 = np.concatenate((data_1, images.cpu().numpy()), axis=0)
        data_2_FF = np.concatenate((data_2_FF, out_numpy_2_FF), axis=0)
    return data_1, data_2_FF, correct_torch.cpu().numpy(), correct_numpy.cpu().numpy()

def count_elements(array, mode=1, prime=None):
    if mode == 1:
        # Task 1: Count the number of elements where element - floor(element) >= 0.5
        return np.sum(array - np.floor(array) >= 0.5)
    elif mode == 2:
        # Task 2: Count the number of elements where round(element * 2^q) < 0
        return np.sum(np.round(array * 2**q) < 0)
    elif mode == 3:
        # Task 3: Count the number of elements within the range [(p-1)/2, p)
        return np.sum((array >= (prime - 1) / 2) & (array < prime))

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

gpu = 0
device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')

q = 8
p = 2**26 - 5

torch_model = PiNet(num_classes=10).to(device)
torch_model.load_state_dict(torch.load('PiNet_CIFAR10_SGD_42_AllPoly_Aug.pth',map_location=device))

# Load CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# train
trainset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=0)
data_1_Train, data_2_FF_Train, correct_torch_Train, correct_numpy_Train = data_prepare(trainloader, torch_model, device, q, p)

count_1_Train = np.zeros((len(trainset)))
count_2_Train = np.zeros((len(trainset)))
count_3_Train = np.zeros((len(trainset)))
for idx in range(len(trainset)):
    count_1_Train[idx] = count_elements(data_1_Train[idx], mode=1)
    count_2_Train[idx] = count_elements(data_1_Train[idx], mode=2)
    count_3_Train[idx] = count_elements(data_2_FF_Train[idx], mode=3, prime=p)

update_npz_file('Cifar10.npz',
                {'prime':p, 'quantization_bit':q,
                 'data_1_Train':data_1_Train, 'data_2_FF_Train':data_2_FF_Train,
                 'count_1_Train':count_1_Train, 'count_2_Train':count_2_Train, 'count_3_Train':count_3_Train,
                 'correct_torch_Train':correct_torch_Train, 'correct_numpy_Train':correct_numpy_Train})