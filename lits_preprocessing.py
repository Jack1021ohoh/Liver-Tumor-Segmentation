import os
import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm

input_image_dir = './lits_dataset/volume'
input_label_dir = './lits_dataset/segmentation'
output_dir = './lits_dataset_preprocessed'

def window_image(image, window_width, window_level):
    min_intensity = window_level - window_width // 2
    max_intensity = window_level + window_width // 2

    image = np.clip(image, min_intensity, max_intensity)

    image = (image - min_intensity) / (max_intensity - min_intensity)
    
    return image.astype(np.float32)

def process_image(image_file):
    print(f'Processing {image_file}')
    label_file = image_file.replace('volume', 'segmentation')
    image_path = os.path.join(input_image_dir, image_file)
    label_path = os.path.join(input_label_dir, label_file)

    if not os.path.exists(label_path):
        print(f"{label_file} not found!")
        return

    image_nii = nib.load(image_path)
    label_nii = nib.load(label_path)
    image_data = image_nii.get_fdata()
    label_data = label_nii.get_fdata()

    image_output_dir = os.path.join(output_dir, image_file.split('.')[0].split('-')[1], 'Image')
    label_output_dir = os.path.join(output_dir, image_file.split('.')[0].split('-')[1], 'GT')
    os.makedirs(image_output_dir, exist_ok = True)
    os.makedirs(label_output_dir, exist_ok = True)

    for i in range(image_data.shape[2]):  
        image_slice = image_data[:, :, i]
        label_slice = label_data[:, :, i]

        if np.any(label_slice != 0):  
            image_slice_path = os.path.join(image_output_dir, f'{i}.npy')
            label_slice_path = os.path.join(label_output_dir, f'{i}.npy')

            normalized_label_slice = label_slice.astype(np.uint8)

            normalized_image_slice = window_image(image_slice, window_width = 400, window_level = 50)

            np.save(image_slice_path, normalized_image_slice)
            np.save(label_slice_path, normalized_label_slice)

if __name__ == '__main__':

    image_files = [f for f in os.listdir(input_image_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]

    for image_file in tqdm(image_files):
        process_image(image_file)

    print("Preprocessing finished")