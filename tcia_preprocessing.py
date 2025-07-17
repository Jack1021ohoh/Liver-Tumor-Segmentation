import pydicom
import numpy as np
import pandas as pd
import os
from tqdm.contrib import tzip

output_dir = './tcia_dataset_preprocessed'

def window_image(image, window_width, window_level):
    """Apply windowing to CT image with proper HU values"""
    min_intensity = window_level - window_width // 2
    max_intensity = window_level + window_width // 2

    # Clip the image to the window range
    image = np.clip(image, min_intensity, max_intensity)

    # Normalize to [0, 1] range
    image = (image - min_intensity) / (max_intensity - min_intensity)
    
    return image.astype(np.float32)

def apply_dicom_scaling(pixel_array, dicom_header):
    """Apply DICOM rescale slope and intercept to get HU values"""
    # Get rescale slope and intercept (default to 1 and 0 if not present)
    rescale_slope = getattr(dicom_header, 'RescaleSlope', 1)
    rescale_intercept = getattr(dicom_header, 'RescaleIntercept', 0)
    rescale_slope = int(rescale_slope)
    rescale_intercept = int(rescale_intercept)
    
    # Convert to HU values
    hu_image = pixel_array * rescale_slope + rescale_intercept
    
    return hu_image.astype(np.float32)

def combine_label(label_data):
    liver_label, tumor_label, pv_label, _= np.array_split(label_data, 4)
    combined_liver_label = np.logical_or(liver_label, pv_label)
    all_label = tumor_label * 2 + combined_liver_label

    return all_label

def find_image_files(path):
    files = os.listdir(path)
    result = [os.path.join(path, f) for f in files]
    result.sort()

    return result

def process_image(image_paths, label_path):
    if not os.path.exists(label_path):
        print(f"{label_path} not found!")
        return
    
    label_dcm = pydicom.dcmread(label_path)
    label_data = label_dcm.pixel_array
    label_data = combine_label(label_data)
    label_data = label_data[::-1, :, :]

    image_output_dir = os.path.join(output_dir, label_path.split('/')[3], 'Image')
    label_output_dir = os.path.join(output_dir, label_path.split('/')[3], 'GT')
    os.makedirs(image_output_dir, exist_ok = True)
    os.makedirs(label_output_dir, exist_ok = True)

    for i, image_path in enumerate(image_paths):
        image_dcm = pydicom.dcmread(image_path)
        image_data = image_dcm.pixel_array
        image_data = image_data.astype(np.int16)
        
        # Apply DICOM scaling to get proper HU values
        image_data = apply_dicom_scaling(image_data, image_dcm)
        
        label_slice = label_data[i, :, :]

        if np.any(label_slice != 0):  
            image_slice_path = os.path.join(image_output_dir, f'{i}.npy')
            label_slice_path = os.path.join(label_output_dir, f'{i}.npy')

            normalized_label_slice = label_slice.astype(np.uint8)

            # Apply windowing with appropriate parameters for liver CT
            # Common liver window: W=150-200, L=30-60
            normalized_image_slice = window_image(image_data, window_width = 400, window_level = 50)

            np.save(image_slice_path, normalized_image_slice)
            np.save(label_slice_path, normalized_label_slice)
            
if __name__ == '__main__':
    pv_df = pd.read_csv('./TCIA_dataset/pv_df_full.csv')
    pv_df['vol_file_location'] = pv_df['vol_file_location'].str.replace('\\', '/')
    pv_df['seg_file_location'] = pv_df['seg_file_location'].str.replace('\\', '/')
    pv_df['seg_file_location'] = pv_df['seg_file_location'] + '/1-1.dcm'

    image_paths = [find_image_files(p) for p in pv_df['vol_file_location'].values]

    for images, label in tzip(image_paths, pv_df['seg_file_location'].values):
        process_image(images, label)

    print("Preprocessing finished")