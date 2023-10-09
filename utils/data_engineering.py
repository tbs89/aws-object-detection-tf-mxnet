#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import shutil
import os
from PIL import Image
import json



def extract_id_from_path(file_path: str) -> str:
    """
    Extracts the identifier from a given file path.
    
    Parameters:
    - file_path (str): The path to the image file.
    
    Returns:
    - str: The extracted identifier.
    """
    base_name = os.path.basename(file_path)  
    file_name_without_extension = os.path.splitext(base_name)[0]
    return file_name_without_extension




def insert_column(df, loc, column, value):
    """
    Inserts a new column into the dataframe at the specified location with the given value if it doesn't already exist.
    
    Parameters:
    - df (pd.DataFrame): The dataframe to modify.
    - loc (int): The location to insert the new column.
    - column (str): The name of the new column.
    - value: The default value for the new column.
    
    Returns:
    - pd.DataFrame: The modified dataframe.
    """
    if column not in df.columns:
        df.insert(loc, column, value)
    return df



def move_files(src, dst, action):
    """
    Move or copy files from the source location to the destination location
    based on the provided action. If the source is a directory, the function 
    will act on all files within that directory.

    Parameters
    ----------
    src : str
        The source file or directory path that needs to be moved or copied.
    dst : str
        The destination path where the files need to be moved or copied to.
    action : str
        The action to be performed - "move" or "copy".

    Returns
    -------
    bool
        Returns True if the action was performed successfully.
        
    Raises
    ------
    ValueError
        If the action is not 'move' or 'copy', or if the source path does not exist.
        
    Example
    -------
    >>> move_files('path/to/source', 'path/to/destination', 'copy')
    Action Copy: 3 files from path/to/source to path/to/destination
    True
    """
    if not os.path.exists(src):
        raise ValueError(f"The source path {src} does not exist")
    
    file_count = 0
    if os.path.isdir(src):
        for root, dirs, files in os.walk(src):
            for file in files:
                file_src = os.path.join(root, file)
                file_dst = os.path.join(dst, file)
                
                directory = os.path.dirname(file_dst)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                if action.lower() == 'move':
                    shutil.move(file_src, file_dst)
                    file_count += 1
                elif action.lower() == 'copy':
                    shutil.copy(file_src, file_dst)
                    file_count += 1
                else:
                    raise ValueError("The action parameter must be 'move' or 'copy'")
    else:
        directory = os.path.dirname(dst)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if action.lower() == 'move':
            shutil.move(src, dst)
            file_count = 1
        elif action.lower() == 'copy':
            shutil.copy(src, dst)
            file_count = 1
        else:
            raise ValueError("The action parameter must be 'move' or 'copy'")
    
    print(f"Action {action.capitalize()}: {file_count} files from {src} to {dst}")
    
    
    

def flip_image(image_path, save_path):
    """
    Flip an image horizontally and save it to the specified path.
    
    Parameters
    ----------
    image_path : str
        Path to the original image to be flipped.
    save_path : str
        Path where the flipped image will be saved.
    
    Returns
    -------
    Tuple[int, int]
        The width and height of the original image.
        
    Example
    -------
    >>> image_size = flip_image("path/to/image.jpg", "path/to/save/flipped_image.jpg")
    >>> print(image_size)
    (width, height)
    """
    img = Image.open(image_path)
    img_flip = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    img_flip.save(save_path)
    return img.size


def augment_data(image_path_pattern: str, df: pd.DataFrame, output_file: str):
    """
    Augment a dataset with flipped images and merge the augmented data 
    with the original dataset, and save the resultant dataset to a specified file.
    
    Parameters
    ----------
    image_path_pattern : str
        Common path pattern to locate original images.
    df : pd.DataFrame
        DataFrame containing information about original images.
    output_file : str
        Name of the output file where the merged dataset will be saved.
    
    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "header_cols": [2, 2],
    ...     "label_width": [5, 5],
    ...     "className": [0.0, 0.0],
    ...     "XMin": [0.1, 0.2],
    ...     "YMin": [0.3, 0.4],
    ...     "XMax": [0.5, 0.6],
    ...     "YMax": [0.7, 0.8],
    ...     "ImagePath": ["path/to/image1.jpg", "path/to/image2.jpg"]
    ... })
    >>> augment_data("path/to/images", df, "output_file.lst")
    Original DataFrame length: 2
    Augmented DataFrame length: 2
    Merged DataFrame length: 4
    """
    # Extracting the common path and dataset type (train/test)
    common_path = os.path.commonpath([image_path_pattern])
    dataset_type = common_path.split('/')[-1]
    
    # Creating a temporary DataFrame to store augmented data
    temp_df = pd.DataFrame(columns=df.columns)
    
    for index, row in df.iterrows():
        # Extracting the image ID from the ImagePath
        img_id = extract_id_from_path(row["ImagePath"])
        img_path = f"{common_path}/{img_id}.jpg"
        
        # Loading the image
        img = Image.open(img_path)
        image_width, image_height = img.size
        
        # Performing the augmentation - flipping
        img_flip = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        
        # Constructing the new ImagePath and saving the flipped image
        new_image_path_for_saving = f"{common_path}/aug_{img_id}.jpg"
        new_image_path_for_df = f"airplanes/images/{dataset_type}/aug_{img_id}.jpg"
        
        img_flip.save(new_image_path_for_saving)
        
        # Calculating new coordinates
        xmin = row['XMin'] * image_width
        xmax = row['XMax'] * image_width
        new_xmin = ((image_width/2)-(xmin-(image_width/2))) / image_width
        new_xmax = ((image_width/2)-(xmax-(image_width/2))) / image_width
        
        # Creating a new row with augmented data
        new_row = pd.Series({
            "header_cols": row["header_cols"],
            "label_width": row["label_width"],
            "className": row["className"],
            "XMin": new_xmin,
            "YMin": row["YMin"],
            "XMax": new_xmax,
            "YMax": row["YMax"],
            "ImagePath": new_image_path_for_df
        })
        
        # Appending the new row to the temporary DataFrame
        temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Merging the original and augmented DataFrames
    df_merged = pd.concat([df, temp_df], ignore_index=True)
    df_merged.reset_index(drop=True, inplace=True)

    
    # Saving the merged DataFrame to the specified output file
    df_merged.to_csv(f"data/lst_files/{output_file}", sep="\t", float_format="%.4f", index=True, header=False)
    df_merged.to_csv(f"data/augmented_csv/{output_file}.csv", index=True)
    
    print(f"Original DataFrame length: {len(df)}")
    print(f"Augmented DataFrame length: {len(temp_df)}")
    print(f"Merged DataFrame length: {len(df_merged)}")
    
    
    
    

def prepare_data_for_tensorflow(df, output_directory, base_image_path="mouse_detection/images/", category=1):
    """
    Prepares data in the required format for TensorFlow Object Detection.

    Parameters:
    - df (pd.DataFrame): DataFrame containing bounding box data.
    - output_directory (str): Directory where the 'images' subdirectory and 'annotations.json' will be created.
    - base_image_path (str): Base path to where the images are located.
    - label (str): The label for the bounding boxes, default is "mouse".

    Returns:
    None. Creates the necessary directory structure and annotations.json file.
    """
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    images_dir = os.path.join(output_directory, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    images = []
    annotations = []
    
    for index, row in df.iterrows():
        image_id = index

        image_file_path = os.path.join(base_image_path, row["ImagePath"])
        width = (row["XMax"] - row["XMin"]) * 1920  
        height = (row["YMax"] - row["YMin"]) * 1080  
        

        images.append({
            "file_name": os.path.basename(image_file_path),
            "height": height,
            "width": width,
            "id": image_id
        })
        

        annotations.append({
            "image_id": image_id,
            "bbox": [row["XMin"], row["YMin"], row["XMax"], row["YMax"]],
            "category_id": category
        })
        
        # Copying the image to the 'images' directory
        shutil.copy(image_file_path, images_dir)
    
    # Write the 'annotations.json' file
    with open(os.path.join(output_directory, "annotations.json"), "w") as json_file:
        json.dump({"images": images, "annotations": annotations}, json_file)

# Note: The function is defined but not executed.
