#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data Manipulation and Analysis
import pandas as pd
import numpy as np

# File and Directory Management
import zipfile
import glob
import shutil
import os
import os.path
from os import path

# Data Visualization and Plotting
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

# Image Manipulation
from PIL import Image

# Others
import random


# In[2]:


def extract_zip_to_folder(zip_file_paths: list):
    """
    Extracts the contents of the specified zip files to corresponding folders.

    Parameters:
    - zip_file_paths (list of str): List of paths to the zip files to be extracted.

    Returns:
    - None: The function performs the extraction operation and prints the progress.
    """

    for path in zip_file_paths:
        print(f'Processing {path}...')
        with zipfile.ZipFile(path, "r") as zip_ref:
            folder_name = path.split(".")[0]
            zip_ref.extractall(f'unzipped/{folder_name}')
        print(f'{path} completed!')


# In[3]:


def load_csv_as_dataset(file_path: str, delimiter: str = ',') -> pd.DataFrame:
    """
    Load a CSV file as a pandas DataFrame.
    
    Parameters:
    - file_path (str): Path to the CSV file.
    - delimiter (str, optional): Delimiter used in the CSV file. Defaults to ','.
    
    Returns:
    - pd.DataFrame: DataFrame containing the CSV data.
    """
    
    return pd.read_csv(file_path, delimiter=delimiter)


# In[4]:


def count_images(directory_path: str) -> int:
    """
    Count the number of images in the specified directory.
    
    Parameters:
    - directory_path (str): Path to the directory containing images (with wildcard for image format).
    
    Returns:
    - int: Total number of images in the directory.
    """
    
    return len(glob.glob(directory_path))


# In[5]:


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


# In[6]:


def filter_dataset(folder_path, df):
    """
    Filtra el dataframe para mantener solo las filas correspondientes 
    a las imágenes que existen en una carpeta específica.
    
    Parameters:
    - folder_path (str): Ruta al directorio que contiene las imágenes.
    - df (pandas.DataFrame): Dataframe que contiene una columna 'ImageID' con los nombres de las imágenes.
    
    Returns:
    - pandas.DataFrame: Dataframe filtrado, conteniendo solo las filas de imágenes existentes en la carpeta.
    
    Prints:
    - Número de filas eliminadas del dataframe original debido a la ausencia de imágenes en la carpeta.
    """
    
    existing_images = set([os.path.basename(img).split(".")[0] for img in glob.glob(f"{folder_path}*.jpg")])
    
    initial_length = len(df)
    df_filtered = df[df["ImageID"].isin(existing_images)]
    num_removed = initial_length - len(df_filtered)
    
    print(f"Number of rows removed: {num_removed}")
    return df_filtered



def filter_df_by_class(df: pd.DataFrame, column_name: str, target: str) -> pd.DataFrame:
    """
    Filter the rows of a DataFrame based on a specific value in a specified column.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame to be filtered.
    - column_name (str): Name of the column to be checked.
    - target_value (Any): The value to filter rows by in the specified column.
    
    Returns:
    - pd.DataFrame: A filtered DataFrame.
    """
    
    return df[df[column_name] == target]


# In[7]:


def visualize_random_image(images_folder: str) -> None:
    """
    Display a random image from the specified dataset split.
    
    Parameters:
    - split (str): A string indicating the dataset split. It can be either "train" or "validation".
    - images_folder (str): The path to the folder containing the images, with appropriate wildcards to match image files.
    
    Returns:
    None. Displays the randomly selected image using matplotlib.
    """
    
    images_paths = glob.glob(images_folder)
    num_of_images = len(images_paths)
    random_int = random.randint(0, num_of_images-1)
    random_image = images_paths[random_int]
    img = mpimg.imread(random_image)
    
    
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.show()


# In[8]:


def visualize_many(from_num: int, to_num: int, datasetpath: str, rows: int = 5, cols: int = 4) -> None:
    """
    Display a range of images from the specified dataset path in a grid.
    
    Parameters:
    - from_num (int): The starting index of the image range.
    - to_num (int): The ending index of the image range.
    - datasetpath (str): The path to the folder containing the images, with appropriate wildcards to match image files.
    - rows (int, optional): Number of rows in the grid. Defaults to 5.
    - cols (int, optional): Number of columns in the grid. Defaults to 4.
    
    Returns:
    None. Displays the selected images in a grid using matplotlib.
    """
    
    images_path = glob.glob(datasetpath)
    
    fig = plt.figure(figsize=(10,10))
    index_counter = 1
    for i in range(from_num, to_num):
        image = images_path[i]
        img = mpimg.imread(image)
        fig.add_subplot(rows, cols, index_counter)
        plt.imshow(img)
        index_counter += 1
    plt.show()


# In[9]:


def visualize_single_bbox(image_path: str, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    """
    Display an image with a bounding box overlay.
    
    Parameters:
    - image_path (str): Path to the image.
    - xmin (float): Minimum x-coordinate of the bounding box (as a proportion).
    - xmax (float): Maximum x-coordinate of the bounding box (as a proportion).
    - ymin (float): Minimum y-coordinate of the bounding box (as a proportion).
    - ymax (float): Maximum y-coordinate of the bounding box (as a proportion).
    
    Returns:
    None. Displays the image with the bounding box overlay using matplotlib.
    """
    
    im = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.imshow(im)

    image_width, image_height = im.size

    # Coordinates for bounding boxes
    new_xmin = xmin * image_width
    new_xmax = xmax * image_width
    new_ymin = ymin * image_height
    new_ymax = ymax * image_height

    width = new_xmax - new_xmin
    height = new_ymax - new_ymin

    rect = patches.Rectangle((new_xmin, new_ymin), width, height, linewidth=1, edgecolor="r", facecolor=(1, 0, 0, 0.3))
    ax.add_patch(rect)
    plt.show()


# In[10]:
    
    
def visualize_bb_with_data(image_directory: str, df, target: str) -> None:
    """
    Visualize a random image from a specified directory with bounding boxes for a specific target class.
    
    Parameters:
    - image_directory (str): Directory containing images.
    - df (pd.DataFrame): DataFrame containing bounding box data.
    - target (str): Target label (e.g., label_weapon).
    
    Returns:
    None. Displays the image with the bounding box(es) overlay using matplotlib.
    """
    
    images_paths = glob.glob(os.path.join(image_directory, "*.jpg"))
    if not images_paths:
        raise ValueError("No images found in the provided directory.")
    
    df_target = df[df.LabelName == target]
    
    grouped = df_target.groupby('ImageID').size()
    multiple_bboxes = grouped[grouped > 1].index
    
    if len(multiple_bboxes) == 0:
        raise ValueError("No images found with multiple bounding boxes for the target label.")
    
    eligible_images_paths = [path for path in images_paths if os.path.splitext(os.path.basename(path))[0] in multiple_bboxes]
    
    if not eligible_images_paths:
        raise ValueError("No eligible images found in the provided directory.")
    
    random_image = random.choice(eligible_images_paths)
    img = Image.open(random_image)
    id_of_image = os.path.splitext(os.path.basename(random_image))[0]  

    df_rows = df_target[df_target.ImageID == id_of_image]

    image_width, image_height = img.size
    fig, ax = plt.subplots()
    ax.imshow(img)

    for _, row in df_rows.iterrows():
        xmin, xmax = row['XMin'] * image_width, row['XMax'] * image_width
        ymin, ymax = row['YMin'] * image_height, row['YMax'] * image_height
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=3, edgecolor='r', facecolor=(1, 0, 0, 0.3))
        ax.add_patch(rect)
    plt.show()

