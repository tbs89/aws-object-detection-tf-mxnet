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

# Data Modeling and Splitting
from sklearn.model_selection import train_test_split

# Others
import random


# In[2]:


def clean_dataframe(df_for_train: pd.DataFrame, df_for_validation: pd.DataFrame, target: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Filters the provided training and validation DataFrames based on the target string in the 'LabelName' column.
    
    Parameters:
    - df_for_train (pd.DataFrame): The training DataFrame to be filtered.
    - df_for_validation (pd.DataFrame): The validation DataFrame to be filtered.
    - target (str): The target string to filter rows based on the 'LabelName' column.
    
    Returns:
    - df_train_cleansed (pd.DataFrame): Filtered training DataFrame.
    - df_validation_cleansed (pd.DataFrame): Filtered validation DataFrame.
    """
    
    df_train_cleansed = df_for_train[df_for_train.LabelName.str.contains(target)]
    df_validation_cleansed = df_for_validation[df_for_validation.LabelName.str.contains(target)]
    
    return df_train_cleansed, df_validation_cleansed


# In[3]:


def create_test_folder(test: pd.DataFrame, train: pd.DataFrame) -> (int, int):
    """
    Creates a folder structure for test images and organizes the images based on their ImageID.
    
    Parameters:
    - test (pd.DataFrame): The DataFrame containing information about test images.
    - train (pd.DataFrame): The DataFrame containing information about train images.
    
    Returns:
    - tuple: A tuple containing lengths of the test and train ImageID lists.
    """
    
    # Create the main directory for test images
    main_path = "unzipped/testImages"
    if not os.path.exists(main_path):
        os.mkdir(main_path)
    
    # Create a sub-directory for storing the actual images
    sub_path = os.path.join(main_path, "data")
    if not os.path.exists(sub_path):
        os.mkdir(sub_path)
    
    # Extract ImageID values from the test and train datasets
    test_img_ids = test.ImageID.values.tolist()
    train_img_ids = train.ImageID.values.tolist()
    
    return test_img_ids, train_img_ids


# In[4]:


def move_images_to_test_folder(test_img_ids: list, train_img_ids: list, source_folder: str, dest_folder: str) -> tuple:
    """
    Moves or copies images from the source folder to the test images folder based on their ImageID.
    
    Parameters:
    - test_img_ids (list): List of ImageID values for test dataset.
    - train_img_ids (list): List of ImageID values for train dataset.
    - source_folder (str): Path pattern where the source images are located.
    - dest_folder (str): Path to the destination folder for test images. 
    
    Returns:
    - tuple: A tuple containing the count of files copied and moved.
    """
    
    # Fetching all image paths from the source directory
    all_image_paths = glob.glob(source_folder)
    
    # Counters to keep track of the number of images copied and moved
    count_copy = 0
    count_move = 0
    
    # Loop through each image path and check its ImageID
    for f_path in all_image_paths:
        # Extract ImageID from the path using basename and removing the extension
        img_id = os.path.splitext(os.path.basename(f_path))[0]
        
        if img_id in test_img_ids:
            dest_image_path = os.path.join(dest_folder, f"{img_id}.jpg")
            
            if img_id in train_img_ids:
                # Copy the image if its ID is in both test and train datasets
                shutil.copy(f_path, dest_image_path)
                count_copy += 1
            else:
                # Move the image if its ID is only in the test dataset
                shutil.move(f_path, dest_image_path)
                count_move += 1
                
    return count_copy, count_move


# In[5]:


def save_datasets_to_csv(train_df: pd.DataFrame, test_df: pd.DataFrame, folder_path: str = "data/csv") -> None:
    """
    Saves the provided train and test datasets to the specified folder path.
    
    Parameters:
    - train_df (pd.DataFrame): The training dataset to be saved.
    - test_df (pd.DataFrame): The testing dataset to be saved.
    - folder_path (str, optional): The path to the folder where the datasets will be saved. Default is 'data/csv'.
    
    Returns:
    None
    """
    
    # Ensure the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Define the paths to save the CSV files
    train_csv_path = os.path.join(folder_path, "train.csv")
    test_csv_path = os.path.join(folder_path, "test.csv")
    
    # Save the DataFrames to CSV
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    
    print(f"Training dataset saved to {train_csv_path}")
    print(f"Test dataset saved to {test_csv_path}")

