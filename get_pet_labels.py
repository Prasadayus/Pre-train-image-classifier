#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#                                                                             
# PROGRAMMER: Prasad Ayush
# DATE CREATED: 10/11/2023                                
# REVISED DATE: 12/11/2023
# PURPOSE: Create the function get_pet_labels that creates the pet labels from 
#          the image's filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##
# Imports python modules
from os import listdir

# TODO 2: Define get_pet_labels function below please be certain to replace None
#       in the return statement with results_dic dictionary that you create 
#       with this function
# 
def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels (results_dic) based upon the filenames 
    of the image files. These pet image labels are used to check the accuracy 
    of the labels that are returned by the classifier function, since the 
    filenames of the images contain the true identity of the pet in the image.
    Be sure to format the pet labels so that they are in all lower case letters
    and with leading and trailing whitespace characters stripped from them.
    (ex. filename = 'Boston_terrier_02259.jpg' Pet label = 'boston terrier')
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by the classifier function (string)
    Returns:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
      List. The list contains for following item:
         index 0 = pet image label (string)
    """
    # list items in directory with pet images
    file_list = listdir(image_dir)
   
    pet_labels = []
    
    # create empty dictionary    
    results_dict = dict()
    
    # iterate over each file name to save the pet name individually
    for index in range(0, len(file_list), 1):
        if file_list[index][0] != ".":
            pet_label = ''
            pet_img_filename = file_list[index]
            word_list_pet_img_filename = pet_img_filename.lower().split('_')
            pet_name = ''
            
            for word in word_list_pet_img_filename:
                if word.isalpha():
                    pet_name += word + " "
            
            pet_name = pet_name.strip()
            print('Filename is: ', pet_img_filename, '    label os: ', pet_name)
           
   
            # add every element of pet_labels as a value to the results_dic
            if file_list[index] not in results_dict:
                results_dict[file_list[index]] = [pet_name]
            else:
                print('\n warrning: key =' , filenames[index],  'already exists in results_dict with value =', results_dict[filenames[index]])
          
    print('\nPrinting all key-value pairs in dict results_dict:')
    for key in results_dict:
          print('\nfilename is: ', key, '  and  pet label is:', results_dict[key][0])
            
    # count length in full dictionary
    number_of_items_full_dict = len(results_dict)
    print('\n dictionary has {} items'.format(number_of_items_full_dict))
   
    return results_dict
