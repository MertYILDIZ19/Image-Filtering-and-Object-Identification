import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import histogram_module
import dist_module

#converts 3 color channels into one single value (grey intensity)
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain 
#       handles to distance and histogram functions, and to find out whether histogram function 
#       expects grayvalue or color image

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

    #this value indicates if the rgb2gray() must be applied to the color array or not
    hist_isgray = histogram_module.is_grayvalue_hist(hist_type)

    #list of histograms for models
    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
    #list of histograms for query
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)
    
    #initializing matrix
    D = np.zeros((len(query_images), len(model_images)))
    #iterating query histograms
    for query_index in range(len(query_hists)):
        #comparing query histogram to all model histrograms
        for model_index in range(len(model_hists)):
            #computing distance between one query and one model
            D[query_index][model_index]=dist_module.get_dist_by_name(query_hists[query_index],model_hists[model_index],dist_type)
    # in the case in which dist_type is intersection the most similar image is the one with the biggest value (closest to 1)
    if dist_type == 'intersect':
        best_match = [np.argmax(D[index]) for index in range(len(query_hists))]
    
    # in the case of euclidean and chisq distances we want the lowest value of distance (most similar)
    elif dist_type != 'intersect':
        best_match = [np.argmin(D[index]) for index in range(len(query_hists))] 
    #best match type: list, D type: ndarray
    return best_match, D


def compute_histograms(image_list, hist_type, hist_isgray, num_bins):
    
    image_hist = []

    # Compute histogram for each image and add it at the bottom of image_hist
    for image in image_list:

        #Opening the image, converting it to type double and applying rgb2gray if hist_isgray parameter is True
        img_color = np.array(Image.open(image))
        if hist_isgray == True:
            img_gray = rgb2gray(img_color)
        else: 
            img_color = img_color.astype('double')

        #According to image_type, we compute the histogram by calling get_hist_by_name function
        if hist_type == 'grayvalue':
            image_hist.append(histogram_module.get_hist_by_name(img_gray,num_bins,hist_type))
        elif hist_type == 'rgb':
            image_hist.append(histogram_module.get_hist_by_name(img_color,num_bins,hist_type))
        elif hist_type == 'rg':
            image_hist.append(histogram_module.get_hist_by_name(img_color,num_bins,hist_type))
        elif hist_type == 'dxdy':
            image_hist.append(histogram_module.get_hist_by_name(img_gray,num_bins,hist_type))
        else:
            print ("Histogram type not valid")

    return image_hist



# For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# Note: use the previously implemented function 'find_best_match'
# Note: use subplot command to show all the images in the same Python figure, one row per query image

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):
     # we use the find best match function
    best,matrix=find_best_match(model_images,query_images, dist_type, hist_type, num_bins)
    num_nearest = 5
  
    #iterating through query images 
    for query_idx in range(len(query_images)):
        i=1 #initiating index for subplot position
        #picking indexes of best 5 matches for query
        models=matrix[query_idx]
        neighbors=models.argsort()[-num_nearest:][::-1] #takes indices from the end of the sorted array and selects last 5
        
        name=query_images[query_idx] #name of query image to be plotted
        plt.subplot(1,num_nearest+1,i) #indicating position of subplot where to plot query image
        plt.imshow(np.array(Image.open(name))); plt.title(f'Query {name}') #showing query image
        #iterating through the list if indexes of the num_nearest best matches
        for j in neighbors:
            i+=1 #incrementing index which indicates subplot position
            
            distance=models[j] #storing info on distance value
            name=model_images[j] #storing info on model image name
            plt.subplot(1,num_nearest+1,i) #indicating position of model subplot position
            plt.imshow(np.array(Image.open(name))); plt.title(distance) #showing model image       
        #one plot is generated for each query image
        plt.show()

