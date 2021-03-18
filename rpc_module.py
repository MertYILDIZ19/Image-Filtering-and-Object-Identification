import numpy as np
import matplotlib.pyplot as plt

import histogram_module
import dist_module
import match_module



# compute and plot the recall/precision curve
#
# D - square matrix, D(i, j) = distance between model image i, and query image j
#
# note: assume that query and model images are in the same order, i.e. correct answer for i-th query image is the i-th model image

def plot_rpc(D, plot_color,dist_type):
    
    recall = []
    precision = []
    prec_rcll = {} # this sictionary is to save tresholds
    num_queries = D.shape[1]
    
    num_images = D.shape[0]
    assert(num_images == num_queries), 'Distance matrix should be a square matrix'
    
    labels = np.diag([1]*num_images)
      
    d = D.reshape(D.size)
    l = labels.reshape(labels.size)
    
    if dist_type=='intersect':      
        sortidx = d.argsort()[::-1][:d.shape[0]] #if distance is intersect we want values to reversed sorted list (high values give high precision)
    else:
        sortidx = d.argsort() # if distance is l2 or chi2 we sort it from small to high (small values give high precision)
    
    d = d[sortidx]
    l = l[sortidx]
    
    tp, fp, fn = 0, 0, np.sum(l)
    for idt in range(len(d)):
        tp += l[idt]
        fp += 1 - l[idt]
        fn -= l[idt] 
        
        #Compute precision and recall values and append them to "recall" and "precision" vectors
        precision.append(tp/(tp + fp))
        recall.append(tp/(tp + fn))
        prec_rcll[tp/(tp + fp)*tp/(tp + fn)]=d[idt] # new value in dicitonary that saves the product between recall and precision as key and treshold as value
    print('treshold',round(prec_rcll[max(prec_rcll)],4),' is the best one to maximise precision and recall with value', round(max(prec_rcll),4))
    plt.plot([1-precision[i] for i in range(len(precision))], recall, plot_color+'-')



def compare_dist_rpc(model_images, query_images, dist_types, hist_type, num_bins, plot_colors):
    
    assert len(plot_colors) == len(dist_types), 'number of distance types should match the requested plot colors'

    for idx in range( len(dist_types) ):

        print(dist_types[idx])
        [best_match, D] = match_module.find_best_match(model_images, query_images, dist_types[idx], hist_type, num_bins)

        plot_rpc(D, plot_colors[idx],dist_types[idx]) #I added one parameter (dist_type)
    

    plt.axis([0, 1, 0, 1]);
    plt.xlabel('1 - precision');
    plt.ylabel('recall');
    
    plt.legend( dist_types, loc='best')
