from keras import backend as K
import tensorflow as tf

import numpy as np


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    false_negatives = possible_positives - true_positives
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*((p*r)/(p+r+K.epsilon()))


###USING THIS (NON-AVERAGED) JACCARD INDEX GIVES WORSE SCORES AS IT ONLY CARES ABOUT SLUM CLASSIFICATION, DOES NOT CARE ABOUT NON-SLUM.
###We shall use this as it tells us "worst case" pred ability on the slums, which is what we care about. we don't care about prediction acc on non-slum very much
###  "F score tends to measure something closer to average performance, while the IoU score measures something closer to the worst case performance."
#called SLUM_jaccard_index as it only connsiders slum as positive label classification. does NOT average over non-slum too.
def slum_jaccard_index(y_true, y_pred):
    f = f1(y_true, y_pred)
    j = f/(2-f)
    return j



from keras import backend as K
def kappa(y_true, y_pred):
    import tensorflow as tf
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    num_pred_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(tf.math.subtract(tf.ones_like(y_true), y_true), 0, 1)))
    fn = possible_positives - tp
    fp = num_pred_positives - tp
    tn = possible_negatives - fp
    p_0 = (tn+tp)/(tn+fp+fn+tp+ K.epsilon())
    P_a = ((tn+fp)/(tn+fp+fn+tp+ K.epsilon()))*((tn+fn)/(tn+fp+fn+tp+ K.epsilon()))
    P_b = ((fn+tp)/(tn+fp+fn+tp+ K.epsilon()))*((fp+tp)/(tn+fp+fn+tp+ K.epsilon()))
    pe = P_a + P_b
    k = (p_0-pe)/(1-pe+ K.epsilon())
    return k
    
    
    
def filter_pred(arr_, threshold=0.6):
    #takes in a numpy array of [0,1] entries , returns whether they are above/below the 0.5 threshold. Above 0.5 and they're "slum"
    
    arr = arr_
    super_threshold_indices = arr > threshold
    arr[super_threshold_indices] = 1
    sub_threshold_indices =  arr <= threshold
    arr[sub_threshold_indices] = 0
    return arr

def kwon_aleatoric(array_of_preds):
    ## calculates the aleatoric uncertainty as in https://towardsdatascience.com/what-uncertainties-tell-you-in-bayesian-neural-networks-6fbd5f85648e
    ## this is the approach of Kwon et al in https://openreview.net/pdf?id=Sk_P2Q9sG 
    ## Note that this simplifies significantly for the binary classifcication case. We simply want to average the values p_t(1-p_t) over all T models
    p = array_of_preds
    prod= np.multiply(p, 1-p)
    av = np.mean(prod, axis=0)
    return av

def kwon_epistemic(array_of_preds):
    ## calculates the aleatoric uncertainty as in https://towardsdatascience.com/what-uncertainties-tell-you-in-bayesian-neural-networks-6fbd5f85648e
    ## this is the approach of Kwon et al in https://openreview.net/pdf?id=Sk_P2Q9sG 
    ## Note that this simplifies significantly for the binary classifcication case. We simply want to average the values (p_t- p_bar)^2 over all T models
    p = array_of_preds
    p_bar = np.mean(p, axis=0)
    
    diff = np.subtract(p, p_bar)
    diff_sq = np.square(diff)
    
    aver = np.mean(diff_sq, axis=0)
    return aver


from keras import backend as K
def mask_recall(y_true, y_pred):
    mask = y_true < 2
    y_true_mask = tf.boolean_mask(y_true, mask)
    y_pred_mask = tf.boolean_mask(y_pred, mask)
    true_positives = K.sum(K.round(K.clip(y_true_mask * y_pred_mask, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true_mask, 0, 1)))
    false_negatives = possible_positives - true_positives
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def mask_precision(y_true, y_pred):
    mask = y_true < 2
    y_true_mask = tf.boolean_mask(y_true, mask)
    y_pred_mask = tf.boolean_mask(y_pred, mask)
    true_positives = K.sum(K.round(K.clip(y_true_mask * y_pred_mask, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred_mask, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def mask_bce(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
#     mask =tf.where( y_true > 2) ##this could be the wrong function
    mask = y_true < 2
    y_true_mask = tf.boolean_mask(y_true, mask)
    y_pred_mask = tf.boolean_mask(y_pred, mask)
    return bce(y_true_mask, y_pred_mask)

def mask_f1(y_true, y_pred):
    p = mask_precision(y_true, y_pred)
    r = mask_recall(y_true, y_pred)
    return 2*((p*r)/(p+r+K.epsilon()))
    
import tensorflow as tf
from sklearn.metrics import average_precision_score



import tensorflow as tf
from keras import backend as K


def mask_auroc(y_true, y_pred):
    auroc = tf.keras.metrics.AUC()
    mask = y_true < 2
    y_true_mask = tf.boolean_mask(y_true, mask)
    y_pred_mask = tf.boolean_mask(y_pred, mask)
    return auroc(y_true_mask, y_pred_mask)

def mask_auprc(y_true, y_pred):
    auprc = tf.keras.metrics.AUC(curve = "PR")
    mask = y_true < 2
    y_true_mask = tf.boolean_mask(y_true, mask)
    y_pred_mask = tf.boolean_mask(y_pred, mask)
    return auprc(y_true_mask, y_pred_mask)




# def mask_acc(y_true, y_pred):
#     mask = y_true < 2
#     y_true_mask = tf.boolean_mask(y_true, mask)
#     y_pred_mask = tf.boolean_mask(y_pred, mask)
#     acc = tf.keras.metrics.Accuracy()
#     return acc(y_true_mask, y_pred_mask)

def mask_acc(y_true, y_pred):
    y_true_class = K.argmax(y_true, axis=-1)
    y_pred_class = K.argmax(y_pred, axis=-1)

    mask = K.cast(y_true < 2, 'int32')
    matches = K.cast(K.equal(y_true, y_pred), 'int32') * mask
    accuracy = K.sum(matches) / K.maximum(K.sum(mask), 1)
    return accuracy


## simple explicit model weighted by uncertainty:
def weighted_model(tensor):
    pred_VHR, alea_VHR, epi_VHR, pred_LR, alea_LR, epi_LR = tensor[:,:,0], tensor[:,:,1], tensor[:,:,2], tensor[:,:,3], tensor[:,:,4], tensor[:,:,5]
    prop_VHR_unc = (alea_VHR + epi_VHR)/(alea_VHR + epi_VHR + alea_LR + epi_LR + tf.keras.backend.epsilon()) ## is this a good weighting?! generally epi smaller than alea
    prop_LR_unc = 1 - prop_VHR_unc
    
    weighting_VHR = 1 - prop_VHR_unc #want more weight for model with less uncertainty 
    weighting_LR = 1 - prop_LR_unc #want more weight for model with less uncertainty 
    comb_pred = weighting_VHR*pred_VHR + weighting_LR*pred_LR 
    comb_alea = weighting_VHR*alea_VHR + weighting_LR*alea_LR
    comb_epi = weighting_VHR*epi_VHR + weighting_LR*epi_LR
    return comb_pred, comb_alea, comb_epi


def relabel_3_to_0(tensor):
    # make the labels only have class 0 and 1 ( no 2s or  3s for don't know included )
    mask = tensor > 1 
    tensor_int = tf.cast(tensor, dtype = tf.int32)
    relabeled = tf.where(mask , tf.zeros_like(tensor_int), tensor_int)
    return relabeled
        


## EPISTEMIC UNCERTAINTY: UNet model weights are not set in stone different samples of model weights will result in different predictions
## ALEOTORIC UNCERTAINTY: size of 

## Make Epistemic unncertainty clear: rerun 100 different models with different sampled model weights. plot a heat with intensity as proportion of models which have voted for that pixel to be "slum"

def predictive_entropy(array_of_preds):
    ##takes as input a numpy array of different predictions from different instances of same Bayesian model
    ## this is an approximation from (2) in https://arxiv.org/pdf/1811.12709.pdf
    ## this captures both epistemic and aleatoric uncertainty
    ## NOTE THAT WE USE NATURAL LOG HERE, NOT LOG_2
    mean_c_1 = np.mean(array_of_preds, axis = 0) ## as the softmax output is probability of slum
    log_mean_c_1 = np.log(mean_c_1)
    log_mean_c_1[np.isinf(log_mean_c_1)] = 0 #if it's -inf then the prob will be zero so gets multiplied by zero anyway
    mean_c_0 = 1 - mean_c_1 ## using the fact that there are only two classes
    log_mean_c_0 = np.log(mean_c_0)
    log_mean_c_0[np.isinf(log_mean_c_0)] = 0 #if it's -inf then the prob will be zero so gets multiplied by zero anyway
    entropy_array = - ( np.multiply(mean_c_1,log_mean_c_1) + np.multiply(mean_c_0,log_mean_c_0) )
    return entropy_array


def mutual_info(array_of_preds):
    ##takes as input a numpy array of different predictions from different instances of same Bayesian model
    ## this is an approximation from (3) in https://arxiv.org/pdf/1811.12709.pdf
    ## NOTE THAT WE USE NATURAL LOG HERE, NOT LOG_2
    entropy =  predictive_entropy(array_of_preds)
    ## this captures epistemic uncertainty
    
    p_1 = array_of_preds
    p_0 = 1 - p_1
    
    log_1 = np.log(p_1)
    log_1[np.isinf(log_1)] = 0 #if it's -inf then the prob will be zero so gets multiplied by zero anyway

    log_0 = np.log(p_0)
    log_0[np.isinf(log_0)] = 0 #if it's -inf then the prob will be zero so gets multiplied by zero anyway
    
    summand = np.multiply(p_1, log_1) + np.multiply(p_0, log_0)
    second_term = np.mean(summand, axis=0)
    
    return entropy + second_term   
#     return -second_term ###do not include entropy term as it takes account of 



## currently unused. 
def aleatoric(array_of_preds):
    ##takes as input a numpy array of different predictions from different instances of same Bayesian model
    ## this is an approximation from (3) in https://arxiv.org/pdf/1811.12709.pdf
#     entropy =  predictive_entropy(array_of_preds)
    ## this captures epistemic uncertainty
    
    p_1 = array_of_preds
    p_0 = 1 - p_1
    
    log_1 = np.log(p_1)
    log_1[np.isinf(log_1)] = 0 #if it's -inf then the prob will be zero so gets multiplied by zero anyway

    log_0 = np.log(p_0)
    log_0[np.isinf(log_0)] = 0 #if it's -inf then the prob will be zero so gets multiplied by zero anyway
    
    summand = np.multiply(p_1, log_1) + np.multiply(p_0, log_0)
    second_term = np.mean(summand, axis=0)
    
#     return entropy + second_term   
    return -second_term ###do not include entropy term as it takes account of



## Calculate MMD:
def MMD_sq(kernel_matrix,num_tiles_s1, num_tiles_s2):
    ## num_tiles_s1 and _s2 are the number of samples in each sample. allows us to use unequal number and get a value for e.g. one test point from a whole swathe of training data
    term1 = (1/(num_tiles_s1*(num_tiles_s1-1)))*(np.sum(kernel_matrix[:num_tiles_s1,:num_tiles_s1]) - np.trace(kernel_matrix[:num_tiles_s1,:num_tiles_s1]))

    
    if num_tiles_s2 == 1:
        term2 = kernel_matrix[num_tiles_s1:,num_tiles_s1:]
        term3 = (2/(num_tiles_s1))*(np.sum(kernel_matrix[num_tiles_s1:,:num_tiles_s1]))

    
    else:
        term2 = (1/(num_tiles_s2*(num_tiles_s2-1)))*(np.sum(kernel_matrix[num_tiles_s1:,num_tiles_s1:]) - np.trace(kernel_matrix[num_tiles_s1:,num_tiles_s1:]))
        term3 = (2/(num_tiles_s1*num_tiles_s2))*(np.sum(kernel_matrix[num_tiles_s1:,:num_tiles_s1]))


    MMD_squared = term1 + term2 - term3
    return MMD_squared




### outline boundary plotting utilities

from matplotlib.collections import LineCollection

def get_all_boundary_edges(bool_img):
    """
    Get a list of all edges
    (where the value changes from 'True' to 'False') in the 2D image.
    Return the list as indices of the image.
    """
    ij_boundary = []
    ii, jj = np.nonzero(bool_img)
    for i, j in zip(ii, jj):
        # North
        if j == bool_img.shape[1]-1 or not bool_img[i, j+1]:
            ij_boundary.append(np.array([[i, j+1],
                                         [i+1, j+1]]))
        # East
        if i == bool_img.shape[0]-1 or not bool_img[i+1, j]:
            ij_boundary.append(np.array([[i+1, j],
                                         [i+1, j+1]]))
        # South
        if j == 0 or not bool_img[i, j-1]:
            ij_boundary.append(np.array([[i, j],
                                         [i+1, j]]))
        # West
        if i == 0 or not bool_img[i-1, j]:
            ij_boundary.append(np.array([[i, j],
                                         [i, j+1]]))
    if not ij_boundary:
        return np.zeros((0, 2, 2))
    else:
        return np.array(ij_boundary)




def close_loop_boundary_edges(xy_boundary, clean=True):
    """
    Connect all edges defined by 'xy_boundary' to closed 
    boundary lines.
    If not all edges are part of one surface return a list of closed 
    boundaries is returned (one for every object).
    """

    boundary_loop_list = []
    while xy_boundary.size != 0:
        # Current loop
        xy_cl = [xy_boundary[0, 0], xy_boundary[0, 1]]  # Start with first edge
        xy_boundary = np.delete(xy_boundary, 0, axis=0)

        while xy_boundary.size != 0:
            # Get next boundary edge (edge with common node)
            ij = np.nonzero((xy_boundary == xy_cl[-1]).all(axis=2))
            if ij[0].size > 0:
                i = ij[0][0]
                j = ij[1][0]
            else:
                xy_cl.append(xy_cl[0])
                break

            xy_cl.append(xy_boundary[i, (j + 1) % 2, :])
            xy_boundary = np.delete(xy_boundary, i, axis=0)

        xy_cl = np.array(xy_cl)

        boundary_loop_list.append(xy_cl)

    return boundary_loop_list

def plot_world_outlines(bool_img, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    ij_boundary = get_all_boundary_edges(bool_img=bool_img)
    xy_boundary = ij_boundary - 0.5
    xy_boundary = close_loop_boundary_edges(xy_boundary=xy_boundary)
    cl = LineCollection(xy_boundary, **kwargs)
    ax.add_collection(cl)

    
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow.python.keras.backend import eager_learning_phase_scope
import tensorflow.keras.backend as K   
    
def bayesian_pred_visualiser(dataset, num_tiles, bayesian_model, n_models = 100, data_type="VHR"):
    i = 0
    for element in dataset:
        img_tf, trg_tf, identifier = element
        img, trg, = img_tf.numpy(), trg_tf.numpy()
        pred_fn = K.function([bayesian_model.input], [bayesian_model.output])
        with eager_learning_phase_scope(value=1):
            all_preds = np.array([pred_fn([img]) for _ in range(n_models)])
#         all_preds = bayesian_model(img)
        overall_pred_ = np.mean(all_preds, axis=0)
        overall_pred = filter_pred(overall_pred_, 0.6) ### use 0.6 as that is what we showed is the optimal threshol
        mut_inf = mutual_info(all_preds) ## captures epistemic uncertainty
        pred_ent = predictive_entropy(all_preds) ## captures epistemic uncertainty
        kwon_epi = kwon_epistemic(all_preds) ## using kwon 2018 paper
        kwon_alea = kwon_aleatoric(all_preds)  ## using kwon 2018 paper
        if data_type == "VHR":
            fig, ax = plt.subplots(1,5, figsize=(20, 4))
            ax[0].set_title("Image")
            ax[1].set_title("Truth")
            ax[2].set_title("Bayesian UNet Prediction")

            ax[3].set_title("Aleatoric Uncertainty")
            ax[4].set_title("Epistemic Uncertainty")

            ax[0].imshow(img.astype(np.uint8)[:,:,:])
            plot_world_outlines(trg.astype(np.uint8)[:,:].T, ax=ax[0], color = 'm', label = "Truth")
            plot_world_outlines(overall_pred.astype(np.uint8)[0,0,:,:,0].T, ax=ax[0], color = 'c', label = "Predicted")
            ax[1].imshow(trg.astype(np.uint8)[:,:])
            ax[2].imshow(overall_pred.astype(np.uint8)[0,0,:,:,0])
            ax[0].legend()

            divider3 = make_axes_locatable(ax[3])
            cax3 = divider3.append_axes('right', size='5%', pad=0.1)
            im3 = ax[3].imshow(kwon_alea[0,0,:,:,0])
            fig.colorbar(im3, cax=cax3, orientation='vertical')
            
            divider4 = make_axes_locatable(ax[4])
            cax4 = divider4.append_axes('right', size='5%', pad=0.1)
            im4 = ax[4].imshow(kwon_epi[0,0,:,:,0])
            fig.colorbar(im4, cax=cax4, orientation='vertical')


            ax[0].set_xticks([], [])
            ax[1].set_xticks([], [])
            ax[2].set_xticks([], [])
            ax[3].set_xticks([], [])
            ax[4].set_xticks([], [])
#             ax[5].set_xticks([], [])
#             ax[6].set_xticks([], [])
            ax[0].set_yticks([], [])
            ax[1].set_yticks([], [])
            ax[2].set_yticks([], [])
            ax[3].set_yticks([], [])
            ax[4].set_yticks([], [])
#             ax[5].set_yticks([], [])
#             ax[6].set_yticks([], [])
            
        elif data_type == "LR":
            fig, ax = plt.subplots(1,4, figsize=(16, 4))
            ax[0].set_title("Truth")
            ax[1].set_title("Bayesian UNet Prediction")
            ax[2].set_title("Pred_Ent (Epi+Alea)")
            ax[3].set_title("Mut_Info (Epi Only)")

            ax[0].imshow(trg.astype(np.uint8)[0,:,:])
            ax[1].imshow(overall_pred.astype(np.uint8)[0,0,:,:,0])

            divider2 = make_axes_locatable(ax[2])
            cax2 = divider2.append_axes('right', size='5%', pad=0)
            im2 = ax[2].imshow(pred_ent[0,0,:,:,0])
            fig.colorbar(im2, cax=cax2, orientation='vertical')

            divider3 = make_axes_locatable(ax[3])
            cax3 = divider3.append_axes('right', size='5%', pad=0)
            im3 = ax[3].imshow(mut_inf[0,0,:,:,0])
            fig.colorbar(im3, cax=cax3, orientation='vertical')


            ax[0].set_xticks([], [])
            ax[1].set_xticks([], [])
            ax[2].set_xticks([], [])
            ax[3].set_xticks([], [])
            ax[0].set_yticks([], [])
            ax[1].set_yticks([], [])
            ax[2].set_yticks([], [])
            ax[3].set_yticks([], [])
            
        
            
        fig.tight_layout()    
        plt.show()
        if i>= num_tiles:
            break
        i+=1
        
        
    
    

    
from tensorflow.python.keras.backend import eager_learning_phase_scope
import tensorflow.keras.backend as K

def uncertainty_hist(dataset, num_tiles, bayesian_model, n_models = 100):    
    i = 0
    false_negs_alea_list = []
    false_negs_epi_list = []
    true_pos_alea_list = []
    true_pos_epi_list = []
    for element in dataset:
        img_tf, trg_tf, identifier = element
        img, trg, = img_tf.numpy(), trg_tf.numpy()
        pred_fn = K.function([bayesian_model.input], [bayesian_model.output])
        with eager_learning_phase_scope(value=1):
            all_preds = np.array([pred_fn([img]) for _ in range(n_models)])
        overall_pred_ = np.mean(all_preds, axis=0)
        overall_pred = filter_pred(overall_pred_, 0.6) ### use 0.6 as that is what we showed is the optimal threshol
        kwon_epi = kwon_epistemic(all_preds) ## using kwon 2018 paper
        kwon_alea = kwon_aleatoric(all_preds)  ## using kwon 2018 paper
        
        y_times_y_hat = np.multiply(overall_pred, trg)
        one_min_y_hat = 1 - overall_pred
        y_times_one_min_y_hat = np.multiply(one_min_y_hat, trg)
        true_pos_alea = np.multiply(y_times_y_hat, kwon_alea)
        true_pos_epi = np.multiply(y_times_y_hat, kwon_epi)
        false_neg_alea = np.multiply(y_times_one_min_y_hat, kwon_alea)
        false_neg_epi = np.multiply(y_times_one_min_y_hat, kwon_epi)
        
        
        true_pos_alea_list += np.squeeze(true_pos_alea[np.nonzero(true_pos_alea)]).tolist()
        true_pos_epi_list += np.squeeze(true_pos_epi[np.nonzero(true_pos_epi)]).tolist()
        false_negs_alea_list += np.squeeze(false_neg_alea[np.nonzero(false_neg_alea)]).tolist()
        false_negs_epi_list += np.squeeze(false_neg_epi[np.nonzero(false_neg_epi)]).tolist() 
         
        
        
            
        if i>= num_tiles:
            break
        i+=1
        
    
    bins_alea = np.linspace(0, 0.25, 250)
    bins_epi = np.linspace(0, 0.25, 250)
    fig, ax = plt.subplots(1,2, figsize=(20, 6))
#     ax[0].set_title("Epistemic Uncertainty")
#     ax[1].set_title("Aleatoric Uncertainty")
    
    ax[0].hist(true_pos_epi_list, bins_epi, alpha=0.5, label='True Positives', density =True)
    ax[0].hist(false_negs_epi_list, bins_epi, alpha=0.5, label='False Negatives', density =True)
    ax[1].hist(true_pos_alea_list, bins_alea, alpha=0.5, label='True Positives', density =True)
    ax[1].hist(false_negs_alea_list, bins_alea, alpha=0.5, label='False Negatives', density =True)
    ax[0].set_title("Epistemic Uncertainty Histogram")
    ax[1].set_title("Aleatoric Uncertainty Histogram")
#     ax[0].set_ylim(top = 700000)
#     ax[1].set_ylim(top = 150000)
    ax[0].set_ylabel("Frequency")
    ax[0].set_xlabel("Epistemic Uncertainty")
    ax[1].set_ylabel("Frequency")
    ax[1].set_xlabel("Aleatoric Uncertainty")
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    
    plt.show()
    
    
    
def spectral_hist(dataset, num_tiles, data_type = "VHR"):    
    if data_type == "VHR":
        red_slum = []
        red_not_slum = []
        green_slum = []
        green_not_slum = []
        blue_slum = []
        blue_not_slum = []
        
        ## collect info from bands
        i = 0
        for element in dataset:
            img_tf, trg_tf, identifier = element
            img, trg, = img_tf.numpy(), trg_tf.numpy()
            red = img[:,:,0]
            blue = img[:,:,1]
            green = img[:,:,2]
            
            
            red_slum += red[np.nonzero(trg)].tolist()
            red_not_slum += red[np.nonzero(1-trg)].tolist()
            green_slum += green[np.nonzero(trg)].tolist()
            green_not_slum += green[np.nonzero(1-trg)].tolist()
            blue_slum += blue[np.nonzero(trg)].tolist()
            blue_not_slum += blue[np.nonzero(1-trg)].tolist()

            if i>= num_tiles:
                break
            i+=1

        #plotting
        bins = np.linspace(0, 255, 256)
        fig, ax = plt.subplots(1,3, figsize=(24, 6))

        ax[0].hist(red_slum, bins, alpha=0.5, label='Slum', density =True)
        ax[0].hist(red_not_slum, bins, alpha=0.5, label='Not Slum', density =True)
        ax[1].hist(green_slum, bins, alpha=0.5, label='Slum', density =True)
        ax[1].hist(green_not_slum, bins, alpha=0.5, label='Not Slum', density =True)
        ax[2].hist(blue_slum, bins, alpha=0.5, label='Slum', density =True)
        ax[2].hist(blue_not_slum, bins, alpha=0.5, label='Not Slum', density =True)
        
        ax[0].set_title("Red Band Histogram")
        ax[1].set_title("Green Band Histogram")
        ax[2].set_title("Blue Band Histogram")
        ax[0].set_ylabel("Frequency")
        ax[1].set_ylabel("Frequency")
        ax[2].set_ylabel("Frequency")
        ax[0].set_xlabel("Red Intensity")
        ax[1].set_xlabel("Green Intensity")
        ax[2].set_xlabel("Blue Intensity")
        ax[0].legend(loc='upper right')
        ax[1].legend(loc='upper right')
        ax[2].legend(loc='upper right')

        plt.show()
        
    elif data_type == "LR":
#         bands = ['coastal-aerosol', 'blue','green','red', 'red-edge','red-edge-2','red-edge-3', 'nir','red-edge-4', 'water-vapor', 'cirrus', 'swir1','swir2']
        bands = ['coastal-aerosol', 'blue','green','red', 'red-edge','red-edge-2', 'nir','red-edge-4', 'water-vapor', 'cirrus', 'swir1','swir2'] ### missing red edge 3 as harry left it out

        num_bands = len(bands)
        all_pixel_data = [[[],[]] for band in bands]
        
        i = 0
        for element in dataset:
            img_tf, trg_tf, identifier = element
            img, trg, = img_tf.numpy(), trg_tf.numpy()
            print("Data from tile " + str(i) + " of "+ str(num_tiles))
            for j in range(num_bands):
                all_pixel_data[j][0] += (img[:,:,j][np.nonzero(trg)]*(255/6553)).tolist()
                all_pixel_data[j][1] += (img[:,:,j][np.nonzero(1-trg)]*(255/6553)).tolist()
                


            if i>= num_tiles:
                break
            i+=1
        
        
        
        #plotting
        bins = np.linspace(0, 255 , 256)
        fig, ax = plt.subplots(1,6, figsize=(30, 4))
        for k in range(6):
            band = bands[k]
            print("plotting band "+ str(k) + " slum")
            ax[k].hist(all_pixel_data[k][0], bins, alpha=0.5, label='Slum', density =True)
            print("plotting band "+ str(k) + " notslum")
            ax[k].hist(all_pixel_data[k][1], bins, alpha=0.5, label='Not Slum', density =True)
            ax[k].set_title(band +" Band Histogram")
            ax[k].set_ylabel("Frequency")
            ax[k].set_xlabel(band + " intensity")
            ax[k].legend(loc='upper right')
        fig.tight_layout()    
        plt.show()
        
        fig, ax = plt.subplots(1,6, figsize=(30, 4))
        for k in range(6):
            l = k+6
            band = bands[l]
            print("plotting band "+ str(l) + " slum")
            ax[k].hist(all_pixel_data[l][0], bins, alpha=0.5, label='Slum', density =True)
            print("plotting band "+ str(l) + " notslum")
            ax[k].hist(all_pixel_data[l][1], bins, alpha=0.5, label='Not Slum', density =True)
            ax[k].set_title(band +" Band Histogram")
            ax[k].set_ylabel("Frequency")
            ax[k].set_xlabel(band + " intensity")
            ax[k].legend(loc='upper right')
        fig.tight_layout()    
        plt.show()

        
        
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA


def PCA_plotter(dataset, num_tiles, data_type = "LR"):

    pca = PCA(n_components=2)

    all_tiles = []
    all_trgs = []
    i = 0
    for element in dataset:
        img_tf, trg_tf, identifier = element
        img, trg, = img_tf.numpy(), trg_tf.numpy()
        h,w,b = img.shape
        img_flat = img.reshape(h*w,b)
        all_tiles += img_flat.tolist()
        all_trgs += trg.reshape(h*w).tolist()

        if i>= num_tiles:
            break
        i+=1

    fig, ax = plt.subplots(1,1, figsize = (9,6))

    red_patch = mpatches.Patch(color='red', label='Slum')
    purple_patch = mpatches.Patch(color='darkmagenta', label='Not Slum')

    plt.legend(handles=[red_patch, purple_patch])

    projected = pca.fit_transform(np.array(all_tiles))
    ax.scatter(projected[:, 0], projected[:, 1],
            c=np.array(all_trgs), edgecolor='none', alpha=0.01,
            cmap=plt.cm.get_cmap('rainbow', 2))
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    # ax.legend()
    ax.set_title(data_type +" Data PC Projection")
    plt.plot()
    
    
    
def create_adversarial_pattern(input_image, input_label, model):
    loss_object = tf.keras.losses.BinaryCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = tf.squeeze(model(input_image))
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad



def adversarial_plot(iter_data, number_of_images_to_plot, model, epsilons=[0, 0.01, 0.05],  data_type = "VHR"):
    for n in range(number_of_images_to_plot):
        print("Tile {} of ".format(n) + str(number_of_images_to_plot) +":")
        img_tf, trg_tf = next(iter_data)
        img, trg, = img_tf.numpy(), trg_tf.numpy()
        descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input') for eps in epsilons]

        perturbations = create_adversarial_pattern(img_tf, trg_tf,model)    
        if data_type == "VHR":
            fig, ax = plt.subplots(1,1, figsize=(6, 6))
            ax.set_xticks([], [])
            ax.set_yticks([], [])
            ax.imshow(perturbations[0,:,:,:]*0.5+0.5) # To change [-1, 1] to [0,1]
            ax.set_title("Perturbation")
            plt.show()

        if data_type == "VHR":
            for i, eps in enumerate(epsilons):
                adv_x = tf.cast(img_tf[0] + 255.*eps*perturbations[0], tf.int32)    
                pred_ = model(adv_x)[0,:,:,0]
                fig, ax = plt.subplots(1,2, figsize=(14, 6))
                ax[0].imshow(adv_x)
                ax[0].set_title(descriptions[i])
                ax[0].set_xticks([], [])
                ax[0].set_yticks([], [])
                pred = filter_pred(np.array(pred_) , threshold = 0.6)
                ax[1].imshow(pred)
                ax[1].set_title("Prediction on " + descriptions[i])
                ax[1].set_xticks([], [])
                ax[1].set_yticks([], [])

        if data_type == "LR":
            n_epsilons = len(epsilons)
            fig, ax = plt.subplots(1,n_epsilons, figsize=(6*n_epsilons, 6))
            for i, eps in enumerate(epsilons):
                adv_x = tf.cast(img_tf[0] + 255.*eps*perturbations[0], tf.int32)    
                pred_ = model(adv_x)[0,:,:,0]
                pred = filter_pred(np.array(pred_) , threshold = 0.6)
                ax[i].imshow(pred)
                ax[i].set_title("Prediction on " + descriptions[i])
                ax[i].set_xticks([], [])
                ax[i].set_yticks([], [])
        plt.show()
        
        
import matplotlib.pyplot as plt        
        
def plot_all_metrics(dic):
    '''dic should be a .history training dictionary from calling model.train'''
    num_metrics = int(len(dic)/2) ##as it contains train and val metrics
    fig, ax = plt.subplots(1, num_metrics,  figsize=(4*num_metrics, 4), dpi = 300)
    fig.tight_layout()
    keys = list(dic.keys())
   
    
    for i in range(num_metrics):
        key = keys[i]
        ax[i].plot(dic[key], label='train')
        ax[i].plot(dic["val_"+key], label='val')
        ax[i].set_title("S2 LR "+ key)
        ax[i].set_xlabel('epoch')
        ax[i].set_ylabel(key)
    ax[0].legend(loc="upper right")
    fig.tight_layout()
    plt.savefig('100epoch_all_years_LR.png')
    plt.show()
        
        

        
        
        

def concat_datasets(datasets):
    ds0 = tf.data.Dataset.from_tensors(datasets[0])
    for ds1 in datasets[1:]:
        ds0 = ds0.concatenate(tf.data.Dataset.from_tensors(ds1))
    return ds0





### outline boundary plotting utilities

from matplotlib.collections import LineCollection

def get_all_boundary_edges(bool_img):
    """
    Get a list of all edges
    (where the value changes from 'True' to 'False') in the 2D image.
    Return the list as indices of the image.
    """
    ij_boundary = []
    ii, jj = np.nonzero(bool_img)
    for i, j in zip(ii, jj):
        # North
        if j == bool_img.shape[1]-1 or not bool_img[i, j+1]:
            ij_boundary.append(np.array([[i, j+1],
                                         [i+1, j+1]]))
        # East
        if i == bool_img.shape[0]-1 or not bool_img[i+1, j]:
            ij_boundary.append(np.array([[i+1, j],
                                         [i+1, j+1]]))
        # South
        if j == 0 or not bool_img[i, j-1]:
            ij_boundary.append(np.array([[i, j],
                                         [i+1, j]]))
        # West
        if i == 0 or not bool_img[i-1, j]:
            ij_boundary.append(np.array([[i, j],
                                         [i, j+1]]))
    if not ij_boundary:
        return np.zeros((0, 2, 2))
    else:
        return np.array(ij_boundary)




def close_loop_boundary_edges(xy_boundary, clean=True):
    """
    Connect all edges defined by 'xy_boundary' to closed 
    boundary lines.
    If not all edges are part of one surface return a list of closed 
    boundaries is returned (one for every object).
    """

    boundary_loop_list = []
    while xy_boundary.size != 0:
        # Current loop
        xy_cl = [xy_boundary[0, 0], xy_boundary[0, 1]]  # Start with first edge
        xy_boundary = np.delete(xy_boundary, 0, axis=0)

        while xy_boundary.size != 0:
            # Get next boundary edge (edge with common node)
            ij = np.nonzero((xy_boundary == xy_cl[-1]).all(axis=2))
            if ij[0].size > 0:
                i = ij[0][0]
                j = ij[1][0]
            else:
                xy_cl.append(xy_cl[0])
                break

            xy_cl.append(xy_boundary[i, (j + 1) % 2, :])
            xy_boundary = np.delete(xy_boundary, i, axis=0)

        xy_cl = np.array(xy_cl)

        boundary_loop_list.append(xy_cl)

    return boundary_loop_list

def plot_world_outlines(bool_img, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    ij_boundary = get_all_boundary_edges(bool_img=bool_img)
    xy_boundary = ij_boundary - 0.5
    xy_boundary = close_loop_boundary_edges(xy_boundary=xy_boundary)
    cl = LineCollection(xy_boundary, **kwargs)
    ax.add_collection(cl)

    
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow.python.keras.backend import eager_learning_phase_scope
import tensorflow.keras.backend as K   
    
def pred_visualiser(dataset, num_tiles, model):
    i = 0
    for element in dataset:
        img_tf, trg_tf = element
        img, trg, = img_tf.numpy(), trg_tf.numpy()
        pred_fn = K.function([model.input], [model.output])
        n_models = 1 # as it's frequentist, not dropout
        with eager_learning_phase_scope(value=1):
            all_preds = np.array([pred_fn([img]) for _ in range(n_models)])
#         all_preds = bayesian_model(img)
        overall_pred_ = np.mean(all_preds, axis=0)
        overall_pred = filter_pred(overall_pred_, 0.6) ### use 0.6 as that is what we showed is the optimal threshol

        fig, ax = plt.subplots(1,3, figsize=(20, 4))
        ax[0].set_title("Image")
        ax[1].set_title("Truth")
        ax[2].set_title("UNet Prediction")


        ax[0].imshow(img.astype(np.uint8)[0,:,:,:])
#         print(trg.shape)
        plot_world_outlines(trg.astype(np.uint8)[0,:,:].T, ax=ax[0], color = 'm', label = "Truth")
        
        plot_world_outlines(overall_pred.astype(np.uint8)[0,0,:,:,0].T, ax=ax[0], color = 'c', label = "Predicted")
        ax[1].imshow(trg.astype(np.uint8)[0,:,:])
        ax[2].imshow(overall_pred.astype(np.uint8)[0,0,:,:,0])
        ax[0].legend()


        ax[0].set_xticks([], [])
        ax[1].set_xticks([], [])
        ax[2].set_xticks([], [])

        ax[0].set_yticks([], [])
        ax[1].set_yticks([], [])
        ax[2].set_yticks([], [])


    
            
        
            
        fig.tight_layout()    
        plt.show()
        if i>= num_tiles:
            break
        i+=1
        