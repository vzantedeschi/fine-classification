import numpy as np

def compute_bins(bin_params):

    bins = []

    for start, stop, nb_bins in bin_params:  
        bins.append(np.linspace(start, stop, nb_bins))

    return bins

def distributions_from_labels(properties, labels, prop_bins, nb_classes=2):

    # distribution of fine classes P_p(Y = s)
    distr_s = []

    # list of distributions of property values P_p(C = c)
    distr_c = []

    # list of distribution per property P_p(C = c, Y = s)
    distr_joint = []

    for prop, bins in zip(properties, prop_bins):
        
        B = len(bins) - 1
        
        # ignore pixels with nan as property value in the stats. This is also why also P_p(s) depends on the property
        valid_idx = ~np.isnan(prop)
        valid_prop = prop[valid_idx]
        valid_labels = labels[valid_idx]
        nb_valid_points = len(valid_prop)
        
        hist_joint = np.zeros((nb_classes, B))
        hist_s = np.zeros(nb_classes)

        # get nb points per bin
        hist_c = np.histogram(valid_prop, bins=bins)[0]

        # loop over fine classes
        for s in range(nb_classes):

            s_labels = valid_labels == s
            hist_s[s] = np.sum(s_labels)

            hist_joint[s] = np.histogram(valid_prop[s_labels], bins=bins)[0]

        distr_joint.append(hist_joint / nb_valid_points)
        distr_s.append(hist_s / nb_valid_points)
        distr_c.append(hist_c / nb_valid_points)
        
    return distr_joint, distr_c, distr_s