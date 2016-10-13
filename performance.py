import numpy as np

'''
This module is designed to contain functions that measure
the performance of a neural network. Some of them are 
standard machine learning matrics and others are particular
to biometric authetication.
'''

def distribution(values, density=False, each_unique_in_bucket=False):
    '''
    This method returns a "histogram" representing the distribution
    of the values given. 

    It can be used to generate a genuine distribution or 
    an imposter distribution. For a genuine distribution,
    feed in similarity measures for ground truth positive ("match")
    pairs. For an imposter distribution, feed in ***similarity*** measures
    for ground truth negative pairs.
    
    Args:
        values: the values whose distribution we want to find
        density: whether to have the frequency 
                 or the number of occurences on the y axis
                 True: y axis has number of occurences
                 False: y axis has proportion of occurences
                 Defaults to False
        each_unique_in_bucket: specifies whether to place each unique
                               value in its own bucket
                               When False, the numpy auto method
                               for generating histogram buckets will be used
                               When True, each unique value will 
                               have its own bucket
                               Defaults to False

    Returns:
        a distribution curve in the format (x_values, y_values)
        Note that if the numpy histogram method is employed
        (each_unique_in_bucket = False), then x_values is just
        the borders of the histogram and is an array that is 1 bigger
        than x_values.
    '''
    if each_unique_in_bucket:
        num_of_measures = len(similarity_measures)
        
        # unique_counts returns the number of times each of the 
        # unique_values comes up in the original array
        unique_values, unique_counts = np.unique(similarity_measures,
                                            return_counts=True)
        
        if density:
            # make_frequencies will be a vectorized function
            # that, when applied to a numpy array 
            # will be evaluated element-wise
            make_frequencies = np.vectorize(lambda x: x/num_of_measures)
            frequencies = make_frequencies(unique_counts)
            return (unique_values, frequencies)
        else:
            return unique_values, unique_counts

    else:
        y_values, x_borders = np.histogram(values, bins='auto', density=density)
        return x_borders, y_values


