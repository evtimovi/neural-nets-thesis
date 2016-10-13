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

def fnmr_from_distributions(genuine, imposter, threshold):
    '''
    Calculates the False Non-match Rate (False Reject Rate)
    from the given genuine and imposter distributions at the
    given threshold.
    The computation is based on Area-Under-the-Curve (AUC) ratios.

    Args:
        genuine: the genuine distribution in (x_values, y_values) tuple format
        imposter: the imposter distribution in (x_values, y_values) tuple format
        threshold: a value that cuts off genuines from imposters, should be present
                   in both distributions' ranges of x_values

    Returns:
        a float indicating the FNMR rate

    Questions:
    1. Does it matter if the distributions are "normed" (y_values are frequencies)?
    2. How do we use threshold if it is not present as an exact member of 
       x_values for both distributions? Maybe not a problem...
    '''
    return -1.0

def fmr_from_distributions(genuine, imposter, threshold):
    '''
    Calculates the False Match Rate (False Accept Rate, False positive rate)
    from the given genuine and imposter distributions at the
    given threshold.
    The computation is based on Area-Under-the-Curve (AUC) ratios.

    Args:
        genuine: the genuine distribution in (x_values, y_values) tuple format
        imposter: the imposter distribution in (x_values, y_values) tuple format
        threshold: a value that cuts off genuines from imposters, should be present
                   in both distributions' ranges of x_values

    Returns:
        a float indicating the FMR rate
   '''
    return -1.0

def gar_from_distributions(genuine, imposter, threshold):
    '''
    Calculates the Genuine Accept Rate (True Accept Rate, True Positive Rate)
    from the given genuine and imposter distributions at the
    given threshold.
    The computation is based on Area-Under-the-Curve (AUC) ratios.

    Args:
        genuine: the genuine distribution in (x_values, y_values) tuple format
        imposter: the imposter distribution in (x_values, y_values) tuple format
        threshold: a value that cuts off genuines from imposters, should be present
                   in both distributions' ranges of x_values

    Returns:
        a float indicating the GAR rate
   '''
    return -1.0

def fnmr_by_counting(ground_truth, similarity, threshold):
    '''
    Calculates the False Non-match Rate (False Reject Rate)
    by counting all false negaties
    (similarity below threshold AND ground_truth = positive)
    and then dividing by the sum total of all samples. ?????????????
    ?????? what should we divide by? 
    Note that len(ground_truth) must equal len(similarity)
    and values at the same index must correspond to the same pair.

    Args:
        ground_truth: an array of 0/1 values with 0 representing "same"
                      and 1 representing "different". These are 
                      the ground_truth values of the pairs at each index.
                      The index of a pair here must be the same as the 
                      index of the same pair in similarity.
        similarity: an array of values between 0 and 1 with 0 representing "same"
                    and 1 representing "different". These are the similarity
                    measures obtained by the classifier (e.g. Euclidean
                    distance between two feature vectors)
                    The index of a pair here must be the same as the 
                    index of the same pair in ground_truth.
        threshold: a value that cuts off genuines from imposters
    Returns:
        a float indicating the FNMR rate
   '''
    return -1.0

def fmr_by_counting(ground_truth, similarity, threshold):
    '''
    Calculates the False Match Rate (False Accept Rate, False positive rate)
    by counting all false positives
    (similarity above threshold AND ground_truth = negative)
    and then dividing by the sum total of all samples. ?????????????
    ?????? what should we divide by? 
    Note that len(ground_truth) must equal len(similarity)
    and values at the same index must correspond to the same pair.

    Args:
        ground_truth: an array of 0/1 values with 0 representing "same"
                      and 1 representing "different". These are 
                      the ground_truth values of the pairs at each index.
                      The index of a pair here must be the same as the 
                      index of the same pair in similarity.
        similarity: an array of values between 0 and 1 with 0 representing "same"
                    and 1 representing "different". These are the similarity
                    measures obtained by the classifier (e.g. Euclidean
                    distance between two feature vectors)
                    The index of a pair here must be the same as the 
                    index of the same pair in ground_truth.
        threshold: a value that cuts off genuines from imposters
    Returns:
        a float indicating the FMR rate
   '''
    return -1.0

def gar_by_counting(ground_truth, similarity, threshold):
    '''
    Calculates the Genuine Accept Rate (True Accept Rate, True Positive Rate)
    by counting all true positives 
    (similarity above threshold AND ground_truth = same)
    and then dividing by the sum total of all samples. ?????????????
    ?????? what should we divide by? sum total of all samples or TP+FN????
    Note that len(ground_truth) must equal len(similarity)
    and values at the same index must correspond to the same pair.

    Args:
        ground_truth: an array of 0/1 values with 0 representing "same"
                      and 1 representing "different". These are 
                      the ground_truth values of the pairs at each index.
                      The index of a pair here must be the same as the 
                      index of the same pair in similarity.
        similarity: an array of values between 0 and 1 with 0 representing "same"
                    and 1 representing "different". These are the similarity
                    measures obtained by the classifier (e.g. Euclidean
                    distance between two feature vectors)
                    The index of a pair here must be the same as the 
                    index of the same pair in ground_truth.
        threshold: a value that cuts off genuines from imposters
    Returns:
        a float indicating the GAR rate
   '''
    return -1.0

def equal_error_rate_from_counting(ground_truth, similarity, step=0.01, tolerance=0.01):
    '''
    Calculates the equal error rate by computing FMR and FNMR rates 
    for the similarity measures in similarity given ground_truth
    by varying the threshold in steps of step.
    "Equal" here is defined to mean within a difference of "threshold"

    Args:
        ground_truth: an array of 0/1 values with 0 representing "same"
                      and 1 representing "different". These are 
                      the ground_truth values of the pairs at each index.
                      The index of a pair here must be the same as the 
                      index of the same pair in similarity.
        similarity: an array of values between 0 and 1 with 0 representing "same"
                    and 1 representing "different". These are the similarity
                    measures obtained by the classifier (e.g. Euclidean
                    distance between two feature vectors)
                    The index of a pair here must be the same as the 
                    index of the same pair in ground_truth.
 
        step: how to vary the threshold at each step (default: 0.01)
        tolerance: defines maximum difference between two numbers
                   for them to be termed "equal" (default: 0.01)

    Returns:
        the FMR when the threshold is such that FMR=FNMR (within threshold)

    '''
    for t in range(0, np.amax(similarity), step):
        fmr = fmr_by_counting(ground_truth, similarity, t)
        fnmr = fnmr_by_counting(ground_truth, similarity, t)
        if (abs(fmr-fnmr) < tolerance):
            return fmr

    raise Exception("No fmr=fnmr found within tolerance of " + tolerance)

def equal_error_rate_from_distributions(genuine, imposter, step=0.01, tolerance=0.01):
    '''
    Calculates the equal error rate by computing FMR and FNMR rates 
    from the genuine and imposter distributions provided
    by varying the threshold in steps of step.
    "Equal" here is defined to mean within a difference of "threshold"

    Args:
        genuine: the genuine distribution in (x_values, y_values) tuple format
        imposter: the imposter distribution in (x_values, y_values) tuple format
        step: how to vary the threshold at each step (default: 0.01)
        tolerance: defines maximum difference between two numbers
                   for them to be termed "equal" (default: 0.01)

    Returns:
        the FMR when the threshold is such that FMR=FNMR (within threshold)

    '''
    for t in range(0, np.amax(similarity), step):
        fmr = fmr_from_distributions(genuine, imposter, t)
        fnmr = fnmr_from_distributions(genuine, imposter, t)
        if (abs(fmr-fnmr) < tolerance):
            return fmr

    raise Exception("No fmr=fnmr found within tolerance of " + tolerance)
