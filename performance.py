import numpy as np
import sklearn.metrics as skmet

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

def binary_confusion_matrix(ground_truth, similarity, threshold):
    '''
    Uses the threshold when applied to similarity to construct
    a confusion "matrix" for binary classes.
    Those above the threshold are considered matching,
    those below are considered non-matching.
    Args:
        ground_truth: an array of values with bigger values representing "more similar"
                      and smaller values representing "more different". These are 
                      the ground_truth values of the pairs at each index.
                      The index of a pair here must be the same as the 
                      index of the same pair in similarity.
        similarity: an array of values with bigger values representing "more similar"
                    and smaller values representing "more different". These are the similarity
                    measures obtained by the classifier.
                    Note that if using Euclidean distance, the values should be flipped
                    (because the convention of similar/different is reversed).
                    The index of a pair here must be the same as the 
                    index of the same pair in ground_truth.
        threshold: a value that cuts off genuines from imposters;
                  we classify those above the threshold as genuine
                  and those below as imposters (matching vs non-matching)
    Returns:
        a 4-way tuple representing tn, fp, fn, tp

    '''
    # force both ground_truth and similarity into classes based on threshold
    # assuming that this won't affect ground_truth
    # the labels "same" and "different" don't really matter;
    # I just call them that for convenience
    cutoff = np.vectorize(lambda x: "same" if x > threshold else "different")
    true = cutoff(ground_truth)
    predicted = cutoff(similarity)
    
    # see http://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
    # for details on confusion matrix usage
    return skmet.confusion_matrix(true, predicted).ravel()
 
def fnmr(ground_truth, similarity, threshold):
    '''
    Calculates the False Non-match Rate (False Reject Rate)
    by counting all false negaties
    (similarity below threshold AND ground_truth = match)
    and then dividing by the sum total of all ground truth match samples
    Note that len(ground_truth) must equal len(similarity)
    and values at the same index must correspond to the same pair.

    Args:
        ground_truth: an array of values with bigger values representing "more similar"
                      and smaller values representing "more different". These are 
                      the ground_truth values of the pairs at each index.
                      The index of a pair here must be the same as the 
                      index of the same pair in similarity.
        similarity: an array of values with bigger values representing "more similar"
                    and smaller values representing "more different". These are the similarity
                    measures obtained by the classifier.
                    Note that if using Euclidean distance, the values should be flipped
                    (because the convention of similar/different is reversed).
                    The index of a pair here must be the same as the 
                    index of the same pair in ground_truth.
        threshold: a value that cuts off genuines from imposters;
                  we classify those above the threshold as genuine
                  and those below as imposters (matching vs non-matching)
    Returns:
        a float indicating the FNMR rate
    '''
    tn, fp, fn, tp = binary_confusion_matrix(ground_truth, similarity, threshold)
    return float(fn)/(float(tp)+float(fn))

def fmr(ground_truth, similarity, threshold):
    '''
    Calculates the False Match Rate (False Accept Rate)
    by counting all false positives
    (similarity above threshold AND ground_truth = no match)
    and then dividing by the sum total of all ground truth "no match" samples
    Note that len(ground_truth) must equal len(similarity)
    and values at the same index must correspond to the same pair.

    Args:
        ground_truth: an array of values with bigger values representing "more similar"
                      and smaller values representing "more different". These are 
                      the ground_truth values of the pairs at each index.
                      The index of a pair here must be the same as the 
                      index of the same pair in similarity.
        similarity: an array of values with bigger values representing "more similar"
                    and smaller values representing "more different". These are the similarity
                    measures obtained by the classifier.
                    Note that if using Euclidean distance, the values should be flipped
                    (because the convention of similar/different is reversed).
                    The index of a pair here must be the same as the 
                    index of the same pair in ground_truth.
        threshold: a value that cuts off genuines from imposters;
                   we classify those above the threshold as genuine
                   and those below as imposters (matching vs non-matching)
    Returns:
        a float indicating the FMR rate
    '''
    tn, fp, fn, tp = binary_confusion_matrix(ground_truth, similarity, threshold)
    return float(fp)/(float(tn)+float(fp))

def gar(ground_truth, similarity, threshold):
    '''
    Calculates the Genuine Accept Rate (True Accept Rate)
    by counting all true positives
    (similarity above threshold AND ground_truth = match)
    and then dividing by the sum total of all ground truth "match" samples
    Note that len(ground_truth) must equal len(similarity)
    and values at the same index must correspond to the same pair.

    Args:
        ground_truth: an array of values with bigger values representing "more similar"
                      and smaller values representing "more different". These are 
                      the ground_truth values of the pairs at each index.
                      The index of a pair here must be the same as the 
                      index of the same pair in similarity.
        similarity: an array of values with bigger values representing "more similar"
                    and smaller values representing "more different". These are the similarity
                    measures obtained by the classifier.
                    Note that if using Euclidean distance, the values should be flipped
                    (because the convention of similar/different is reversed).
                    The index of a pair here must be the same as the 
                    index of the same pair in ground_truth.
        threshold: a value that cuts off genuines from imposters;
                   we classify those above the threshold as genuine
                   and those below as imposters (matching vs non-matching)
    Returns:
        a float indicating the GAR rate
    '''
    tn, fp, fn, tp = binary_confusion_matrix(ground_truth, similarity, threshold)
    return float(tp)/(float(tp)+float(fn))

def equal_error_rate(ground_truth, similarity):
    '''
    Calculates the equal error rate by computing FMR and FNMR rates 
    for the similarity measures in similarity given ground_truth
    by varying the threshold over each individual similarity measure.

    Args:
        ground_truth: an array of values with 
                      bigger values representing "more similar"
                      and smaller values representing "more different". These are 
                      the ground_truth values of the pairs at each index.
                      The index of a pair here must be the same as the 
                      index of the same pair in similarity.
        similarity: an array of values with bigger values representing "more similar"
                    and smaller values representing "more different". These are 
                    the similarity measures obtained by the classifier.
                    Note that if using Euclidean distance, 
                    the values should be flipped
                    (because the convention of similar/different is reversed).
                    The index of a pair here must be the same as the 
                    index of the same pair in ground_truth.

    Returns:
        the FMR when the threshold is such that FMR=FNMR (within threshold)
    '''
    # compute the fmr and fnmr taking each possible
    # similarity measure as a threshold t
    # sort the similarity measures along the way (might be redundant)
    map_fmr = np.vectorize(lambda t: fmr(ground_truth, similarity, t))
    map_fnmr = np.vectorize(lambda t: fnmr(ground_truth, similarity, t))
    sim_sorted = np.sort(similarity)
    fmrs = map_fmr(sim_sorted)
    fnmrs = map_fnmr(sim_sorted)
    
    #print 'fmrs', fmrs
    #print 'fnmrs', fnmrs
    # find all differences between fnmr and fmr
    # and return the minimum
    map_diff = np.vectorize(lambda x,y: abs(x-y))
    diffs = map_diff(fmrs,fnmrs)
    #print diffs
    return fmrs[list(diffs).index(np.amin(diffs))]

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

def roc_curve(ground_truth, similarity):
    '''
    Finds the roc curve that can be generated from these
    ground truths and similarity measures.
    Args:
        ground_truth: an array of values with 
                      bigger values representing "more similar"
                      and smaller values representing "more different". These are 
                      the ground_truth values of the pairs at each index.
                      The index of a pair here must be the same as the 
                      index of the same pair in similarity.
        similarity: an array of values with bigger values representing "more similar"
                    and smaller values representing "more different". These are 
                    the similarity measures obtained by the classifier.
                    Note that if using Euclidean distance, 
                    the values should be flipped
                    (because the convention of similar/different is reversed).
                    The index of a pair here must be the same as the 
                    index of the same pair in ground_truth.
    Returns:
        the ROC cuve in format (far_values, gar_values)
    '''
    x_values, y_values, thresholds = skmet.roc_curve(ground_truth, similarity)
    return x_values, y_values

def roc_auc_score(ground_truth, similarities):
    '''
    Finds the AUC score from the ROC curve that can be generated from these
    ground truths and similarity measures.
    Args:
        ground_truth: an array of values with 
                      bigger values representing "more similar"
                      and smaller values representing "more different". These are 
                      the ground_truth values of the pairs at each index.
                      The index of a pair here must be the same as the 
                      index of the same pair in similarity.
        similarity: an array of values with bigger values representing "more similar"
                    and smaller values representing "more different". These are 
                    the similarity measures obtained by the classifier.
                    Note that if using Euclidean distance, 
                    the values should be flipped
                    (because the convention of similar/different is reversed).
                    The index of a pair here must be the same as the 
                    index of the same pair in ground_truth.
    Returns:
        the ROC AUC score
    '''
    return skmet.roc_auc_score(ground_truth, similarities)

def gar_at_zero_far(ground_truth, similarity):
    '''
    Computes the GAR at 0 FAR measure
    (indicating true positive rate given no false positive rates)
    given the scores in ground_truth and similarity.
    If there is no FAR = 0, then the GAR at min FAR is returned.
    Args:
        ground_truth: an array of values with bigger values 
                      representing "more similar"
                      and smaller values representing "more different". These are 
                      the ground_truth values of the pairs at each index.
                      The index of a pair here must be the same as the 
                      index of the same pair in similarity.
        similarity: an array of values with bigger values 
                    representing "more similar"
                    and smaller values representing "more different". These are 
                    the similarity measures obtained by the classifier.
                    Note that if using Euclidean distance, 
                    the values should be flipped
                    (because the convention of similar/different is reversed).
                    The index of a pair here must be the same as the 
                    index of the same pair in ground_truth.


   Return:
        the GAR at min FAR measure
    '''
    far_values, gar_values = roc_curve(ground_truth, similarity)
    return gar_values[0]
