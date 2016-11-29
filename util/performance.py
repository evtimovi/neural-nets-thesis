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
        num_of_measures = len(values)
        
        # unique_counts returns the number of times each of the 
        # unique_values comes up in the original array
        unique_values, unique_counts = np.unique(values,
                                                 return_counts=True)
        
        if density:
            # make_frequencies will be a vectorized function
            # that, when applied to a numpy array 
            # will be evaluated element-wise
            make_frequencies = np.vectorize(lambda x: x/num_of_measures)
            frequencies = make_frequencies(unique_counts)
            return unique_values, frequencies
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
                      Major assumption: the ground truth labels are binary. 
                      If not, an exception is raised.
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
    # the assumption here is that ground_truth is binary
    # with the max value representing 'same' and the min value representing 'different'
    # the previous assumption that ground_truth won't be affected
    # if the cut-off was applied to it was wrong - 
    # that forced the entire ground truth into one class even when there were two of them
    # when the threshold was the maximum possible

    if len(np.unique(ground_truth)) != 2:
        raise Exception("ground truth must be binary") 

    same = np.amax(ground_truth)
    different = np.amin(ground_truth)
    cutoff = np.vectorize(lambda x: same if x >= threshold else different)
    predicted = cutoff(similarity)
#    print '**********scores', predicted
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for i in xrange(len(predicted)):
        if predicted[i] == same and ground_truth[i] == same:
            tp = tp + 1
        elif predicted[i] == same and ground_truth[i] == different:
            fp = fp + 1
        elif predicted[i] == different and ground_truth[i] == same:
            fn = fn + 1
        elif predicted[i] == different and ground_truth[i] == different:
            tn = tn + 1

    if (tp+fp+tn+fn) != len(predicted):
        raise Exception("confusion matrix entries don't add up")

    return tn, fp, fn, tp
    # see http://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
    # for details on confusion matrix usage
#    cm = skmet.confusion_matrix(ground_truth, predicted, [same,different])
#    return cm.ravel()


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
    bfm = binary_confusion_matrix(ground_truth, similarity, threshold)
    tn, fp, fn, tp = bfm
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
    bfm = binary_confusion_matrix(ground_truth, similarity, threshold)
    tn, fp, fn, tp = bfm
#    print '*******at threshold', threshold, 'tp=', tp, 'fp=', fp, 'fn=', fn, 'tn=', tn
    fmr = float(fp)/(float(tn)+float(fp))
#    print 'fmr', fmr
    return fmr


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
    bfm = binary_confusion_matrix(ground_truth, similarity, threshold)
    tn, fp, fn, tp = bfm

    #tn, fp, fn, tp = binary_confusion_matrix(ground_truth, similarity, threshold)
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

    map_fmr = np.vectorize(lambda t: fmr(ground_truth, similarity, t))
    map_fnmr = np.vectorize(lambda t: fnmr(ground_truth, similarity, t))

    # filter the similarities to leave away the maximum
    # possible similarity measure
    # if you don't do this, there is a case when there are no negatives)
    sim_sorted = np.unique(similarity)
#    max_sim = np.amax(similarity)
#    sim_sorted = filter(lambda x: x != max_sim, sim_sorted)

    fmrs = map_fmr(sim_sorted)
    fnmrs = map_fnmr(sim_sorted)
    
    # find all differences between fnmr and fmr
    # and return the minimum
    map_diff = np.vectorize(lambda x,y: abs(x-y))
    diffs = map_diff(fmrs,fnmrs)
    return fmrs[list(diffs).index(np.amin(diffs))]


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


def gar_at_zero_far_from_roc_curve(ground_truth, similarity):
    '''
    Computes the GAR at 0 FAR measure
    (indicating true positive rate given no false positive rates)
    given the scores in ground_truth and similarity.
    This method uses a generated ROC curve to compute the measure.
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
    
    # the far values have many 0s, so 
    # pick the last one
    i = 0
    while far_values[i] == 0:
        i += 1
    
    return gar_values[i]


def gar_at_zero_far_by_iterating(ground_truth, similarity):
    '''
    Computes the GAR at 0 FAR measure
    (indicating true positive rate given no false positive rates)
    given the scores in ground_truth and similarity.
    This method generates the GAR at 0 FAR by iterating through the 
    array of similarity measurements.
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
        the largest GAR at 0 FAR 
    '''


    sim_sorted = np.unique(similarity)
 
    map_gars = np.vectorize(lambda x: gar(ground_truth, similarity, x))
    map_fars = np.vectorize(lambda x: fmr(ground_truth, similarity, x))

    gars = map_gars(sim_sorted)
    fars = map_fars(sim_sorted)

#    print '*******fars', fars, 'given thresholds', sim_sorted

    gars_at_zero=[]
    for i in xrange(len(fars)):
        if fars[i] == 0:
            gars_at_zero.append(gars[i])

    return max(gars_at_zero) if len(gars_at_zero) > 0 else None
