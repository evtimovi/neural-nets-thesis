from util import performance as p
import matplotlib.pyplot as plt 
execfile('foo/euclidean_dist_colorferet_1000.txt')

true = [1 for i in xrange(500)]
true.extend([0 for i in xrange(500)])

flipped_dist=map(lambda x: 1/x, euclidean_dist)

print 'The ROC AUC score (with original Euclidean distances) is', p.roc_auc_score(true, euclidean_dist)
print 'The ROC AUC score (with reciprocal Euclidean distances) is', p.roc_auc_score(true, flipped_dist)

#true2 = map(lambda x: 100000 if x == 1 else 0, true)
#for t in xrange(1,70,9):
#    tn, fp, fn, tp = p.binary_confusion_matrix(true2, dist_distribution, t)
#    print 'at threshold', t, 'tp =', tp, 'fp =', fp, 'fn =', fn, 'tn =', tn,
#    fmr = p.fmr(true2, dist_distribution, t)
#    print 'fmr =', fmr, 
#    fnmr = p.fnmr(true2, dist_distribution, t)
#    print 'fnmr =', fnmr,
#    gar = p.gar(true2, dist_distribution, t)
#    print 'gar =', gar

print 'the equal error rate is', p.equal_error_rate(true, flipped_dist)
print 'the gar at 0 far from the roc curve is', p.gar_at_zero_far_from_roc_curve(true, flipped_dist)
print 'the max gar at 0 far is', p.gar_at_zero_far_by_iterating(true, flipped_dist)

far_values, gar_values = p.roc_curve(true,flipped_dist)
plt.plot(far_values, gar_values)
plt.xlabel('far')
plt.ylabel('gar')
plt.title('ROC curve with true, flipped_dist')
plt.show()
