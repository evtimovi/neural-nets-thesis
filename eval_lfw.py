from util import performance as p
execfile('distributions.py')
true=[0 for _ in xrange(500)]
true.extend([-40 for _ in xrange(500)])
print 'equal error rate', p.equal_error_rate(true, simscores)
print 'gar at 0 far', p.gar_at_zero_far_by_iterating(true,simscores)
