from util import processimgs as p
from scipy.spatial import distance
from vggface import networks as n
import os
import random
import sys

# ensure that no elements of the two arrays are at the same index
def index_match(arr1, arr2):
    for i in xrange(len(arr1)):
        if arr1[i] == arr2[i]:
            return True
    return False

subjects=[749,719,718,582,919,583,913,912,911,910,917,916,915,914,722,723,146,728,717,729,716,988,984,985,986,987,980,981,982,983,773,772,678,770,777,776,775,774,672,673,670,671,676,677,674,675,519,959,958,957,956,955,954,953,952,951,950,580,581,469,468,584,585,586,587,588,589,465,782,783,780,781,786,787,784,785,788,789,737,736,735,734,733,732,731,730,739,738,81,918,247,603,602,601,600,607,606,605,604,609,608,543,618,619,1011,1010,1012,904,905,906,907,900,901,902,903,908,909,758,880,760,761,647,646,645,644,643,642,641,640,648,885,884,887,886,768,769,883,882,764,765,766,767,889,888,762,763,508,500,501,505,948,949,940,941,942,943,944,945,946,947,474,470,472,473,579,578,575,574,577,576,571,570,573,572,939,938,935,934,937,936,931,930,933,932,720,721,689,688,724,725,726,727,683,682,681,680,687,686,685,684,157,93,254,256,755,754,757,756,751,750,753,752,614,615,616,617,610,611,612,613,971,879,531,533,532,1008,1009,1006,1007,1004,1005,1002,1003,1000,1001,878,970,973,972,974,977,976,979,978,876,877,809,805,804,807,806,801,800,803,802,193,771,650,652,679,654,655,656,657,658,659,896,897,894,895,892,893,890,891,711,710,713,712,715,714,898,899,108,107,268,778,625,624,627,626,621,620,623,622,629,628,568,569,566,567,564,565,562,563,779,928,929,926,927,924,925,922,923,920,921,498,491,694,695,697,690,691,692,693,698,699,140,992,991,990,997,996,995,994,999,998,383,746,747,744,745,742,743,740,741,661,660,663,662,665,664,667,666,522,962,963,960,961,966,967,964,965,968,969,597,596,595,594,593,592,591,590,599,598,993,791,790,793,792,795,794,797,796,799,798,816,817,814,815,812,813,810,811,70,71,708,709,703,700,701,706,707,704,705,669,668,636,637,634,635,632,633,630,631,638,639,653]
imposters = subjects[:]
while index_match(imposters, subjects):
    sys.stderr.write("shuffling imposters...")
    random.SystemRandom().shuffle(imposters)

path_base = "./datasets/feret-meb-vars/"

total_data = [] # map from subject to tuple (subjectid, fbdist, rcdist, imposterid, fbdist_imposter, rcdist_imposter)

network=n.VGGFaceVanilla()
network.load_weights("./output/rc_subjects/weights/weights_epoch_11_final.ckpt")

def calculate_fb_rc_for_subject(fa_id, fb_id, rc_id):
        # get the paths to the subject's folder
        fa_path = os.path.join(path_base,"fa", str(fa_id).zfill(5))
        fb_path = os.path.join(path_base,"fb", str(fb_id).zfill(5))
        rc_path = os.path.join(path_base,"rc", str(rc_id).zfill(5))

        # get the name of the image
        fa_img = sorted(os.listdir(fa_path))[0]
        fb_img = sorted(os.listdir(fb_path))[0]
        rc_img = sorted(os.listdir(rc_path))[0]

        # get the full path to the image
        fa_img_path = os.path.join(fa_path, fa_img)
        fb_img_path = os.path.join(fb_path, fb_img)
        rc_img_path = os.path.join(rc_path, rc_img)

        # get the vectors for the genuine images from the network
        favec = network.get_l2_vector([p.load_adjust_avg(fa_img_path),])
        fbvec = network.get_l2_vector([p.load_adjust_avg(fb_img_path),])
        rcvec = network.get_l2_vector([p.load_adjust_avg(rc_img_path),])
        
        # compute the distances between the fa and the genuine fb's and rc's
        distance_fb = distance.euclidean(favec, fbvec)
        distance_rc = distance.euclidean(favec, rcvec)
        return distance_fb, distance_rc

for i in range(len(subjects)):
    # get subject id and imposter id
    s = subjects[i]
    imposter = imposters[i]

    distance_fb, distance_rc = calculate_fb_rc_for_subject(s, s, s)
    distance_imposter_fb, distance_imposter_rc = calculate_fb_rc_for_subject(s, imposter, imposter)
    total_data.append((s, distance_fb, distance_rc, imposter, distance_imposter_fb, distance_imposter_rc))

with open('howfar_data.csv', 'w') as f:
    f.write('subject_id,dist_fa_to_genuine_fb,dist_fa_to_genuine_rc,imposter_id,dist_fa_to_imposter_fb,dist_fa_to_imposter_rc\n')
    for row in total_data:
        f.write(','.join(map(str,row))+'\n')
