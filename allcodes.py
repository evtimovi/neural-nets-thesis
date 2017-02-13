from util import processimgs as p
from scipy.spatial import distance
from vggface import networks as n
import os
import random
import sys
import json

# params
GPU = 2
WEIGHTS_PATH = "./output/rc_subjects/weights/weights_epoch_11_final.ckpt"

# ensure that no elements of the two arrays are at the same index
def index_match(arr1, arr2):
    for i in xrange(len(arr1)):
        if arr1[i] == arr2[i]:
            return True
    return False

path_base = "./datasets/feret-meb-vars/"
sample_n = 99
subjects=[749,719,718,582,919,583,913,912,911,910,917,916,915,914,722,723,146,728,717,729,716,988,984,985,986,987,980,981,982,983,773,772,678,770,777,776,775,774,672,673,670,671,676,677,674,675,519,959,958,957,956,955,954,953,952,951,950,580,581,469,468,584,585,586,587,588,589,465,782,783,780,781,786,787,784,785,788,789,737,736,735,734,733,732,731,730,739,738,81,918,247,603,602,601,600,607,606,605,604,609,608,543,618,619,1011,1010,1012,904,905,906,907,900,901,902,903,908,909,758,880,760,761,647,646,645,644,643,642,641,640,648,885,884,887,886,768,769,883,882,764,765,766,767,889,888,762,763,508,500,501,505,948,949,940,941,942,943,944,945,946,947,474,470,472,473,579,578,575,574,577,576,571,570,573,572,939,938,935,934,937,936,931,930,933,932,720,721,689,688,724,725,726,727,683,682,681,680,687,686,685,684,157,93,254,256,755,754,757,756,751,750,753,752,614,615,616,617,610,611,612,613,971,879,531,533,532,1008,1009,1006,1007,1004,1005,1002,1003,1000,1001,878,970,973,972,974,977,976,979,978,876,877,809,805,804,807,806,801,800,803,802,193,771,650,652,679,654,655,656,657,658,659,896,897,894,895,892,893,890,891,711,710,713,712,715,714,898,899,108,107,268,778,625,624,627,626,621,620,623,622,629,628,568,569,566,567,564,565,562,563,779,928,929,926,927,924,925,922,923,920,921,498,491,694,695,697,690,691,692,693,698,699,140,992,991,990,997,996,995,994,999,998,383,746,747,744,745,742,743,740,741,661,660,663,662,665,664,667,666,522,962,963,960,961,966,967,964,965,968,969,597,596,595,594,593,592,591,590,599,598,993,791,790,793,792,795,794,797,796,799,798,816,817,814,815,812,813,810,811,70,71,708,709,703,700,701,706,707,704,705,669,668,636,637,634,635,632,633,630,631,638,639,653]
subjects=subjects[:2]
#total_data = [] 

network=n.VGGFaceMEB(1, gpu="/gpu:"+str(GPU))
network.load_all_weights(WEIGHTS_PATH)

def sample_img_names(directory, sample_size):
    all_imgs = os.listdir(directory)
    n_total = len(all_imgs)
    step = int(n_total/sample_size)

    if step <= 1:
        return all_imgs
    
    final = []
    for i in xrange(0, n_total, step):
        final.append(all_imgs[i])

    return final

def do_mebs_for_subject(subject_id, img_code):
    # get the paths to the subject's folder
    fa_path = os.path.join(path_base, img_code, str(subject_id).zfill(5))
    fa_img_names = sample_img_names(fa_path, sample_n)
    for img_name in fa_img_names:
        # get the full path to the image
        fa_img_path = os.path.join(fa_path, img_name)
        favec = network.get_raw_output_for([p.load_adjust_avg(fa_img_path),])
        tup = (subject_id,img_code,img_name,favec,"")
        print ','.join(map(str,tup))
#        total_data.append(tup)

with open(os.path.realpath('./datasplits/subjtomeb_colorferet.json'),'r') as f:
        stom = json.load(f)

print 'subject_id,img_code,img_filename,raw_meb,raw_vgg'
for subject in subjects:
    true_meb=stom[str(subject).zfill(5)]
    tup=(subject, "true", "true", true_meb, "")
#    total_data.append(tup)
    print ','.join(map(str,tup))

    for code in ["fa","fb","rc"]:
        do_mebs_for_subject(subject,code)

#    with open('data_howfar_meb_subject_'+subject+'.csv', 'w') as f:
#        f.write()
#        for row in total_data:
#            f.write(','.join(map(str,row))+'\n')
