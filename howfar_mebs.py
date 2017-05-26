from util import processimgs as p
from scipy.spatial import distance
from vggface import networks as n
import os
import random
import sys
import json

# ensure that no elements of the two arrays are at the same index
def index_match(arr1, arr2):
    for i in xrange(len(arr1)):
        if arr1[i] == arr2[i]:
            return True
    return False

path_base = "./datasets/feret-meb-vars/"
sample_n = 99
#subjects=['00070','00140','00468','00682','00636']
#imposters={'00070': '00071', '00140': '00146', '00468': '00960', '00682': '00093',  '00636': '00591'}
imposters={70: "936" ,71: "688" ,81: "580" ,93: "254" ,107: "645" ,108: "  650" ,140: "993" ,146: "974" ,157: "639" ,193: "986" ,247: "  943" ,254: "900" ,256: "1010" ,268: "888" ,383: "522" ,465: "  565" ,468: "799" ,469: "695" ,470: "655" ,472: "891" ,473: "  633" ,474: "735" ,491: "977" ,498: "606" ,500: "878" ,501: "  783" ,505: "753" ,508: "648" ,519: "974" ,522: "573" ,531: "  889" ,532: "913" ,533: "690" ,543: "740" ,562: "562" ,563: "  813" ,564: "770" ,565: "769" ,566: "919" ,567: "771" ,568: "  766" ,569: "1003" ,570: "963" ,571: "681" ,572: "642" ,573: "  691" ,574: "633" ,575: "664" ,576: "681" ,577: "994" ,578: "  612" ,579: "568" ,580: "601" ,581: "636" ,582: "522" ,583: "  724" ,584: "570" ,585: "896" ,586: "505" ,587: "753" ,588: "  610" ,589: "532" ,590: "955" ,591: "981" ,592: "682" ,593: "  929" ,594: "519" ,595: "491" ,596: "798" ,597: "966" ,598: "  543" ,599: "912" ,600: "640" ,601: "922" ,602: "950" ,603: "  905" ,604: "472" ,605: "571" ,606: "774" ,607: "916" ,608: "  694" ,609: "93" ,610: "632" ,611: "566" ,612: "707" ,613: "  999" ,614: "697" ,615: "771" ,616: "674" ,617: "623" ,618: "  473" ,619: "952" ,620: "780" ,621: "904" ,622: "953" ,623: "  588" ,624: "596" ,625: "500" ,626: "880" ,627: "906" ,628: "  985" ,629: "877" ,630: "617" ,631: "498" ,632: "638" ,633: "  742" ,634: "977" ,635: "1008" ,636: "815" ,637: "637" ,638: "  613" ,639: "876" ,640: "731" ,641: "498" ,642: "934" ,643: "  93" ,644: "703" ,645: "971" ,646: "522" ,647: "745" ,648: "  578" ,650: "610" ,652: "671" ,653: "606" ,654: "967" ,655: "  815" ,656: "885" ,657: "639" ,658: "886" ,659: "247" ,660: "  595" ,661: "813" ,662: "930" ,663: "608" ,664: "762" ,665: "  949" ,666: "792" ,667: "776" ,668: "567" ,669: "612" ,670: "  967" ,671: "644" ,672: "691" ,673: "987" ,674: "942" ,675: "  908" ,676: "983" ,677: "774" ,678: "729" ,679: "743" ,680: "  1001" ,681: "787" ,682: "923" ,683: "934" ,684: "652" ,685: "  679" ,686: "882" ,687: "929" ,688: "1005" ,689: "645" ,690: "  1002" ,691: "981" ,692: "566" ,693: "698" ,694: "581" ,695: "  722" ,697: "924" ,698: "926" ,699: "923" ,700: "779" ,701: "  890" ,703: "684" ,704: "501" ,705: "923" ,706: "930" ,707: "  71" ,708: "594" ,709: "979" ,710: "771" ,711: "589" ,712: "  986" ,713: "959" ,714: "782" ,715: "772" ,716: "591" ,717: "  877" ,718: "595" ,719: "531" ,720: "742" ,721: "710" ,722: "  611" ,723: "470" ,724: "599" ,725: "593" ,726: "997" ,727: "  674" ,728: "650" ,729: "383" ,730: "268" ,731: "996" ,732: "  724" ,733: "752" ,734: "799" ,735: "988" ,736: "619" ,737: "  704" ,738: "733" ,739: "619" ,740: "569" ,741: "925" ,742: "  937" ,743: "667" ,744: "583" ,745: "732" ,746: "731" ,747: "  734" ,749: "591" ,750: "742" ,751: "720" ,752: "921" ,753: "  625" ,754: "799" ,755: "946" ,756: "716" ,757: "778" ,758: "  565" ,760: "627" ,761: "964" ,762: "804" ,763: "690" ,764: "  713" ,765: "660" ,766: "976" ,767: "730" ,768: "994" ,769: "  1006" ,770: "716" ,771: "765" ,772: "883" ,773: "718" ,774: "  777" ,775: "898" ,776: "900" ,777: "659" ,778: "783" ,779: "  879" ,780: "594" ,781: "755" ,782: "981" ,783: "780" ,784: "  578" ,785: "565" ,786: "966" ,787: "883" ,788: "684" ,789: "  574" ,790: "1006" ,791: "568" ,792: "890" ,793: "688" ,794: "  1000" ,795: "970" ,796: "664" ,797: "629" ,798: "611" ,799: "  889" ,800: "932" ,801: "611" ,802: "681" ,803: "608" ,804: "  607" ,805: "876" ,806: "744" ,807: "569" ,809: "508" ,810: "  756" ,811: "917" ,812: "998" ,813: "642" ,814: "976" ,815: "  816" ,816: "519" ,817: "628" ,876: "910" ,877: "989" ,878: "  655" ,879: "594" ,880: "985" ,882: "925" ,883: "929" ,884: "  717" ,885: "905" ,886: "146" ,887: "586" ,888: "925" ,889: "  729" ,890: "721" ,891: "609" ,892: "668" ,893: "590" ,894: "  577" ,895: "978" ,896: "997" ,897: "946" ,898: "721" ,899: "  723" ,900: "573" ,901: "765" ,902: "889" ,903: "910" ,904: "  569" ,905: "692" ,906: "700" ,907: "617" ,908: "907" ,909: "  636" ,910: "669" ,911: "784" ,912: "751" ,913: "469" ,914: "  656" ,915: "574" ,916: "140" ,917: "972" ,918: "730" ,919: "  590" ,920: "543" ,921: "628" ,922: "599" ,923: "724" ,924: "  911" ,925: "737" ,926: "941" ,927: "653" ,928: "533" ,929: "  801" ,930: "953" ,931: "608" ,932: "765" ,933: "724" ,934: "  588" ,935: "998" ,936: "473" ,937: "648" ,938: "677" ,939: "  70" ,940: "789" ,941: "1012" ,942: "784" ,943: "698" ,944: "  745" ,945: "666" ,946: "672" ,947: "727" ,948: "543" ,949: "  1003" ,950: "634" ,951: "727" ,952: "707" ,953: "611" ,954: "  573" ,955: "594" ,956: "634" ,957: "695" ,958: "659" ,959: "  626" ,960: "604" ,961: "724" ,962: "627" ,963: "714" ,964: "  644" ,965: "909" ,966: "708" ,967: "705" ,968: "790" ,969: "  659" ,970: "682" ,971: "804" ,972: "996" ,973: "924" ,974: "  608" ,976: "157" ,977: "790" ,978: "676" ,979: "809" ,980: "  533" ,981: "672" ,982: "491" ,983: "927" ,984: "925" ,985: "  633" ,986: "891" ,987: "752" ,988: "653" ,989: "628" ,990: "  790" ,991: "660" ,992: "908" ,993: "762" ,994: "750" ,995: "  645" ,996: "745" ,997: "911" ,998: "770" ,999: "995" ,1000: "  715" ,1001: "916" ,1002: "738" ,1003: "989" ,1004: "747" ,1005: "  592" ,1006: "470" ,1007: "891" ,1008: "628" ,1009: "963" ,1010: "  685" ,1011: "629" ,1012: "247"}
subjects = imposters.keys()

total_data = [] # map from subject to tuple (subjectid, fbdist, rcdist, imposterid, fbdist_imposter, rcdist_imposter)

network=n.VGGFaceMEB(1, gpu="/gpu:2")
network.load_all_weights("./output/rc_subjects/weights/weights_epoch_11_final.ckpt")

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

def calculate_mebs_for_subject(subject_id, img_code):
    # get the paths to the subject's folder
    fa_path = os.path.join(path_base, img_code, str(subject_id).zfill(5))

    fa_img_names = sample_img_names(fa_path, sample_n)

    mebs = []
    for img_name in fa_img_names:
        # get the full path to the image
        fa_img_path = os.path.join(fa_path, img_name)
        favec = network.get_raw_output_for([p.load_adjust_avg(fa_img_path),])
        mebs.append(favec)
    return mebs 

with open(os.path.realpath('./datasplits/subjtomeb_colorferet.json'),'r') as f:
        stom = json.load(f)

for subject in subjects:
    true_meb=stom[str(subject).zfill(5)]
    imposter = int(imposters[subject].strip())

    mebs_fa=calculate_mebs_for_subject(subject,"fa")
    mebs_fb=calculate_mebs_for_subject(subject,"fb")
    mebs_rc=calculate_mebs_for_subject(subject,"rc")

    mebs_fa_imposter=calculate_mebs_for_subject(imposter,"fa")
    mebs_fb_imposter=calculate_mebs_for_subject(imposter,"fb")
    mebs_rc_imposter=calculate_mebs_for_subject(imposter,"rc")

    for i in xrange(sample_n):
        euclidean_fa = distance.euclidean(true_meb, mebs_fa[i])
        euclidean_fb = distance.euclidean(true_meb, mebs_fb[i])
        euclidean_rc = distance.euclidean(true_meb, mebs_rc[i])
        hamming_fa = distance.hamming(true_meb, map(lambda x: 1 if x > 0.5 else 0, mebs_fa[i]))
        hamming_fb = distance.hamming(true_meb, map(lambda x: 1 if x > 0.5 else 0, mebs_fb[i]))
        hamming_rc = distance.hamming(true_meb, map(lambda x: 1 if x > 0.5 else 0, mebs_rc[i]))
        
        euclidean_fa_imposter = distance.euclidean(true_meb, mebs_fa_imposter[i])
        euclidean_fb_imposter = distance.euclidean(true_meb, mebs_fb_imposter[i])
        euclidean_rc_imposter = distance.euclidean(true_meb, mebs_rc_imposter[i])
        hamming_fa_imposter = distance.hamming(true_meb, map(lambda x: 1 if x > 0.5 else 0, mebs_fa_imposter[i]))
        hamming_fb_imposter = distance.hamming(true_meb, map(lambda x: 1 if x > 0.5 else 0, mebs_fb_imposter[i]))
        hamming_rc_imposter = distance.hamming(true_meb, map(lambda x: 1 if x > 0.5 else 0, mebs_rc_imposter[i]))


        total_data.append((subject,imposter,euclidean_fa, euclidean_fb, euclidean_rc,hamming_fa,hamming_fb,hamming_rc,euclidean_fa_imposter, euclidean_fb_imposter, euclidean_rc_imposter,hamming_fa_imposter,hamming_fb_imposter,hamming_rc_imposter))

    with open('data_howfar_meb_all_subjects.csv', 'w') as f:
        f.write('subject,imposter,euclidean_fa,euclidean_fb,euclidean_rc,hamming_fa,hamming_fb,hamming_rc,euclidean_fa_imposter,euclidean_fb_imposter,euclidean_rc_imposter,hamming_fa_imposter,hamming_fb_imposter,hamming_rc_imposter\n')
        for row in total_data:
            f.write(','.join(map(str,row))+'\n')
