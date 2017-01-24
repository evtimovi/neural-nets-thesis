EVAL_SET_BASE = './datasets/feret-meb-vars/fb'
TRAIN_SET_BASE = './datasets/feret-meb-vars/fa'
PATH_STOM = './datasplits/subjtomeb_colorferet.json'
PATH_FTOS =  './datasplits/fa_filepath_to_subject_colorferet.json'
VGG_WEIGHTS_PATH = './vggface/weights/plain-vgg-trained.ckpt'
EVAL_WEIGHTS_PATH = './output/rc_subjects/weights/'
#FILES = #['weights_epoch_19_final.ckpt','weights_epoch_18_final.ckpt','weights_epoch_17_final.ckpt','weights_epoch_16_final.ckpt',
FILES = ['weights_epoch_15_final.ckpt','weights_epoch_14_final.ckpt',
        'weights_epoch_13_final.ckpt','weights_epoch_12_final.ckpt','weights_epoch_11_final.ckpt','weights_epoch_10_final.ckpt',
        'weights_epoch_9_final.ckpt','weights_epoch_7_final.ckpt','weights_epoch_5_final.ckpt']
SAVE_FOLDER = "./output/rc_subjects"
EVAL_SAMPLE_SIZE = 196  # number of images per subject used to evaluate the network 
NUM_EVAL_ITERS = 2 #how many times should the accuracy measures be computed with randomized imposters
EPOCHS = 20 
BATCH_SIZE = 2*49
LEARNING_RATE = 0.02
CHECKPOINT = 8
#SUBJ_FOR_TRAINING = sorted(os.listdir(TRAIN_SET_BASE))[:500]
SUBJ_FOR_TRAINING=list(set(['00070', '00070', '00070', '00071', '00081', '00081', '00093', '00093', '00107', '00107', '00108', '00108', '00140', '00146', '00146', '00146', '00146', '00146', '00157', '00193', '00247', '00254', '00256', '00256', '00268', '00268', '00383', '00383', '00465', '00468', '00468', '00468', '00469', '00469', '00469', '00470', '00472', '00473', '00474', '00491', '00498', '00500', '00501', '00505', '00508', '00519', '00522', '00531', '00532', '00533', '00543', '00562', '00562', '00563', '00564', '00564', '00565', '00565', '00566', '00566', '00567', '00568', '00568', '00569', '00570', '00571', '00572', '00573', '00574', '00575', '00576', '00576', '00577', '00578', '00579', '00579', '00580', '00581', '00582', '00583', '00584', '00584', '00585', '00586', '00587', '00588', '00588', '00588', '00589', '00590', '00591', '00592', '00593', '00593', '00594', '00594', '00594', '00595', '00596', '00596', '00597', '00597', '00598', '00598', '00599', '00600', '00600', '00601', '00602', '00603', '00604', '00604', '00604', '00605', '00605', '00606', '00607', '00608', '00609', '00610', '00611', '00612', '00613', '00614', '00614', '00615', '00615', '00616', '00616', '00617', '00618', '00618', '00619', '00620', '00621', '00621', '00622', '00622', '00623', '00624', '00625', '00626', '00626', '00627', '00627', '00628', '00629', '00630', '00630', '00630', '00631', '00632', '00633', '00634', '00635', '00636', '00637', '00638', '00638', '00639', '00640', '00641', '00642', '00642', '00643', '00644', '00645', '00646', '00647', '00648', '00648', '00650', '00652', '00653', '00654', '00655', '00656', '00657', '00658', '00659', '00660', '00660', '00660', '00660', '00661', '00662', '00663', '00664', '00665', '00666', '00667', '00668', '00669', '00670', '00671', '00672', '00673', '00674', '00675', '00676', '00677', '00678', '00679', '00680', '00681', '00682', '00683', '00684', '00685', '00686', '00687', '00688', '00689', '00690', '00691', '00692', '00693', '00694', '00695', '00697', '00698', '00699', '00700', '00701', '00703', '00703', '00704', '00704', '00704', '00705', '00705', '00705', '00705', '00706', '00706', '00706', '00706', '00707', '00707', '00708', '00708', '00708', '00708', '00709', '00709', '00710', '00710', '00710', '00711', '00711', '00711', '00711', '00712', '00712', '00713', '00713', '00713', '00714', '00714', '00714', '00715', '00715', '00716', '00716', '00717', '00717', '00717', '00717', '00718', '00718', '00719', '00719', '00720', '00720', '00720', '00721', '00721', '00722', '00722', '00722', '00722', '00723', '00724', '00724', '00724', '00725', '00725', '00726', '00726', '00727', '00727', '00728', '00728', '00729', '00729', '00730', '00730', '00730', '00731', '00732', '00732', '00732', '00732', '00732', '00732', '00733', '00733', '00734', '00735', '00735', '00736', '00736', '00737', '00737', '00737', '00738', '00739', '00739', '00740', '00740', '00741', '00742', '00743', '00743', '00743', '00744', '00744', '00744', '00744', '00745', '00745', '00745', '00745', '00746', '00746', '00747', '00747', '00749', '00750', '00750', '00750', '00751', '00751', '00751', '00751', '00752', '00752', '00752', '00752', '00753', '00754', '00755', '00756', '00757', '00758', '00758', '00760', '00760', '00760', '00761', '00761', '00761', '00761', '00762', '00762', '00763', '00763', '00763', '00764', '00764', '00765', '00766', '00766', '00766', '00767', '00767', '00768', '00768', '00768', '00769', '00769', '00770', '00770', '00770', '00770', '00771', '00771', '00771', '00772', '00772', '00772', '00773', '00773', '00773', '00774', '00774', '00774', '00775', '00775', '00776', '00777', '00778', '00779', '00779', '00780', '00780', '00781', '00781', '00782', '00783', '00784', '00785', '00786', '00787', '00787', '00788', '00789', '00789', '00789', '00790', '00791', '00792', '00793', '00794', '00794', '00794', '00795', '00796', '00797', '00797', '00798', '00799', '00800', '00801', '00802', '00803', '00804', '00804', '00805', '00806', '00806', '00807', '00807', '00809', '00809', '00809', '00810', '00810', '00811', '00811', '00812', '00812', '00813', '00814', '00815', '00816', '00816', '00816', '00817', '00876', '00876', '00877', '00878', '00879', '00879', '00880', '00882', '00883', '00884', '00885', '00886', '00887', '00888', '00889', '00890', '00891', '00892', '00893', '00894', '00894', '00895', '00896', '00897', '00898', '00899', '00899', '00900', '00900', '00901', '00902', '00903', '00903', '00904', '00905', '00906', '00907', '00907', '00908', '00909', '00910', '00911', '00912', '00913', '00914', '00915', '00916', '00917', '00918', '00919', '00920', '00921', '00922', '00923', '00924', '00925', '00926', '00927', '00928', '00929', '00930', '00931', '00932', '00933', '00934', '00935', '00936', '00937', '00938', '00939', '00940', '00941', '00942', '00943', '00944', '00945', '00946', '00947', '00948', '00949', '00950', '00951', '00952', '00953', '00954', '00955', '00956', '00957', '00958', '00959', '00960', '00961', '00962', '00963', '00964', '00965', '00966', '00967', '00968', '00969', '00970', '00971', '00972', '00973', '00974', '00976', '00977', '00978', '00979', '00980', '00981', '00982', '00983', '00984', '00985', '00986', '00987', '00988', '00989', '00990', '00991', '00992', '00993', '00994', '00995', '00996', '00997', '00998', '00999', '01000', '01001', '01002', '01003', '01004', '01005', '01006', '01007', '01008', '01009', '01010', '01011', '01012']))
