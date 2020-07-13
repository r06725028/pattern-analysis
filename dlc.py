import sys, os                                                                                                                                                                                         
from src.process import *
from src.cluster import *


inpath = sys.argv[1]
shuttle = sys.argv[2]
usage = 'dlc'

if usage == 'dlc':
    save_dir = process(inpath, shuttle, usage=usage, use_shuttle='0505', use_model='augencoder_32.h5')
    outlier_dict, num_list, db_split_dict, pv_df, total_outlier = cluster_group(save_dir)
    generate_dlc_report(save_dir, outlier_dict, num_list, pv_df, total_outlier)
    generate_calibre_db(save_dir, db_split_dict)
    
elif usage == 'cbca':
    save_dir = process(inpath, shuttle, usage=usage, use_shuttle='0505', use_model='augencoder_32.h5')
    iso, pvtorcp_dict = train_isotree(use_dlc='0519', old_cbca=['0525', '0526', '0619'])
    group_list, case_dict, num_list, db_split_dict, pv_df, total_outlier = test_cacb(save_dir, iso, pvtorcp_dict)
    generate_cbca_report(save_dir, case_dict, group_list, num_list, pv_df, total_outlier)
    generate_calibre_db(save_dir, db_split_dict)
else:
    print('ERROR : only dlc, cbca')
