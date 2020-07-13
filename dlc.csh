#!/bin/csh -f
source /tools/dotfile_new/cshrc.lsf


## Pls change this section 
set shuttle_name = "0712" ## must align shuttle_name used in run.csh                                                                                                                                   
set q_name = "testchip_1.q"
##

set plot_dir = "./output/figure_$shuttle_name"

if (! -d src) then
    cp -rf /FSIM/SOIC_3DICTV_BE_20211231_1/users/uchuangn/DLC/src ./
endif
/FSIM/SOIC_3DICTV_BE_20211231_1/users/sylinzzq/anaconda36/bin/python3 /FSIM/SOIC_3DICTV_BE_20211231_1/users/uchuangn/DLC/dlc.py $plot_dir $shuttle_name
if (! -d dlc_raw_data) then
    mkdir -p dlc_raw_data
    #mv dlc_${shuttle_name}_p128/*.npy ./dlc_raw_data/
else
    rm -rf dlc_raw_data
    mkdir -p dlc_raw_data
endif
mv dlc_${shuttle_name}_p128/*.npy ./dlc_raw_data/
mv dlc_${shuttle_name}_p128/calibre_report ./dlc_raw_data
