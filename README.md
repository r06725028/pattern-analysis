# pattern-analysis

src/cluster.py 第348行def generate_dlc_report(save_dir, group_list, num_list, pv_df, total_outlier, clu_alg=cf.clu_alg, clu=cf.CLU+1, rule_num=cf.rule_num) 

src/dlc_template.html  最新版的模板

各個參數替代值
save_dir = 'dlc_0712_p128' -->  算L1_num和L2_num和這個參數有關，所以354~360要註解掉 
  L1_num = L2_num = 100

group_list = [('M/S/low', (10, 0.1)), ('M/S/mid', (5, 0.2)), ('M/S/top', (3, 0.5))]

num_list = [('DSP', 11), ('enc', 4), ('maia', 2)]

pv_df 是dataframe
name是index,其他是欄位 : name, count, tot_fr, out_pent, out_fr, in_pent, in_fr, sugg_rcp, rank
分別值是 VIA1.S.1, 303, 0.0,	19.14,	0.0,	80.86,	0.0, EDA, 502.071

total_outlier = 10

clu_alg和clu是為了抓圖，你可以用其他圖代替就不用這兩個參數了

rule_num = 3
