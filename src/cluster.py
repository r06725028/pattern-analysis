from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import math
plt.switch_backend('agg')
import sys, re, os
from random import sample

sys.path.append('/FSIM/SOIC_3DICTV_BE_20211231_1/users/uchuangn/DLC/')
import src.config as cf
from src.process import *

from time import time
import webbrowser

def extract_rcp_fr(address, if_convert=True):
    rule_name =  address.split('/')[-2]
    
    if 'env' in address:
        checkrcpfr = re.compile('^[A-Z]+\d+_(.*)_fr1um_(.*)_frclu_(.*)_ori_(.+)_(.+)_(.+)_(.+)_remark_(.+)_env_(.+)_X.+')
        rcp_fr = re.match(checkrcpfr, address.split('/')[-1])
        [rcp, fr1um, frclu, llx, lly, urx, ury, remark, env] = [rcp_fr.group(i+1) for i in range(9)]
    elif 'remark' in address:
        checkrcpfr = re.compile('^[A-Z]+\d+_(.*)_fr1um_(.*)_frclu_(.*)_ori_(.+)_(.+)_(.+)_(.+)_remark_(.+)_X.+')
        rcp_fr = re.match(checkrcpfr, address.split('/')[-1])
        [rcp, fr1um, frclu, llx, lly, urx, ury, remark] = [rcp_fr.group(i+1) for i in range(8)]
        env = ''
    elif 'ori' in address:
        checkrcpfr = re.compile('^[A-Z]+\d+_(.*)_fr1um_(.*)_frclu_(.*)_ori_(.+)_(.+)_(.+)_(.+)_X.+')
        rcp_fr = re.match(checkrcpfr, address.split('/')[-1])
        [rcp, fr1um, frclu, llx, lly, urx, ury] = [rcp_fr.group(i+1) for i in range(7)]
        remark, env = '', ''
    else:
        checkrcpfr = re.compile('^[A-Z]+\d+_(.*)_fr1um_(.*)_frclu_(.*)_X.+')
        rcp_fr = re.match(checkrcpfr, address.split('/')[-1])
        [rcp, fr1um, frclu] = [rcp_fr.group(i+1) for i in range(3)]
        llx = lly = urx = ury = '0'
        remark, env = '', ''

    rcp = rcp if rcp != '' else 'none'
    fr1um = fr1um if fr1um not in ['None', ''] else 0.0
    frclu = frclu if frclu not in ['None', ''] else 0.0
           
    if if_convert:
        rcp = convert_rcp(rcp, fr1um)
    
    if rcp not in cf.label_list:
        print('new recipe !!!', rcp)
        print(address)
    return [rcp, float(fr1um), float(frclu), rule_name, float(llx), float(lly), float(urx), float(ury), remark, env]

def convert_rc(ct_rcp_frrcp, fr1um):
    if rcp == 'none':
        if float(fr1um) >= cf.hfr:
            rcp = 'EDA'

    return rcp

def read_feature(save_dir, tp, stp, lyr):
    fea = np.load(os.path.join(save_dir, tp, stp, lyr, cf.fea_name))
    f = np.load(os.path.join(save_dir, tp, stp, lyr, cf.f_name))
    ftofea = {address:feature for address, feature in zip(f, fea)}

    return ftofea, fea, f

def out2table(all_outlier, pvtorcp_dict, pvtofr_dict):
    db_split_dict = {}
    num_dict = {}

    ids = list(pvtofr_dict.keys())
    pv_outfr_dict = {}
    
    for f in all_outlier:
        split = f.split('/')[-4]
        [fr, _, rule] = extract_rcp_fr(f, True)[1:4]
        
        if rule in pv_outfr_dict:
            pv_outfr_dict[rule].append(fr)
        else:
            pv_outfr_dict[rule] = [fr]
        
        if split in db_split_dict:
            db_split_dict[split].append(f)
        else:
            db_split_dict[split] = [f]
        
    for split in db_split_dict:
        num_dict[split] = len(db_split_dict[split])

    frs = []
    for rule in ids:
        frs += pvtofr_dict[rule]
    shuttle_fr = np.mean(frs)
    fr_th = shuttle_fr + abs(shuttle_fr)*0.1
    
    rows = []
    for rule in ids:
        drc_num = len(pvtofr_dict[rule])
        tot_fr = round(np.mean(pvtofr_dict[rule]), 3)

        if rule in pv_outfr_dict:
            out_fr = round(np.mean(pv_outfr_dict[rule]), 3)
            if len(pvtofr_dict[rule])-len(pv_outfr_dict[rule]) > 0:
                in_fr = round((np.sum(pvtofr_dict[rule]) - np.sum(pv_outfr_dict[rule]))/(len(pvtofr_dict[rule])-len(pv_outfr_dict[rule])), 3)
            else:
                in_fr = 1.0
            out_pent = round((len(pv_outfr_dict[rule]) / len(pvtofr_dict[rule]))*100, 2)
        else:
            out_fr = 1.0
            in_fr = tot_fr
            out_pent = 0.00
        
        in_pent = round(100 - out_pent, 2)

        sugg_rcp = ''
        if rule in pvtorcp_dict:
            rcp_cand = []
            for rcp, fr in pvtorcp_dict[rule]:
                if ((fr >= fr_th) or (fr >= cf.hfr)):
                    rcp_cand.append(rcp)
            if len(rcp_cand) > 0:
                sugg_rcp = max(rcp_cand, key=rcp_cand.count)
                if 'none' in sugg_rcp:
                    sugg_rcp = ''

        rank = round(math.log(drc_num)*((1-out_fr)*out_pent+0.85*(1-in_fr)*in_pent), 3)
        rows.append([drc_num, tot_fr, out_pent, out_fr, in_pent, in_fr, sugg_rcp, rank])

    df = pd.DataFrame(rows, index=ids, columns=['#DRC', 'tot_FR', '%outlier', 'out_FR', '%inlier', 'in_FR', 'recipe suggestion', 'rank'])
    df.sort_values(by=['rank'], ascending=[False], inplace=True)

    return sorted(num_dict.items(), key=lambda d: d[1], reverse=True), db_split_dict, df


def generate_sub_fig(rcp_dict, title, fr_dict):
    keys = list(rcp_dict.keys())
    size = []
    for k in keys:
        size.append(rcp_dict[k])
    
    if title != '':
        plt.title(title, fontsize=12, pad=18)
    
    ratio = list(map(lambda x: round(x*100/sum(size)), size))
    p, tx, autotexts = plt.pie(size, labels=keys, autopct="", startangle=90, textprops={'fontsize':6})
    for i in range(len(tx)):
        if size[i] > round(sum(size)*0.01):
            autotexts[i].set_text("{}".format(size[i]))
            tx[i].set_fontsize(7)
            
            if keys[i] in ['none', 'Ignored', 'EDA', 'Customized']:
                tx[i].set_text("{}({}%, {:.3f})\n\n".format(keys[i], ratio[i], sum(fr_dict[keys[i]])/len(fr_dict[keys[i]])))
            elif keys[i] in ['ReRoute', 'Add_RBLK']:
                tx[i].set_text("\n\n{}({}%)\n({:.3f})\n".format(keys[i], ratio[i], sum(fr_dict[keys[i]])/len(fr_dict[keys[i]])))
            else:
                tx[i].set_text("\n\n{}\n({}%, {:.3f})\n".format(keys[i], ratio[i], sum(fr_dict[keys[i]])/len(fr_dict[keys[i]])))
        else:
            autotexts[i].set_text("")
            tx[i].set_text("")

    plt.axis('equal')

def plot_pie_chart(group_dir, clu_alg, clu, tp, stp, lyr, score, recipe, rule_name, totclu, totdrc, fr_rcp, fr_rule, demo):
    pie_path = os.path.join(group_dir, cf.pie_name.format(clu_alg, clu))
    
    fig = plt.figure(figsize=(16,1.8+clu*1.5), dpi=80)                                                                   
    plt.subplots_adjust(left=0.045, bottom=0.03, right=0.96, top=0.92, wspace=0.45, hspace=0.5)               
    plt.suptitle('Recipe/RuleName of {}/{}/{} DRC ({}, #DRCs = {})'.format(tp, stp, lyr, score[0], totdrc), fontsize=16)
    
    st = 1 if sum(list(recipe[0].values())) == 0 else 0
    for i in range(st, clu):
        if sum(list(recipe[i].values())) >0:
            ax = plt.subplot2grid((clu-st, cf.demo_num+2),(i-st,0))
            generate_sub_fig(recipe[i], "Cluster {} ({}, {})".format(i, score[i+1], totclu[i]), fr_rcp[i])
        
            ax = plt.subplot2grid((clu-st,cf.demo_num+2),(i-st,1))
            generate_sub_fig(rule_name[i], '', fr_rule[i])

            for j in range(len(demo[i])):
                ax = plt.subplot2grid((clu-st,cf.demo_num+2),(i-st,j+2))
                lout = plt.imread(demo[i][j])
                ax.axis('off')
                ax.spines['left'].set_visible(False)
                ax.imshow(lout,interpolation='none', origin='upper')

    fig = plt.gcf()
    fig.savefig(pie_path, dpi=100)
    #plt.show()
    plt.close()

def plot_tsne(group_dir, clu_alg, clu, encoder_feature, labels):
    tsne_path = os.path.join(group_dir, 'only_L1_c.png')#cf.tsne_name.format(clu_alg, clu))    
    X_embedded = TSNE(n_components=2).fit_transform(encoder_feature)
    plt.figure()
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c=labels, s=5, alpha=2, cmap=plt.cm.Spectral)
    plt.savefig(tsne_path)
    #plt.show()
    plt.close()

def plot_outlier_demo(save_dir, all_outlier, total_outlier, NUM=(cf.demo_num+2), row=2):
    if total_outlier >= NUM*row:
        demo = [all_outlier[j] for j in sample(range(0, total_outlier), NUM*row)]
    else:
        demo = all_outlier
        
    out_path = os.path.join(save_dir, cf.out_name)
    fig = plt.figure(figsize=(8, 1.5*row), dpi=80)
    plt.subplots_adjust(left=0.045, bottom=0.03, right=0.96, top=0.96, wspace=0.25, hspace=0.1)               
    
    for i in range(row):
        for j in range(NUM):
            idx = i*NUM+j
            if len(demo) > idx:
                ax = plt.subplot2grid((row, NUM),(i,j))
                rule_name = extract_rcp_fr(demo[idx])[3]
                plt.title(rule_name, fontsize=9, pad=3)
                lout = plt.imread(demo[idx])
                ax.axis('off')
                ax.spines['left'].set_visible(False)
                ax.imshow(lout,interpolation='none', origin='upper')

    fig = plt.gcf()
    fig.savefig(out_path, dpi=100)
    #plt.show()
    plt.close()

def generate_res(group_dir, clu_alg, clu, encoder_feature, fname, out_fea, out_f):                                     
    print(encoder_feature.shape)
    if clu_alg == 'km':
        labels = KMeans(n_clusters=clu, n_init=20, random_state=42).fit(encoder_feature).labels_
    elif clu_alg == 'spect':    
        labels = SpectralClustering(clu, affinity='rbf', assign_labels='discretize',n_neighbors=5, random_state=42, n_jobs=-1).fit_predict(encoder_feature)
    
    #w/o outlier
    s = silhouette_score(encoder_feature, labels)

    clu +=1
    clu_s = []

    if len(out_f) > 0:
    #w/ outlier
        encoder_feature = np.concatenate((out_fea, encoder_feature), axis=0)
        fname = np.concatenate((out_f, fname), axis=0)
        labels = np.concatenate((np.array([-1]*len(out_f)), labels), axis=0)+1
        st = 0
    else:
        clu_s = [0.0]
        labels += 1
        st = 1

    sample_silhouette_values = silhouette_samples(encoder_feature, labels)
    for i in range(st, clu):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        clu_s.append(np.mean(ith_cluster_silhouette_values))          

    res_path = os.path.join(group_dir, cf.res_name.format(clu_alg, clu))
    with open(res_path, 'w') as f:
        f.write("The mean of silhouette : {:.4f}\n".format(s))
        for i in range(clu):
            f.write("The {:2>d} silhouette_score is : {:.4f}\n".format(i, clu_s[i]))
        
        for (lab, fn) in sorted(zip(labels, fname)):
            f.write('{:2>d} : {}\n'.format(int(lab), fn)) 
    
    return labels, clu


def outlier_detection(encoder_feature):
    if len(encoder_feature) > 0:
        iso = IsolationForest(n_estimators=100, bootstrap="new", contamination='auto', behaviour='new', random_state=42, n_jobs=-1)
        labels = iso.fit_predict(np.array(encoder_feature).reshape(len(encoder_feature), -1))
        n_noise = list(labels).count(-1)

        if n_noise > 0:
            score = round(np.mean(iso.decision_function(np.array(encoder_feature)[np.array(labels)==-1])), 4)
            pent = n_noise/len(labels)
        else:
            score = 0.0
            pent = 0.0
    else:
        labels = []
        score = 0.0
        n_noise = 0
        pent = 0.0

    print('# of noise = ', n_noise)
    print('# 0f L1 drc = ', len(labels))
    print('% of noise = ', pent)
    
    return labels, n_noise, score

def process_res(group_dir, CLU, NUM, clu_alg):
    res_path = os.path.join(group_dir, cf.res_name.format(clu_alg, CLU))
    recipe = []
    rule_name = []
    demo = []
    fr_rcp = []
    fr_rule = []

    for i in range(CLU):
        demo.append([])
        rcp_dict = {}
        fr_rcp_dict = {}
        rule_dict = {}
        fr_rule_dict = {}

        for lab in cf.label_list:
            rcp_dict[lab] = 0
            fr_rcp_dict[lab] = []

        recipe.append(rcp_dict)
        fr_rcp.append(fr_rcp_dict)
        rule_name.append(rule_dict)
        fr_rule.append(fr_rule_dict)

    with open(res_path, 'r') as f:
        content = f.readlines()
        score = []
        
        for line in content[:CLU+1]:
            score.append(line.strip().split(' : ')[1])
        content = content[CLU+1:]

        totdrc = len(content)
        # M4_none_fr1um_1.0_frclu_None_X_833.476_834.476_Y_818.552_819.552_VIA4.png', 'M4_none_fr1um_1.0_frclu_None_X_833.476_834.476_Y_818.552_819.552_M5.png
        for line in content:
            [no, address] = line.strip().split(' : ')
            [rcp, fr1um, frclum, rule] = extract_rcp_fr(address, True)[:4]
            demo[int(no)].append(address)
            

            recipe[int(no)][rcp] += 1

            if rule not in rule_name[int(no)]:
                rule_name[int(no)][rule] = 1
                fr_rule[int(no)][rule] = []
            else:
                rule_name[int(no)][rule] += 1

            if cf.fr == 'fr1um':
                fr_rcp[int(no)][rcp].append(fr1um)
                fr_rule[int(no)][rule].append(fr1um)
            else:
                fr_rcp[int(no)][rcp].append(float(frclu))
                fr_rule[int(no)][rule].append(float(frclu))
         
        totclu = []
        for i in range(CLU):
            totclu.append(len(demo[i]))
            if len(demo[i]) >= NUM:
                demo[i] = [demo[i][j] for j in sample(range(0,totclu[i]), NUM)]

    return score, recipe, rule_name, demo, totdrc, fr_rcp, fr_rule, totclu

def try_k(group_dir, clu_alg, encoder_feature):
    try_path = os.path.join(group_dir, cf.try_name.format(clu_alg))

    # try to find best num of cluster
    if not os.path.exists(try_path):
        with open(try_path, 'w') as f:                                              
            for clu in range(2,6):
                if clu_alg == 'km':
                    labels = KMeans(n_clusters=clu, n_init=20, random_state=42).fit(encoder_feature).labels_
                else:
                    labels = SpectralClustering(clu, affinity='nearest_neighbors', assign_labels='discretize', random_state=42).fit_predict(encoder_feature)
        
                s = silhouette_score(encoder_feature, labels)
                f.write('clu no. : {:2>d}, s : {:.4f}\n'.format(clu, s))

def generate_dlc_report(save_dir, group_list, num_list, pv_df, total_outlier, clu_alg=cf.clu_alg, clu=cf.CLU+1, rule_num=cf.rule_num):
    outpath = os.path.join(os.getcwd(), save_dir)

    data_path = os.path.join(save_dir, cf.data_name)
    outlier_path = os.path.join(cf.out_name)
    
    L1_num, L2_num = 0, 0
    all_data = np.load(data_path, allow_pickle=True).item()
    for address in all_data:
        if 'ADFL1' in address:
            L1_num += 1
        elif 'ADFL2' in address:
            L2_num += 1

    buffer_html = ""
    with open("./src/dlc_template.html", "r") as f:
      buffer_html += f.read()

    buffer_html = buffer_html.replace("%total_clips%", str(int((L1_num+L2_num)/5))+'(L1:{}, L2:{})'.format(int(L1_num/5), int(L2_num/5)))
    buffer_html = buffer_html.replace("%analysis_output_path%", outpath)
    buffer_html = buffer_html.replace("%outlier_count%", str(total_outlier))
    buffer_html = buffer_html.replace("%outlier_img_path%", outlier_path)

    pv_df_html = ""
    outlier_db_html = ""
    outlier_group_html = ""
    detailed_blocks_html = ""
    for tup in group_list:
        (name, (count, score)) = tup
        outlier_group_html += "<tr><td>{name}</td><td>{count}</td><td>{score}</td></tr>".format(name=name, count=count, score=score)

        [tp, stp,lyr] = name.split('/')
        img_path = os.path.join(tp, stp, lyr, cf.pie_name.format(clu_alg, clu))
        detailed_blocks_html += '''<div class="cluster-block">
         <section>
          <span>- {tp}/{stp}/{lyr}</span>
          <span class="toggle" toggle-target-id="{target_id}">(toggle)</span></li>
         </section>
         <img id="{target_id}" class="vanish" src="{img_path}">
        </div>'''.format(tp=tp, stp=stp, lyr=lyr, target_id=tp+stp+lyr, img_path=img_path)

    for db in num_list:
        (name, count) = db                                                                                  
        outlier_db_html += "<tr><td>{name}</td><td>{count}</td></tr>".format(name=name, count=count)

    for tup in pv_df[:rule_num].itertuples():
        (name, count, tot_fr, out_pent, out_fr, in_pent, in_fr, sugg_rcp, rank) = tup
        pv_df_html += "<tr><td>{name}</td><td>{count}</td><td>{tot_fr}</td><td>{out_pent}</td><td>{out_fr}</td><td>{in_pent}</td><td>{in_fr}</td><td>{sugg_rcp}</td><td>{rank}</td></tr>".format(name=name, count=count, tot_fr=tot_fr, out_pent=out_pent, out_fr=out_fr, in_pent=in_pent, in_fr=in_fr, sugg_rcp=sugg_rcp, rank=rank)

    buffer_html = buffer_html.replace("%outlier_db%", outlier_db_html)
    buffer_html = buffer_html.replace("%outlier_group%", outlier_group_html)
    buffer_html = buffer_html.replace("%cluster_detailed_blocks%", detailed_blocks_html)
    buffer_html = buffer_html.replace("%rulename_table%", pv_df_html)
    
    file_path=os.path.join(os.getcwd(), save_dir, "summary.html")
    with open(file_path, "w") as f:
        f.write(buffer_html)

    print("Output file: ", file_path)
    print("Display report on browser automatically ...")
    
    filename = 'file:///' + file_path
    #webbrowser.open_new_tab(filename)


def generate_calibre_db(save_dir, db_split_dict, mutli=cf.mutli):
    if not os.path.exists(os.path.join(save_dir, cf.db_report_dir)):
        os.mkdir(os.path.join(save_dir, cf.db_report_dir))

    for split in db_split_dict:
        with open(os.path.join(save_dir, cf.db_report_dir, '{}.db'.format(split)), 'w') as f:
            f.write(split+' '+str(cf.mutli)+'\n')
            for i, address in enumerate(db_split_dict[split]):
                [rule, x1, y1, x2, y2] = extract_rcp_fr(address)[3:8]
                f.write(rule+'/'+str(i)+'\n')
                f.write('1 1 3 July 2 16:16:38 2020\n')
                f.write(rule+'/'+str(i)+'\n')
                f.write('Runset File Pathname: scr/runset.cmd\n')
                f.write('Runset Commands: none\n')
                f.write('p 1 4\n')
                f.write(str(int(x1*mutli))+' '+str(int(y1*mutli))+'\n')
                f.write(str(int(x2*mutli))+' '+str(int(y1*mutli))+'\n')
                f.write(str(int(x2*mutli))+' '+str(int(y2*mutli))+'\n')
                f.write(str(int(x1*mutli))+' '+str(int(y2*mutli))+'\n')
                
def generate_cbca_report(save_dir, case_list, group_list, num_list, pv_df, total_outlier, clu_alg=cf.clu_alg, clu=cf.CLU+1, rule_num=cf.rule_num):
    outpath = os.path.join(os.getcwd(), save_dir)
    outlier_path = os.path.join(os.getcwd(), save_dir, cf.out_name)

    buffer_html = ""
    with open("src/cbca_template.html", "r") as f:
      buffer_html += f.read()

    L1_num, L2_num = 0, 0
    all_f = np.load(os.path.join(save_dir, cf.f_name))
    for address in all_f:
        rcp = extract_rcp_fr(address, False)[0]
        if rcp == 'none':
            L1_num += 1
        else:
            if rcp not in ['CongestionFix', 'NoFix', '']:
                L2_num += 1

    buffer_html = buffer_html.replace("%total_clips%", str(int((L1_num+L2_num)))+'(L1:{}, L2:{})'.format(int(L1_num), int(L2_num)))
    buffer_html = buffer_html.replace("%total_case%", str(int(len(case_list))))
    buffer_html = buffer_html.replace("%analysis_output_path%", outpath)
    buffer_html = buffer_html.replace("%outlier_count%", str(total_outlier))
    buffer_html = buffer_html.replace("%outlier_img_path%", outlier_path)

    outlier_rows_html = ""
    outlier_db_html = ""
    outlier_group_html = ""
    pv_df_html = ""
    detailed_blocks_html = ""

    for (name, (count, out, score)) in case_list:
        outlier_rows_html += "<tr><td>{name}</td><td>{count}</td><td>{out}</td><td>{score}</td></tr>".format(name=name, count=count, out=out, score=score)
    
    for db in num_list[:(cf.db_split_num)]:
        (name, count) = db
        outlier_db_html += "<tr><td>{name}</td><td>{count}</td></tr>".format(name=name, count=count)

    for tup in group_list:
        (name, (count, score)) = tup
        outlier_group_html += "<tr><td>{name}</td><td>{count}</td><td>{score}</td></tr>".format(name=name, count=count, score=score)

        [tp, stp,lyr] = name.split('/')
        img_path = os.path.join(os.getcwd(), save_dir, tp, stp, lyr, cf.pie_name.format(clu_alg, clu))
        detailed_blocks_html += '''<div class="cluster-block">
         <section>
          <span>- {tp}/{stp}/{lyr}</span>
          <span class="toggle" toggle-target-id="{target_id}">(toggle)</span></li>
         </section>
         <img id="{target_id}" class="vanish" src="{img_path}">
        </div>'''.format(tp=tp, stp=stp, lyr=lyr, target_id=tp+stp+lyr, img_path=img_path)

    for tup in pv_df[:rule_num].itertuples():
        (name, count, tot_fr, out_pent, out_fr, in_pent, in_fr, sugg_rcp, rank) = tup
        pv_df_html += "<tr><td>{name}</td><td>{count}</td><td>{tot_fr}</td><td>{out_pent}</td><td>{out_fr}</td><td>{in_pent}</td><td>{in_fr}</td><td>{sugg_rcp}</td><td>{rank}</td></tr>".format(name=name, count=count, tot_fr=tot_fr, out_pent=out_pent, out_fr=out_fr, in_pent=in_pent, in_fr=in_fr, sugg_rcp=sugg_rcp, rank=rank)

    buffer_html = buffer_html.replace("%outlier_rows%", outlier_rows_html)
    buffer_html = buffer_html.replace("%rulename_table%", pv_df_html)
    buffer_html = buffer_html.replace("%outlier_db%", outlier_db_html)
    buffer_html = buffer_html.replace("%outlier_group%", outlier_group_html)
    buffer_html = buffer_html.replace("%cluster_detailed_blocks%", detailed_blocks_html)

    file_path=os.path.join(os.getcwd(), save_dir, "summary.html")
    with open(file_path, "w") as f:
        f.write(buffer_html)

    print("Output file: ", file_path)
    print("Display report on browser automatically ...")
    
    filename = 'file:///' + file_path
    #webbrowser.open_new_tab(filename)

def cluster_group(save_dir, usage='dlc'):
    outlier_dict = {}
    pvtorcp_dict = {}
    pvtofr_dict = {}
    all_outlier = []
    
    for tp in cf.type_list:
        for dirpath, dirnames, filenames in os.walk(os.path.join(save_dir, tp)):
            if len(dirpath.split('/')) == 4:
                [stp, lyr] = dirpath.split('/')[-2:]
                group_dir = os.path.join(save_dir, tp, stp, lyr)
                ftofea, encoder_feature, fname = read_feature(save_dir, tp, stp, lyr) 

                print('----------', tp, stp, lyr, 'feature:', encoder_feature.shape )
                no_recipe_feature, no_recipe_f = [], []
                recipe_feature, recipe_f = [], []
                for f, fea in ftofea.items():
                    [rcp, fr, _, rule_name] = extract_rcp_fr(f, True)[:4]
                    if rcp in cf.label_list:
                        if 'ADFL1' in f:
                            no_recipe_feature.append(fea)
                            no_recipe_f.append(f)
                        else:
                            recipe_feature.append(fea)
                            recipe_f.append(f)
                    
                        if rule_name not in pvtofr_dict:
                            pvtofr_dict[rule_name] = [fr]
                        else:
                            pvtofr_dict[rule_name].append(fr)
                        
                labels_noise, n_noise, score = outlier_detection(no_recipe_feature)                    
                #plot_tsne(group_dir, cf.clu_alg, cf.CLU+1, no_recipe_feature, labels_noise)
                outlier_dict['/'.join([tp, stp, lyr])] = (n_noise, score)

                if n_noise > 0:
                    if len(recipe_feature) > 0:
                        inlier_fea = np.concatenate((np.array(recipe_feature), np.array(no_recipe_feature)[labels_noise == 1]), axis=0) 
                        inlier_f = np.concatenate((np.array(recipe_f), np.array(no_recipe_f)[labels_noise == 1]), axis=0)
                        outlier_fea = np.array(no_recipe_feature)[labels_noise == -1]
                        outlier_f = np.array(no_recipe_f)[labels_noise == -1]
                    else:
                        inlier_fea = np.array(no_recipe_feature)[labels_noise == 1] 
                        inlier_f = np.array(no_recipe_f)[labels_noise == 1]
                        outlier_fea = np.array(no_recipe_feature)[labels_noise == -1] 
                        outlier_f = np.array(no_recipe_f)[labels_noise == -1]
                    all_outlier += list(outlier_f)
                else:
                    inlier_fea = np.array(recipe_feature)
                    inlier_f = np.array(recipe_f)
                    outlier_fea = [] 
                    outlier_f = []
                
                for f in inlier_f:
                    [rcp, fr, rule, remark] = [extract_rcp_fr(f,True)[i] for i in [0, 1, 3, 8]]
                    if rule not in pvtorcp_dict:
                        pvtorcp_dict[rule] = [('{}({})'.format(rcp, remark), fr)]
                    else:
                        pvtorcp_dict[rule].append(('{}({})'.format(rcp, remark), fr))

                for clu_alg in cf.cluster_list:
                    for clu in cf.clu_range:
                        print('clustering...')
                        labels, clu = generate_res(group_dir, clu_alg, clu, inlier_fea, inlier_f, outlier_fea, outlier_f)
                        
                        #print('generate TSNE scatter...')
                        #plot_tsne(group_dir, clu_alg, clu, encoder_feature, labels)

                        print('generate pie chart...')
                        score, recipe, rule_name, demo, totdrc, fr_rcp, fr_rule, totclu = process_res(group_dir, clu, cf.demo_num, clu_alg)
                        plot_pie_chart(group_dir, clu_alg, clu, tp, stp, lyr, score, recipe, rule_name, totclu, totdrc, fr_rcp, fr_rule, demo)
                        
    total_outlier = len(all_outlier)
    plot_outlier_demo(save_dir, all_outlier, total_outlier)
    num_list, db_split_dict, pv_df = out2table(all_outlier, pvtorcp_dict, pvtofr_dict)
    
    return sorted(outlier_dict.items(), key=lambda d: d[1], reverse=True), num_list, db_split_dict, pv_df, total_outlier

def cluster_group_cbca(save_dir, all_outlier, all_outfea, iso, usage='cbca'):
    allout_dict = {}
    outlier_dict = {}
    for f, fea in zip(all_outlier, all_outfea):
        tp, stp, lyr = extract_group_info(f)
        group_name = '/'.join([tp, stp, lyr])
        if group_name not in allout_dict: 
            allout_dict[group_name] = {f:fea}
        else:
            allout_dict[group_name][f] = fea
    print('group_name', allout_dict.keys())

    for tp in cf.type_list:
        for lyr in cf.layer_list:
            for stp in cf.subtype_list:
                group_dir = os.path.join(save_dir, tp, stp, lyr)                                                                                                          
                if os.path.exists(group_dir):
                    ftofea, encoder_feature, fname = read_feature(save_dir, tp, stp, lyr)
                    print('----------', tp, stp, lyr, 'feature:', encoder_feature.shape )
                    group_name = '/'.join([tp, stp, lyr])
                    if group_name in allout_dict:
                        n_noise = len(allout_dict[group_name])
                        outlier_fea = np.array(list(allout_dict[group_name].values())).reshape(n_noise,-1)
                        outlier_f = list(allout_dict[group_name].keys())
                        score = round(np.mean(iso.decision_function(outlier_fea)), 4)
                    else:
                        n_noise = 0
                        outlier_fea = outlier_f = []
                        score = 0.0

                    outlier_dict[group_name] = (n_noise, score)
                    
                    inlier_fea = []
                    inlier_f = []

                    for f, fea in ftofea.items():
                        if f not in outlier_f:
                            inlier_fea.append(fea)
                            inlier_f.append(f)
                    
                    for clu_alg in cf.cluster_list:
                        for clu in cf.clu_range:
                            print('clustering...')
                            labels, clu = generate_res(group_dir, clu_alg, clu, np.array(inlier_fea).reshape(len(inlier_fea), -1), inlier_f, outlier_fea, outlier_f)

                            print('generate pie chart...')
                            score, recipe, rule_name, demo, totdrc, fr_rcp, fr_rule, totclu = process_res(group_dir, clu, cf.demo_num, clu_alg)
                            plot_pie_chart(group_dir, clu_alg, clu, tp, stp, lyr, score, recipe, rule_name, totclu, totdrc, fr_rcp, fr_rule, demo)

    return sorted(outlier_dict.items(), key=lambda d: d[1], reverse=True)

def train_isotree(use_dlc='0802', old_cbca=[]):
    save_dir = 'dlc_{}_p{}'.format(use_dlc, cf.DIM)
    x, all_f = [], []
    pvtorcp_dict = {}
    for tp in cf.type_list:
        for stp in cf.subtype_list:
            for lyr in cf.layer_list:
                group_dir = os.path.join(save_dir, tp, stp, lyr)
                if os.path.exists(group_dir):
                    fea = np.load(os.path.join(group_dir, cf.fea_name))
                    f = np.load(os.path.join(group_dir, cf.f_name))
                    x += list(fea)
                    all_f += list(f)
                else:
                    small_fea_path = os.path.join(cf.small_group_dir, cf.small_fea_name.format(tp, stp, lyr))
                    if os.path.exists(small_fea_path):
                        fea = np.load(small_fea_path)
                        f = np.load(small_f_path)
                        x += list(fea)
                        all_f == list(f)
    
    for cbca in old_cbca:
        save_dir = 'cbca_{}_p{}'.format(cbca, cf.DIM)
        infea = np.load(os.path.join(save_dir, cf.infea_name))
        #inf = np.load(os.path.join(save_dir, cf.inf_name))
        x += list(infea)
        #all_f += list(inf)
    
    for f in all_f:
        [rcp, fr, rule, remark] = [extract_rcp_fr(f, True)[i] for i in [0, 1, 3, 8]]
        if rule not in pvtorcp_dict:
            pvtorcp_dict[rule] = [('{}({})'.format(rcp, remark), fr)]
        else:
            pvtorcp_dict[rule].append(('{}({})'.format(rcp, remark), fr))
        
    x = np.array(x).reshape(len(x), -1)    
    print('train isolationforest ... (shape:{})'.format(x.shape))
    iso = IsolationForest(n_estimators=100, bootstrap=True, contamination='auto', behaviour='new', random_state=42, n_jobs=-1).fit(x)
    
    return iso, pvtorcp_dict

def test_cacb(save_dir, iso, pvtorcp_dict):
    test_data = np.load(os.path.join(save_dir, cf.fea_name))
    test_f = np.load(os.path.join(save_dir, cf.f_name))
   
    pvtofr_dict = {} 
    case_dict = {}    
    for fea, f in zip(test_data, test_f):
        case = f.split('/')[-3]
        [rcp, fr, _, rule_name] = extract_rcp_fr(f, False)[:4]
        if rcp not in ['CongestionFix', 'NoFix', '']:
            if rule_name not in pvtofr_dict:
                pvtofr_dict[rule_name] = [fr]
            else:
                pvtofr_dict[rule_name].append(fr)

            if case not in case_dict:
                case_dict[case] = {f:fea}
            else:
                case_dict[case][f] = fea
    
    in_fea, in_f = [], []
    all_outlier, all_outfea = [], []
    
    for case in case_dict:
        case_fea = list(case_dict[case].values())
        case_f = list(case_dict[case].keys())
       
        no_recipe_feature, no_recipe_f = [], []
        recipe_feature, recipe_f = [], []

        for fea, f in zip(case_fea, case_f):
            rcp = extract_rcp_fr(f, False)[0]
            if rcp == 'none':
                no_recipe_feature.append(fea)
                no_recipe_f.append(f)
            else:
                recipe_feature.append(fea)
                recipe_f.append(f) 
        
        if len(no_recipe_feature) > 0:
            no_recipe_feature = np.array(no_recipe_feature).reshape(len(no_recipe_feature), -1)
            no_recipe_f = np.array(no_recipe_f)        
            score = round(np.mean(iso.decision_function(no_recipe_feature)), 4)
            labels = iso.predict(no_recipe_feature)
            n_out = list(labels).count(-1)
        else:
            no_recipe_feature = no_recipe_f = []
            score = 0.0
            labels = []
            n_out = 0
        case_dict[case] = (len(case_f), n_out, score)
        
        if n_out > 0:
            all_outlier += list(no_recipe_f[labels==-1])
            all_outfea += list(no_recipe_feature[labels==-1]) 
            
            if (len(case_fea)-n_out) > 0:
                in_fea += recipe_feature + list(no_recipe_feature[labels==1])#.reshape(case_fea.shape[0]-n_out, -1))
                in_f += recipe_f + list(no_recipe_f[labels==1])
        else:
            in_fea += recipe_feature
            in_f += recipe_f

    np.save(os.path.join(save_dir, cf.infea_name), np.array(in_fea))
    np.save(os.path.join(save_dir, cf.inf_name), np.array(in_f))
    print('in_fea', np.array(in_fea).shape)

    num_list, db_split_dict, pv_df = out2table(all_outlier, pvtorcp_dict, pvtofr_dict)
    total_outlier = len(all_outlier)
    plot_outlier_demo(save_dir, all_outlier, total_outlier)
    outlier_dict = cluster_group_cbca(save_dir, all_outlier, all_outfea, iso, usage='cbca')

    return outlier_dict, sorted(case_dict.items(), key=lambda d: d[1][1], reverse=True), num_list, db_split_dict, pv_df, total_outlier


#outlier_dict, db_split_dict, total_outlier = cluster_group('dlc_0505_p128')
#generate_dlc_report('dlc_0505_p128', outlier_dict, db_split_dict, total_outlier)

#iso = train_isotree()
#case_dict = test_cacb('cbca_0619_p128', iso)
#geneirate_cbca_report('cbca_0619_p128', case_dict)

