import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import re
from collections import Counter
from scipy import stats

def heatmap(inp_file, columns, index, save_name=None):
    data = pd.read_csv(inp_file)
    metrics = ['rel-acc', 'a-mrr', 'c-mrr', 'a-hits1', 'a-hits5', 'a-hits10', 'c-hits1', 'c-hits5', 'c-hits10']
    dfs = {} 
    for metric in metrics:
        df = pd.pivot_table(data=data, columns= columns, index=index, values=metric)
        df.head()
        dfs[metric]= df

    plt.rcParams['font.size'] = 10
    bg_color = (0.88,0.85,0.95)
    plt.rcParams['figure.facecolor'] = bg_color
    plt.rcParams['axes.facecolor'] = bg_color
    subplot_rows = 3
    subplot_columns = 3
    #f, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(32, 25))
    f,  axes = plt.subplots(subplot_rows, subplot_columns, figsize=(30, 48))
    idx=0
    for i in range(subplot_rows):
        for j in range(subplot_columns):
            ax = sns.heatmap(dfs[metrics[i*subplot_columns+j]], cmap='coolwarm', annot=True, fmt=".3f", annot_kws={'size':9}, linewidth=0.5, square=True, vmin=0, vmax=1,cbar_kws={"shrink": 0.5}, center=0.2, ax=axes[i, j])
            ax.set_xlabel(columns)
            ax.set_ylabel(index)
            ax.invert_yaxis()
            ax.set_title(metrics[i*subplot_columns + j])
            idx +=1
    plt.xlabel(columns)
    plt.ylabel(index)
    plt.tight_layout()
    save_name1= save_name +'_heatmap_metrics.png'
    plt.savefig(save_name1, format='png')
    print("save {}".format(save_name1))

    f,  axes = plt.subplots(2, 1, figsize=(10, 10))

    metrics = [['a-mrr','a-hits1', 'a-hits5', 'a-hits10'], ['c-mrr', 'c-hits1', 'c-hits5', 'c-hits10']]
    for i in range(2):
        ax = axes[i]
        for j in range(len(metrics[i])):
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            print(metrics[i][j])
            sns.lineplot('rel-acc', metrics[i][j],  data=data, label=metrics[i][j], lw=0.5, ax =ax)
        ax.set_ylabel("values")
    #plt.ylabel('metrics')
    save_name2= save_name +'_lineplot_metrics.png'
    plt.savefig(save_name2, format='png')
    print("save {}".format(save_name2))


def get_freq_distribution(data):
    cnt = Counter()
    for (k,v) in data.items():
        cnt[v] += 1
    return cnt

def frequency_distribution(inp_file, save_name=None):
    datas=list()
    for inp_file in inp_files:
        save_name = re.split( "/|.csv", inp_file)[-2]
        data = pd.read_csv(inp_file)
        datas.append(data)

    save_name= 'log/'+save_name
    f,  ax1 = plt.subplots(1, 1, figsize=(10, 10))
    sns.distplot(datas[0]["score-1"], hist=False, kde=True, norm_hist=True, rug=False, label="TransE-1", ax=ax1);
    sns.distplot(datas[1]["score-1"], hist=False, kde=True, norm_hist=True, rug=False, label="TransER-1", ax=ax1);
    ax1.lines[1].set_linestyle("--")
    #sns.distplot(datas[1]["score-1"], hist=True, kde=True, norm_hist=True, rug=True, label="TransE_REL", ax=ax2);

    accum_score_1_5=[]
    accum_score_2_5=[]
    for i in range(1, 6):
        accum_score_1_5.append(datas[0]["score-{}".format(i)])
        accum_score_2_5.append(datas[1]["score-{}".format(i)])

    sns.distplot(accum_score_1_5, hist=False, kde=True, norm_hist=True, rug=False, label="TransE-5", ax=ax1);
    sns.distplot(accum_score_2_5, hist=False, kde=True, norm_hist=True, rug=False, label="TransER-5", ax=ax1);
    #sns.distplot(datas[0]["score-5"], hist=False, kde=True, norm_hist=True, rug=True, label="TransE-5", ax=ax1);
    #sns.distplot(datas[1]["score-5"], hist=False, kde=True, norm_hist=True, rug=False, label="TransER-5", ax=ax1);
    ax1.lines[3].set_linestyle("--")
    ax1.set_xlabel("Scores")
    ax1.set_ylabel("Normed Frequency")
    print("TransE average hits1 score:{}".format(np.mean(datas[0]["score-1"])))
    print("TransE-REL average hits1 score:{}".format(np.mean(datas[1]["score-1"])))

    print("TransE average hits5 score:{}".format(np.mean(accum_score_1_5)))
    print("TransE-REL average hits5 score:{}".format(np.mean(accum_score_2_5)))

    accum_score_1_10=[]
    accum_score_2_10=[]
    for i in range(1, 11):
        accum_score_1_10.append(datas[0]["score-{}".format(i)])
        accum_score_2_10.append(datas[1]["score-{}".format(i)])
    sns.distplot(accum_score_1_10, hist=False, kde=True, norm_hist=True, rug=False, label="TransE-10", ax=ax1);
    sns.distplot(accum_score_2_10, hist=False, kde=True, norm_hist=True, rug=False, label="TransER-10", ax=ax1);
    #sns.distplot(datas[0]["score-10"], hist=False, kde=True, norm_hist=True, rug=True, label="TransE-10", ax=ax1);
    #sns.distplot(datas[1]["score-10"], hist=False, kde=True, norm_hist=True, rug=False, label="TransER-10", ax=ax1);
    ax1.lines[5].set_linestyle("--")

    print("TransE average hits10 score:{}".format(np.mean(accum_score_1_10)))
    print("TransE-REL average hits10 score:{}".format(np.mean(accum_score_2_10)))
    
    save_name2= save_name +'_rank_score.png'
    plt.savefig(save_name2, format='png')
    print("save {}".format(save_name2))





if __name__=='__main__':
    ''' example
    heatmap(inp_file='flights.csv', 
            columns='year',
            index='month', 
            values='passengers',
            save_name='log/heatmap.png')
    '''
    ####### plot heatmap #######
    '''
    #heatmap(inp_file='../../../output/log/iptranse/iptranse_C_S_V6.csv', 
    #inp_files = ['../../../output/log/iptranse/iptranse_C_S_V6.csv',
    inp_files = ['../../../output/log/iptranse/iptranse_C_S_V7_0621.csv']
                #'../../../output/log/iptranse/iptranse_C_S_V7_bkup.csv'] 
    for inp_file in inp_files:
        save_name = re.split( "/|.csv", inp_file)[-2]
        heatmap(inp_file=inp_file, 
                columns='beta1',
                index='beta2',
                save_name= 'log/'+save_name
                )
    '''
    ####### plot heatmap #######
    inp_files = ['../../../output/results/IPTransE/C_S_V8/271_5fold/1/20200628231322/20200629231240/test_t_prediction.csv',
                '../../../output/results/IPTransE/C_S_V8/271_5fold/1/20200628231827/20200629232041/test_t_prediction.csv']
    frequency_distribution(inp_file=inp_files) #save_name= 'log/'+save_name)

    inp_files = ['../../../output/results/IPTransE/C_S_V8/271_5fold/1/20200628231322/20200629231240/test_h_prediction.csv',
                '../../../output/results/IPTransE/C_S_V8/271_5fold/1/20200628231827/20200629232041/test_h_prediction.csv']
    frequency_distribution(inp_file=inp_files) #save_name= 'log/'+save_name) 

    inp_files = ['../../../output/results/IPTransE/C_S_V8/271_5fold/1/20200628231322/20200629231240/alignment_results_12.csv',
                '../../../output/results/IPTransE/C_S_V8/271_5fold/1/20200628231827/20200629232041/alignment_results_12.csv']
    frequency_distribution(inp_file=inp_files) #save_name= 'log/'+save_name) 

    inp_files = ['../../../output/results/IPTransE/C_S_V8/271_5fold/1/20200628231322/20200629231240/alignment_results_21.csv',
                '../../../output/results/IPTransE/C_S_V8/271_5fold/1/20200628231827/20200629232041/alignment_results_21.csv']
    frequency_distribution(inp_file=inp_files) #save_name= 'log/'+save_name) 