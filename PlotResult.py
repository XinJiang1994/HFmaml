from scipy import io
import numpy as np
import matplotlib.pyplot as plt
def load_data(path):
    loss=io.loadmat(path)
    loss=np.squeeze(loss['losses']).tolist()
    return loss

def plot_result(filenames,fig_size=(14,8)):
    plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号

    # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1
    plt.figure(figsize=fig_size)
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    legends=['HFmaml','Fmaml','FedAvg']
    losses = [load_data(p) for p in filenames]
    for i,loss in enumerate(losses):
        plt.plot(loss, label=legends[i])
    plt.xlabel('Num rounds',fontsize=13, fontweight='bold')
    plt.ylabel('Loss',fontsize=13, fontweight='bold')

    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold')

    plt.savefig('loss.png',format='svg')
    plt.show()

if __name__=='__main__':
    # filenames = ['losses_OPT_HFfmaml_Datasetcifar10_round_150_rho_0.5.mat', 'losses_OPT_fmaml_Datasetcifar10_round_150.mat',
    #              'losses_OPT_fedavg_Datasetcifar10_round_150.mat']
    filenames=['losses_OPT_HFfmaml_Datasetcifar10_round_500_rho_1.mat']
    plot_result(filenames)


