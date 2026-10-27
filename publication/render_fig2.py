"""P4aの固定済み集計CSVから、提出用の静止図だけを作る。再評価は行わない。"""
from pathlib import Path
import os, csv, json, hashlib
P=Path.home()/'projects/pai-final-smolvla/p4b'
os.environ['MPLCONFIGDIR']=str(P/'cache/matplotlib')
os.environ['XDG_CACHE_HOME']=str(P/'cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
font_manager.fontManager.addfont(str(font))
family=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.family':family,'font.size':14,'axes.labelsize':14,'axes.titlesize':18,'xtick.labelsize':12,'ytick.labelsize':14,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#AAB3BD','axes.labelcolor':'#273340','text.color':'#182532','xtick.color':'#37434E','ytick.color':'#37434E','svg.fonttype':'path','svg.hashsalt':'p4a-20261011','axes.unicode_minus':False,'figure.facecolor':'white','savefig.facecolor':'white'})
colors={'clean':'#24688B','dim':'#8054A2','bright':'#C67E12'}
out=P/'figures'
def read(name): return list(csv.DictReader((out/'data'/name).open()))
def base(title,subtitle):
 fig=plt.figure(figsize=(16,9),dpi=120)
 fig.text(.045,.953,title,fontsize=25,weight='bold',va='top')
 fig.text(.045,.901,subtitle,fontsize=13,va='top',color='#46525E')
 return fig
def foot(fig,text): fig.text(.045,.04,text,fontsize=11,va='bottom',color='#46525E',linespacing=1.55)
def grid(ax,axis='x'):
 ax.grid(axis=axis,color='#E3E7EB',linewidth=.8)
 ax.set_axisbelow(True)
def save(fig,name):
 fig.savefig(out/(name+'.png'),dpi=120)
 fig.savefig(out/(name+'.svg'),metadata={'Date':'2026-10-11','Creator':'P4a static analysis'})
 assert Image.open(out/(name+'.png')).size==(1920,1080)
 plt.close(fig)

# 図2：成功率の対と、同じindexを全条件・全組で同時に抽出した差の区間。
primary=read('fig2_primary_paired.csv')
fig=base('図2　主比較：N20B − N20','同じ初期状態・同じ条件で対応　｜　各組20試行、3組平均60観測（独立な配置は20）')
fig.legend(handles=[Line2D([],[],marker='o',color='#273340',markerfacecolor='white',linestyle='',markersize=8,label='N20'),Line2D([],[],marker='D',color='#273340',linestyle='',markersize=7,label='N20B')],loc='upper right',bbox_to_anchor=(.95,.95),ncol=2,frameon=False,fontsize=13)
for condition,bottom in [('dim',.575),('bright',.185)]:
 rate=fig.add_axes([.105,bottom,.385,.23]);delta=fig.add_axes([.585,bottom,.205,.23]);nums=fig.add_axes([.805,bottom,.19,.23]);nums.axis('off')
 selected=[next(r for r in primary if r['group']==g and r['condition']==condition) for g in ['R1','R2','R3','mean_R1_R3']]
 fig.text(.045,bottom+.245,condition,color=colors[condition],fontsize=18,weight='bold')
 rate.set_title('対応する成功率',fontsize=15,pad=14);delta.set_title('差と95%参考区間',fontsize=15,pad=14);nums.text(0,1.08,'差 [下限, 上限]（pt）',transform=nums.transAxes,fontsize=12)
 for y,r in zip([3,2,1,0],selected):
  a,b=float(r['N20_rate_percent']),float(r['N20B_rate_percent']);c=colors[condition]
  rate.plot([a,b],[y,y],color=c,linewidth=2)
  rate.plot(a,y,marker='o',markerfacecolor='white',markeredgecolor=c,markeredgewidth=2,markersize=9)
  rate.plot(b,y,marker='D',color=c,markersize=8)
  rate.annotate(f"{r['N20_successes']}/{r['denominator']}",(a,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=11)
  rate.annotate(f"{r['N20B_successes']}/{r['denominator']}",(b,y),xytext=(0,-19),textcoords='offset points',ha='center',fontsize=11)
  value,lo,hi=[float(r[k]) for k in ['difference_pp','ci95_low_pp','ci95_high_pp']]
  delta.errorbar(value,y,xerr=[[value-lo],[hi-value]],fmt='D',color=c,markersize=7,capsize=5,linewidth=2)
  nums.text(0,y,f'{value:+.1f} [{lo:+.1f}, {hi:+.1f}]',fontsize=12,va='center')
 rate.set_yticks([3,2,1,0],['R1','R2','R3','3組平均']);rate.set_ylim(-.55,3.55);rate.set_xlim(0,103);rate.set_xticks([0,25,50,75,100]);rate.set_xlabel('成功率 (%)',labelpad=4);grid(rate)
 delta.axvline(0,color='#67717A',linestyle='--',linewidth=1);delta.set_xlim(-60,40);delta.set_xticks([-60,-40,-20,0,20,40]);delta.set_ylim(-.55,3.55);delta.set_yticks([]);delta.set_xlabel('成功率差（ポイント）',labelpad=4);grid(delta);nums.set_ylim(-.55,3.55)
foot(fig,'95%参考区間：初期状態20個を単位に対応bootstrap、100,000回、seed=20261011、percentile法。\n全条件・全組で同じindexを同時に抽出。訓練乱数全般・別タスクへは一般化しない。p値・有意差表示なし。出典：P3結果CSV。')
save(fig,'fig2_primary_paired')

