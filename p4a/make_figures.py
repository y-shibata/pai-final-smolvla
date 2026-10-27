"""P4aの固定済み集計CSVから、提出用の静止図だけを作る。再評価は行わない。"""
from pathlib import Path
import os, csv, json, hashlib
P=Path.home()/'projects/pai-final-smolvla/p4a'
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

# 図1：各セルは同じ20配置。棒端の数値は成功数/20で、色は全図共通。
cells=read('fig1_success_rates.csv'); d={(r['run_id'],r['condition']):r for r in cells}
fig=base('図1　16モデルの条件別成功率','評価条件：clean ×1.00 / dim ×0.65 / bright ×1.35　｜　各棒は20試行、棒端は成功数 / 20')
fig.legend(handles=[Patch(facecolor=colors[c],label=c) for c in colors],loc='upper right',bbox_to_anchor=(.96,.867),ncol=3,frameon=False,fontsize=12)
gs=fig.add_gridspec(2,2,left=.11,right=.955,top=.805,bottom=.16,wspace=.33,hspace=.47)
panels=[('R1',[f'R1{m}' for m in ['N10','N20','N20B','N46']]),('R2',[f'R2{m}' for m in ['N10','N20','N20B','N46']]),('R3',[f'R3{m}' for m in ['N10','N20','N20B','N46']]),('参考：更新なし / 既存モデル',['R1Z20','R2Z20','R3Z20','OLD46'])]
for ax,(title,models) in zip([fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)],panels):
 for y,m in enumerate(models):
  for k,c in enumerate(colors):
   r=d[(m,c)];v=float(r['rate_percent']); yy=y+(k-1)*.23
   ax.barh(yy,v,height=.20,color=colors[c],zorder=2)
   if v==0: ax.plot(0,yy,'|',color=colors[c],markersize=10)
   ax.text(v+1.2,yy,f"{r['successes']}/20",fontsize=10.5,va='center')
 ax.set_yticks(range(4),models);ax.invert_yaxis();ax.set_xlim(0,105);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('成功率 (%)',labelpad=2);ax.set_title(title,loc='left',pad=13);grid(ax)
foot(fig,'共通の初期状態 index 30–49。Z20は更新0回（各組N20の統計を使用）。\nOLD46は配布の正規化統計、新N46はsubsetから再計算した統計。同条件の比較とは扱わない。出典：P3結果CSV（960試行）。')
save(fig,'fig1_success_rates')

# 図2：成功率の対と、同じindexを全条件・全組で同時に抽出した差の区間。
primary=read('fig2_primary_paired.csv')
fig=base('図2　主比較：N20B − N20','同じ初期状態・同じ条件で対応　｜　各組20試行、3組平均60観測（独立な配置は20）')
fig.legend(handles=[Line2D([],[],marker='o',color='#273340',markerfacecolor='white',linestyle='',markersize=8,label='N20'),Line2D([],[],marker='D',color='#273340',linestyle='',markersize=7,label='N20B')],loc='upper right',bbox_to_anchor=(.95,.95),ncol=2,frameon=False,fontsize=13)
for condition,bottom in [('dim',.555),('bright',.205)]:
 rate=fig.add_axes([.105,bottom,.385,.25]);delta=fig.add_axes([.585,bottom,.205,.25]);nums=fig.add_axes([.805,bottom,.19,.25]);nums.axis('off')
 selected=[next(r for r in primary if r['group']==g and r['condition']==condition) for g in ['R1','R2','R3','mean_R1_R3']]
 fig.text(.045,bottom+.265,condition,color=colors[condition],fontsize=18,weight='bold')
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

# 図3：主比較の失敗だけを集計。全5分類の0件も右表に残す。
fail=read('fig3_failure_counts.csv')
cats=['未把持','持上げ後の脱落','移動・設置未完了','判別不能','技術的不成立']
fc=['#AE4B36','#D99B24','#307B92','#9A9EA5','#403F47']
fd={(r['condition'],r['model'],r['category']):int(r['count']) for r in fail}
fig=base('図3　失敗の内訳：N20とN20B','R1–R3を合算、各モデル・各条件60試行　｜　失敗97件を映像で分類（Z20の参考9件は含めない）')
ax=fig.add_axes([.14,.23,.49,.58]);tab=fig.add_axes([.68,.23,.285,.58]);tab.axis('off')
ys=[7,6,4.5,3.5,2,1]; groups=[(c,m) for c in ['clean','dim','bright'] for m in ['N20','N20B']]
for y,(c,m) in zip(ys,groups):
 left=0
 for j,cat in enumerate(cats):
  n=fd[c,m,cat]
  ax.barh(y,n,left=left,height=.64,color=fc[j],edgecolor='white',linewidth=1)
  if n: ax.text(left+n/2,y,str(n),ha='center',va='center',fontsize=14,color='white' if j in [0,2,4] else '#17212B',weight='bold')
  tab.text(j,y,str(n),ha='center',va='center',fontsize=17,color='#17212B')
  left+=n
 ax.text(left+.35,y,f'{left}/60',va='center',fontsize=13)
ax.set_yticks(ys,[f'{c} / {m}' for c,m in groups]);ax.set_ylim(.2,7.9);ax.set_xlim(0,25);ax.set_xticks([0,5,10,15,20,25]);ax.set_xlabel('失敗件数');grid(ax)
for j,t in enumerate(['未把持','脱落','未完了','不明','技術']):
 tab.text(j,7.9,t,ha='center',fontsize=13,color=fc[j],weight='bold')
tab.set_xlim(-.5,4.5);tab.set_ylim(.2,7.9)
fig.legend(handles=[Patch(facecolor=col,label=cat) for col,cat in zip(fc,cats)],loc='lower center',bbox_to_anchor=(.53,.122),ncol=5,frameon=False,fontsize=12,handlelength=1.3,columnspacing=1.4)
foot(fig,'判別不能5/97件（5.2%）。技術的不成立は全条件0件。単一レビュー者による観察上の分類で、学習上の原因を示すものではない。\n失敗分類規則を適用前に固定。根拠frame・制御step・理由はfailure_labels.csvに記録。出典：P3結果CSV・保存動画。')
save(fig,'fig3_failure_counts')

# 図4：点が各組、黒い菱形が平均。範囲線はCIではなく3組の最小～最大。
demo=read('fig4_demonstrations_clean.csv');dd={(int(r['demonstrations']),r['group']):r for r in demo}
fig=base('図4　教示数とclean成功率','N10 / N20 / N46：各組20試行、平均60観測　｜　Z20は追加学習なし（更新0回）')
ax=fig.add_axes([.09,.255,.62,.565]);side=fig.add_axes([.765,.25,.20,.53]);side.axis('off')
rc=['#24688B','#8A50A0','#C67E12'];marks=['o','s','^'];offsets=[-1.8,-.8,1.8]
for n in [0,10,20,46]:
 r=dd[n,'mean_R1_R3'];mean=float(r['rate_percent']);lo=float(r['min_percent']);hi=float(r['max_percent'])
 ax.vlines(n,lo,hi,color='#BBC1C8',linewidth=3,zorder=1)
 for j,g in enumerate(['R1','R2','R3']):
  v=float(dd[n,g]['rate_percent'])
  ax.scatter(n+offsets[j],v,marker=marks[j],s=95,color=rc[j],edgecolor='white',linewidth=.8,zorder=3,clip_on=False)
 ax.scatter(n,mean,marker='D',s=90,color='#1E2934',zorder=4,clip_on=False)
 if n>0: ax.annotate(f"{mean:.1f}%\n({r['successes']}/60)",(n,mean),xytext=(22,-38),textcoords='offset points',fontsize=13,color='#182532')
 else: ax.annotate('各組 0/20\n平均 0/60',(n,0),xytext=(16,15),textcoords='offset points',fontsize=13)
ax.set_xlim(-4,54);ax.set_ylim(0,102);ax.set_yticks([0,20,40,60,80,100]);ax.set_xticks([0,10,20,46],['0\nZ20','10\nN10','20\nN20','46\nN46']);ax.set_xlabel('追加学習に用いる教示数（本）',labelpad=10);ax.set_ylabel('clean成功率 (%)');grid(ax,'y')
handles=[Line2D([],[],marker=marks[j],color=rc[j],linestyle='',markersize=9,label=g) for j,g in enumerate(['R1','R2','R3'])]+[Line2D([],[],marker='D',color='#1E2934',linestyle='',markersize=8,label='3組平均'),Line2D([],[],color='#BBC1C8',linewidth=3,label='3組の最小–最大（CIではない）')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.12),ncol=5,frameon=False,fontsize=12,columnspacing=1.5)
side.text(0,1,'成功数 / 20',fontsize=16,weight='bold');side.text(0,.88,'教示数    R1    R2    R3',fontsize=13)
for y,n in zip([.73,.58,.43,.28],[0,10,20,46]):
 vals=[dd[n,g]['successes'] for g in ['R1','R2','R3']]
 side.text(0,y,f'{n:>2}本       {vals[0]:>2}     {vals[1]:>2}     {vals[2]:>2}',fontsize=14)
side.text(0,.06,'平均は N20 が最大。\n教示数に対する単調増加は\nこの結果では見られない。',fontsize=13,linespacing=1.65)
foot(fig,'同じ20配置の観測を各組で繰り返す。平均の分母60は60個の独立配置を意味しない。点は識別のため横にずらして表示。\nZ20は各組N20の正規化統計を使う更新0回モデル。横軸の0は追加学習なしを表す。出典：P3結果CSV。')
save(fig,'fig4_demonstrations_clean')

# PNG/SVGおよび図用CSVを対応させ、後段の報告で同じhashを使う。
inventory=[]
for name in ['fig1_success_rates','fig2_primary_paired','fig3_failure_counts','fig4_demonstrations_clean']:
 item={'figure':name,'png_dimensions':[1920,1080]}
 for ext in ['png','svg']:
  path=out/(name+'.'+ext);item[ext]={'path':'~/projects/pai-final-smolvla/p4a/figures/'+path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 data=out/'data'/(name+'.csv')
 item['data']={'path':'~/projects/pai-final-smolvla/p4a/figures/data/'+data.name,'sha256':hashlib.sha256(data.read_bytes()).hexdigest()}
 inventory.append(item)
(out/'figure_inventory.json').write_text(json.dumps({'figures':inventory,'font':font.name,'font_family':family,'svg_text':'glyph paths for portable Japanese rendering'},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'figures':inventory,'font_family':family},ensure_ascii=False))
