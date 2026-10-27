"""R1/N20の保存済みログだけを描画する。平滑化や条件間比較は行わない。"""
from pathlib import Path
import os,csv
P=Path.home()/'projects/pai-final-smolvla/p4b'
os.environ['MPLCONFIGDIR']=str(P/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
rows=list(csv.DictReader((P/'figures/data/loss_curve_R1N20.csv').open()))
fig=plt.figure(figsize=(17.1,6.45),dpi=100,facecolor='#1C3246')
ax=fig.add_axes([.083,.23,.89,.64],facecolor='#1C3246')
ax.plot([int(r['update']) for r in rows],[float(r['loss']) for r in rows],color='#86D8D3',linewidth=1.5)
ax.set_xlim(0,30000);ax.set_ylim(0,2.75)
ax.set_xticks([0,5000,10000,15000,20000,25000,30000])
ax.ticklabel_format(axis='x',style='plain')
ax.set_xlabel('更新回数',fontproperties=font,fontsize=18,color='#EAF0F6',labelpad=7)
ax.set_ylabel('損失',fontproperties=font,fontsize=18,color='#EAF0F6',labelpad=12)
ax.tick_params(colors='#EAF0F6',labelsize=16)
for s in ax.spines.values():s.set_color('#71879B')
ax.grid(alpha=.18,color='white')
fig.text(.05,.94,'R1 / N20：学習損失の推移（保存ログ・20更新ごと）',fontproperties=font,fontsize=21,color='#EAF0F6',va='center')
fig.text(.05,.062,'subset ごとに正規化が違うため、他条件との損失比較はしない',fontproperties=font,fontsize=18,color='#EAF0F6',va='center')
fig.savefig(P/'figures/loss_curve_R1N20.png',dpi=100,facecolor=fig.get_facecolor())
plt.close(fig)
