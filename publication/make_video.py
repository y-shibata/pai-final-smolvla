"""P4bの確定済み素材を編集する。モデル・評価・統計計算は実行しない。"""
from pathlib import Path
import csv,json,subprocess,hashlib,math,os
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
R=Path.home()/'projects/pai-final-smolvla'; P=R/'p4b'; V=P/'video'
W,H=1920,1080
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
fonts={}
def font(n,b=False):
 key=(n,b)
 if key not in fonts: fonts[key]=ImageFont.truetype(BOLD if b else FONT,n)
 return fonts[key]
def txt(im,xy,s,n=36,fill='#EAF0F6',b=False):
 ImageDraw.Draw(im).text(xy,s,font=font(n,b),fill=fill,spacing=10)
def base(title,kicker='SmolVLA / 固定条件の比較'):
 im=Image.new('RGB',(W,H),'#101D2B')
 txt(im,(80,28),kicker,24,fill='#80CACA')
 txt(im,(80,78),title,52,b=True)
 return im
def caption(im,s):
 ImageDraw.Draw(im).rectangle((0,938,W,H),fill='#07131F')
 lines=s.split('\n')
 for j,line in enumerate(lines):
  box=ImageDraw.Draw(im).textbbox((0,0),line,font=font(39))
  txt(im,((W-(box[2]-box[0]))/2,948+j*56),line,39)
def card(im,x,y,w,h,title,body):
 d=ImageDraw.Draw(im);d.rounded_rectangle((x,y,x+w,y+h),radius=20,fill='#1C3246')
 txt(im,(x+30,y+24),title,38,fill='#86D8D3',b=True)
 for i,t in enumerate(body):txt(im,(x+30,y+82+i*59),t,32)
def probe(f):
 return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(f)]))
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
story=json.loads((V/'storyboard.json').read_text())
pairs=list(csv.DictReader((R/'p4a/video_candidates.csv').open()))
clips=[]; loaded=[]
(V/'segments').mkdir(exist_ok=True);(V/'realtime').mkdir(exist_ok=True)
for j,row in enumerate(pairs):
 pair=[]
 for model in ['N20','N20B']:
  src=Path(row[model+'_video_path']).expanduser()
  cap=cv2.VideoCapture(str(src)); arr=[]
  while True:
   ok,fr=cap.read()
   if not ok:break
   arr.append(Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)).resize((650,650),Image.Resampling.LANCZOS))
  cap.release()
  n=int(row[model+'_control_steps']);assert len(arr)==n
  terminal=Image.open(src.parent/f"state_{row['initial_state_index']}_terminal.png").convert('RGB').resize((650,650),Image.Resampling.LANCZOS)
  realtime=V/'realtime'/f"R1{model}_{row['condition']}_{row['initial_state_index']}_20hz.mp4"
  # raw80は20Hzの動作を4倍速で格納。各原フレームを20fpsで保持して等速素材を作る。
  subprocess.run(['ffmpeg','-v','error','-y','-i',str(src),'-vf','setpts=4*PTS','-r','20','-fps_mode','cfr','-an','-c:v','libx264','-threads','4','-preset','medium','-crf','18','-pix_fmt','yuv420p','-map_metadata','-1','-movflags','+faststart',str(realtime)],check=True)
  q=probe(realtime);st=q['streams'][0]
  assert int(st['nb_frames'])==n and abs(float(q['format']['duration'])-n/20)<0.002
  start=story[7+j][0];success=row[model+'_success']=='True'
  clips.append(dict(pair=j+1,model='R1'+model,condition=row['condition'],index=row['initial_state_index'],success=success,
      source=str(src).replace(str(Path.home()),'~'),source_sha256=sha(src),source_frames=n,raw_fps=80,raw_seconds=n/80,
      control_hz=20,control_steps=n,realtime_seconds=n/20,output_speed=2,ideal_motion_seconds=n/40,
      output_pair_start=start,output_pair_end=start+7,first_hold_frame=math.ceil(n/40*30),
      first_hold_seconds=math.ceil(n/40*30)/30,hold_description='成功・静止中' if success else '上限終了',
      realtime_file=str(realtime.relative_to(P)),realtime_sha256=sha(realtime),public_video_path='videos/'+realtime.name))
  pair.append(dict(frames=arr,terminal=terminal,n=n,success=success))
 loaded.append(pair)
with (V/'clip_table.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,clips[0].keys());w.writeheader();w.writerows(clips)
# メタデータのframe/control_step/秒を照合する。評価値は再集計しない。
row=pairs[0];src=Path(row['N20_video_path']).expanduser()
meta=json.loads((src.parent/'state_44_frames.json').read_text())
mapping=[]
for f in [0,20,40,80,160,279]:
 m=meta[f];assert m['frame']==m['control_step']==f and abs(m['simulation_time_seconds']-f/20)<1e-10
 mapping.append(dict(source_frame=f,control_step=f,raw_seconds=f/80,simulation_seconds=f/20,ideal_2x_seconds=f/40,main_output_seconds=73+f/40))
(V/'timebase_verification.json').write_text(json.dumps({'trial':'R1N20/dim/44','source':'P3 saved frame metadata','mapping':mapping,'interval_end':{'steps':280,'realtime_seconds':14,'double_seconds':7},'output_sampling':'30fps; midpoint sampling from 40fps; no motion cuts; timing quantization <1/30 second'},indent=2))
training=json.loads((R/'p2/summary.json').read_text())['runs']
def static(kind):
 if kind.startswith('fig'):
  names={'fig1':'fig1_success_rates','fig2':'fig2_primary_paired','fig3':'fig3_failure_counts','fig4':'fig4_demonstrations_clean'}
  im=Image.new('RGB',(W,H),'white')
  graph=Image.open(P/'figures'/(names[kind]+'.png')).convert('RGB').resize((1660,934),Image.Resampling.LANCZOS)
  im.paste(graph,(130,0));return im
 if kind=='mechanism':
  im=base('AIが画像・言葉・状態から動作を選ぶ')
  card(im,80,220,550,570,'入力',['俯瞰と手元のカメラ','「ボウルを皿へ」の指示','ロボットの状態'])
  card(im,690,220,540,570,'SmolVLA + LoRA',['公開の視覚・言語・行動AI','LoRA：一部の小さな行列を','学習して動作を調整'])
  card(im,1290,220,550,570,'LIBEROのロボット',['予測した動作を実行','20 Hzの制御周期','成功または上限で終了'])
  return im
 if kind=='contribution':
  im=base('再利用したものと、今回加えたもの')
  card(im,80,210,820,630,'公開成果の再利用',['SmolVLAの学習済みベース','LeRobotの学習・推論機能','LIBEROの環境と教示データ','既存の制御・前処理の接続'])
  card(im,960,210,880,630,'今回自作した部分',['教示subsetと正規化統計の固定','2カメラ共通の明るさ変更','同じ配置での対応比較・全件記録','失敗分類・図表・動画と再現記録'])
  return im
 if kind=='training':
  im=base('主比較は、同じ20本の教示で明るさ変更だけを変える')
  for i,(model,note) in enumerate([('N10','10本 / 通常学習'),('N20','20本 / 通常学習'),('N20B','同じ20本 / 明るさ変更あり'),('N46','46本 / 通常学習')]):
   card(im,100,210+i*153,1720,129,model,[note])
  txt(im,(1180,218),'全条件 30,000 更新',32,fill='#FFFFFF')
  txt(im,(1180,275),'3組（R1–R3）を実施',30)
  return im
 if kind=='evidence':
  im=base('学習の完了記録：12モデルすべて30,000更新')
  record=json.loads((P/'_evidence/v2/training_source.json').read_text())
  minutes=round(record['mean_minutes'])
  txt(im,(100,183),f'12 モデル × 30,000 更新を完了（1 run 約 {minutes} 分、R1〜R3 × N10/N20/N20B/N46）',30)
  # 保存ログのR1/N20だけを描いた図。異なるsubsetの損失は比較しない。
  im.paste(Image.open(P/'figures/loss_curve_R1N20.png').convert('RGB'),(105,265))
  return im
 if kind=='inputs':
  im=base('実際にAIへ渡した画像（同じindex 30・開始時点）')
  for i,(cond,k) in enumerate([('clean','1.00'),('dim','0.65'),('bright','1.35')]):
   x=135+i*580
   inp=Image.open(R/f'p3/eval/R1N20_{cond}_A/attempt_1/model_inputs/state_30_agentview.png').convert('RGB').resize((490,490),Image.Resampling.LANCZOS)
   txt(im,(x,215),cond+' ×'+k,36,b=True);im.paste(inp,(x,290))
  txt(im,(100,838),'2カメラともRGBに同じ倍率を掛け、0–1へclip。以後の動作映像は環境カメラの録画。',28)
  return im
 if kind=='hypothesis':
  im=base('考察（未検証）：映像だけでは原因を決められない')
  card(im,80,235,820,570,'確認できた現象',['bright：未把持 1 → 12 件','dim：未完了と脱落が増加','映像からの単一レビュー者の分類','見えない把持や力は推定しない'])
  card(im,960,235,880,570,'次に検証する候補',['対象の見え方や接近動作への影響','持上げ後の保持・運搬への影響','学習更新や拡張分布の影響','今回の結果で原因を断定しない'])
  return im
 if kind=='conclusion':
  im=base('明るさ変更の効果は、今回の条件では改善せず')
  txt(im,(190,280),'−13.3 ポイント',130,fill='#88DBD2',b=True)
  txt(im,(200,477),'N20B − N20 ／ dim・bright等重みの3組平均',37)
  txt(im,(200,539),'95% 参考区間 [−24.2, −3.3]・20 配置 × 3 組 × 2 条件',30)
  txt(im,(200,605),'固定配置・単一タスクのシミュレーション検証です。',38)
  txt(im,(200,731),'次の検証：影・反射と実機への適用',34)
  txt(im,(200,842),'出典：P3固定評価 / P4a対応分析。公開モデル・データの詳細は説明PDF。',26)
  return im
 raise ValueError(kind)
def moving(j,t,kind):
 row=pairs[j]; im=base(('タスクと問い' if kind=='intro' else '失敗の注目箇所（比較映像の再掲）' if kind.startswith('failure') else '同じ配置・同じ入力条件で比較'))
 mult={'clean':'1.00','dim':'0.65','bright':'1.35'}[row['condition']]
 txt(im,(80,145),f"R1 / {row['condition']} ×{mult} / index {row['initial_state_index']} / agentview",28)
 txt(im,(1015,35),'終了静止（動作映像は2倍速）' if t>=7 else '映像2倍速・シミュレーション時間基準',29,fill='#F5D891')
 for k,m in enumerate(['N20','N20B']):
  x=160+900*k
  d=loaded[j][k]
  ended=t>=d['n']/40
  # 30fpsの各表示区間中央に最も近い原フレームを選ぶ。終端以降は保存済みPNGに切り替える。
  idx=min(d['n']-1,int((t+1/60)*40))
  pic=d['terminal'] if ended else d['frames'][idx]
  im.paste(pic,(x+25,270))
  label='お手本20回・通常学習（N20）' if k==0 else 'お手本20回・明るさ変更あり（N20B）'
  txt(im,(x-15,180),label,29,b=True)
  status=('成功・静止中' if d['success'] else '上限終了・静止中') if ended else ('動作中（最終結果：成功）' if d['success'] else '動作中（最終結果：失敗）')
  if kind.startswith('failure') and ((j==3 and k==1) or (j==2 and k==0)):
   status='失敗例：'+('未把持' if j==3 else '持上げ後の脱落')+'（動作中）'
  ImageDraw.Draw(im).rectangle((x,218,x+700,262),fill='#123D39' if ended and d['success'] else '#24303F')
  txt(im,(x+22,221),status,30)
 return im
def encode_frames(path,frames,count):
 proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','30','-i','pipe:0','-frames:v',str(count),'-an','-c:v','libx264','-threads','4','-preset','fast','-crf','20','-pix_fmt','yuv420p','-map_metadata','-1',str(path)],stdin=subprocess.PIPE)
 try:
  for im in frames:proc.stdin.write(im.tobytes())
 finally:proc.stdin.close()
 assert proc.wait()==0
# storyboardの保存済み時刻を唯一の編集基準にする。連結後に速度は変更しない。
for seg,(a,b,kind,source,captext) in enumerate(story):
 path=V/'segments'/f'{seg:02d}_{kind}.mp4';count=(b-a)*30
 # 指定区間だけを修正する場合も、他区間の既存ファイルを必ず確認する。
 only=os.environ.get('P4B_SEGMENTS')
 if only and seg not in [int(x) for x in only.split(',')]:
  assert path.is_file();continue
 if kind=='intro' or kind.startswith(('pair','failure')):
  j=0 if kind=='intro' else int(kind[-1])
  def frames():
   for n in range(count):
    im=moving(j,n/30,kind);caption(im,captext);yield im
  encode_frames(path,frames(),count)
 else:
  im=static(kind);caption(im,captext);png=V/'segments'/f'{seg:02d}_{kind}.png';im.save(png)
  subprocess.run(['ffmpeg','-v','error','-y','-loop','1','-framerate','30','-i',str(png),'-frames:v',str(count),'-an','-c:v','libx264','-threads','4','-preset','fast','-crf','20','-pix_fmt','yuv420p','-map_metadata','-1',str(path)],check=True)
 print('SEGMENT_COMPLETE',seg,kind,flush=True)
listing=V/'segments/concat.txt';listing.write_text(''.join(f"file '{x.name}'\n" for x in sorted((V/'segments').glob('*.mp4'))))
output=V/'PAI最終課題_ACCOUNT.mp4'
subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy','-map_metadata','-1','-movflags','+faststart',str(output)],check=True)
q=probe(output);(V/'ffprobe.json').write_text(json.dumps(q,indent=2))
assert float(q['format']['duration'])<=180 and q['streams'][0]['nb_frames']=='5400'
print('VIDEO_COMPLETE',q['format']['duration'],sha(output),flush=True)
