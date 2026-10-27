"""P3検査結果を報告し、生成物hash一覧を確定する。比較判断・統計解析はしない。"""
from pathlib import Path
import datetime,json,os
from p1_runtime import ROOT,sha,save,short
P3=ROOT/"p3"
def read(p):return json.loads(Path(p).read_text())
def q(text):return chr(96)+str(text)+chr(96)
def jst(t):return datetime.datetime.fromisoformat(t).astimezone(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
def dur(v):return f"{int(v//60)}分{v%60:.1f}秒"
def main():
 v=read(P3/"validated_summary.json");f=read(P3/"freeze.json");s=read(P3/"summary.json")
 assert v["status"] in ["P3_COMPLETE","P3_EVALUATED_HASH_REVIEW_REQUIRED"] and s["status"]=="EVALUATION_COMPLETE"
 # reportに一覧hashを記載するためreport自身は除外。独立hashをMac記録へ渡す。
 scope={"excluded_files":["SHA256SUMS","P3_REPORT.md"],"excluded_directories":["cache","tmp","__pycache__"],
  "excluded_symlinks":True,"reason":"manifest self-reference and report-manifest hash cycle; exclude mutable cache/temp",
  "P3_REPORT_hash_location":"Mac P3 note, Mac JSON p3 key, completion report"}
 save(P3/"_evidence/hash_list_scope.json",scope)
 records=[]
 for current,dirs,files in os.walk(ROOT,followlinks=False):
  current=Path(current)
  dirs[:]=[d for d in dirs if d not in scope["excluded_directories"] and not (current/d).is_symlink()]
  for name in files:
   p=current/name;rel=str(p.relative_to(ROOT))
   if p.is_symlink() or rel in scope["excluded_files"]:continue
   records.append((rel,sha(p)))
 records.sort();(ROOT/"SHA256SUMS").write_text("".join(d+"  "+p+"\n" for p,d in records))
 for p,d in records:assert sha(ROOT/p)==d,p
 manifest_sha=sha(ROOT/"SHA256SUMS")
 cell={(x["model_id"],x["condition"]):x for x in v["cell_counts"]}
 # 内部timerの合計と、その外側のtimerとの差も内訳に含める。
 timed=v["timing_seconds"]
 timed["intra_wave_other_seconds"]=timed["total_seconds"]-sum(timed[k] for k in [
  "env_build_and_reset_seconds","model_load_seconds","inference_and_simulation_seconds","video_save_seconds"])
 timestamp_difference=(datetime.datetime.fromisoformat(v["finished_at_utc"])-datetime.datetime.fromisoformat(v["started_at_utc"])).total_seconds()
 before_start_record=timed["evaluation_wall_seconds"]-timestamp_difference
 L=["# P3 最終評価記録（2026-10-11）","",
 "**"+("P3完了。計画§10 P3の次へ進む条件を達成した。" if v["status"]=="P3_COMPLETE" else "評価収録済み。対応hashの不一致について判断が必要。")+"**",
 "16モデル×3画像条件×20初期状態＝960件。成功数のみを記録し、比較の差・区間・判断・図表・動画編集はP4へ持ち越す。","",
 "## 1. 次へ進む条件","",
 "| 条件 | 判定 | 証拠 |","|---|---|---|",
 f'| 評価の対応が正しい | {"達成" if not v["paired_hash_mismatches"] else "未達・要確認"} | p3/_evidence/paired_hash_validation.json、960行・96wave・48セルの照合 |',
 "| 技術エラーの扱いが明確 | 達成 | 凍結した1回再実行・欠損・停止規則、p3/summary.json |","",
 "## 2. 実行前の凍結","",
 f'- 凍結時刻（JST）：{jst(f["frozen_at_utc"])}。',
 f'- freeze.json SHA-256：{q(v["freeze_sha256"])}。',
 f'- 計画MD5：{q(f["plan_md5"])}。計画本文は変更していない。',
 f'- 初期状態抽出開始（JST）：{jst(read(P3/"_evidence/initial_state_extraction.json")["started_at_utc"])}。凍結より後。',
 "- 実行順：R1→R2→R3の各組でN20→N20B→N10→N46、次にR1Z20→R2Z20→R3Z20、最後にOLD46。",
 "- 各モデル内はclean→dim→bright、各条件内はwave A（30〜39）→B（40〜49）。環境順は各wave内でindex昇順。",
 "- hard reset・安定化10step・20Hz・上限280・固定revisionのcheck_success。各waveで10環境を新規構成し、初回rolloutだけを使用。",
 "- 環境seed＝20263010＋global index、policy seed＝モデル読込後に20264010へ戻す。",
 "- clean1.00／dim0.65／bright1.35を試行中固定。両カメラをclip(k×RGB,0,1)としpolicy前処理へ渡す。",
 f'- LeRobot HEAD：{q(f["lerobot_head"])}。',
 f'- ベースrevision：{q(f["base_revision"])}、重みSHA-256：{q(f["base_weights_sha256"])}。',
 f'- データsnapshot：{q(f["dataset_revision"])}。32実shardのhashを凍結時・終了時に照合。',
 f'- pruned_init SHA-256：{q(f["pruned_init_sha256"])}。凍結後・抽出前および終了時に全体照合。',
 "- 初期状態の表記は「本計画の開発・モデル選定に使用しない固定評価条件」。過去未使用や訓練教示との非重複とは断定しない。","",
 "### モデル16個のhash","",
 "前処理hashはpolicy前後処理・tokenizerの名前/hashリスト全体のSHA-256。normalizer個別ファイルhashと新モデルのnormalizer tensor hashもfreeze.jsonに保存。",
 "Z20は各組N20の更新0回の重み・stats、OLD46は既存LoRA adapterと元の配布前処理を使用。","",
 "| モデル | 重みSHA-256 | 前処理SHA-256 |","|---|---|---|"]
 for m in f["models"]:L.append(f'| {m["model_id"]} | {q(m["weights_sha256"])} | {q(m["processor_sha256"])} |')
 L+=["","### 評価コード","",
 "既存eval_paired.py・brightness_transform.py・p1_runtime.pyは変更なし。新しいeval_p3.pyへ30〜49と本checkpointを接続し、必要な記録を追加した。",
 "入力記録用コピーを除いた制御ループのASTと、model読込・前後処理・動画writer等6関数のASTはP1と同一。p3/_evidence/pre_freeze_logic_review.jsonと差分ファイルに保存。",
 "評価後の検査・報告用コードは評価プロセスから呼ばれず、凍結された評価ロジックの依存ではない。","",
 "| 自作ファイル | 凍結時SHA-256 |","|---|---|"]
 for n,d in f["code_sha256"].items():L.append(f"| {q(n)} | {q(d)} |")
 L+=["","## 3. 完了・不成立・再実行","",
 f'- 予定960件、完了{v["completed_trials"]}件、最終不成立{v["unestablished_trials"]}件。',
 f'- 技術的不成立の試行記録{v["technical_error_attempt_records"]}件、再実行対象{v["retry_trial_count"]}件。',
 "- タスク失敗は再実行しない。技術的不成立だけ同条件を1回再実行し、それでも不成立なら成否をnullとし完了分母から除く。",
 "- batch10の乱数列を保つため再実行時は同じwaveを再構成し、不成立indexだけ採用。他indexは再実行前の結果を保持する規則を凍結。",
 "- 1セル3件以上・全体20件以上の技術的不成立、OOM、3時間超見込み、空き50GiB未満で停止する規則を適用。","",
 "技術的不成立・再実行一覧："+("該当なし。" if not s["technical_errors"] else "p3/summary.jsonのtechnical_errorsと各attemptログを参照。"),"",
 "### セル別成功数／完了数（判断文なし）","",
 "| モデル | clean | dim | bright | 予定件数 |","|---|---:|---:|---:|---:|"]
 for m in f["models"]:
  mid=m["model_id"];vals=[f'{cell[(mid,c)]["successes"]}/{cell[(mid,c)]["completed"]}' for c in ["clean","dim","bright"]]
  L.append(f'| {mid} | {" | ".join(vals)} | 各20 |')
 L+=["",
 "**損失の解釈**：subsetごとに正規化統計が違うため、学習損失の尺度は揃っていない。N46とN10/N20の損失の大小を学習品質や成功率の代わりにしない。P2の初期状態0〜9・cleanの10試行も参考値であり、P3の結論を代替しない。","",
 "## 4. 対応・保存記録の整合性","",
 f'- 20状態×16モデル×3条件の{v["paired_hash_checked_observations"]}観測を照合。原初期状態・安定化後状態・元画像2カメラの不一致は{len(v["paired_hash_mismatches"])}件。',
 "- global indexで対応付け、AとBを配列位置だけで比較していない。",
 "- results_all.csvは960行・重複なし。48セル各20行、cell_counts.csvは96wave。結果行の重み・前処理hashは凍結一覧と一致。",
 f'- 受理した両カメラMP4は{v["videos"]}本。全件ffprobe走査でデコードframe数＝制御step数、80fps、256×256、再生時間＝step数/80を確認。',
 f'- モデル入力代表記録は{v["input_representative_trial_count"]}試行×2カメラ。初回policy前処理後のfloat32入力・hash・PNGを保存し、倍率・クリップ・PNGとの数値対応を確認。',
 "- 全フレームに制御step・相対シミュレーション時刻・MuJoCo時刻を保存。映像は編集前のraw80素材。",
 "- 16モデル終了時と各wave終了時に計算プロセス0件を確認。Xorgは表示用として除外。",
 f'- P2までの生成物2,559件・原本33件・凍結コード・checkpoint・データshardは終了時もhash一致。終了時空き容量{v["free_GiB"]:.2f} GiB。',"",
 "## 5. 時刻・所要時間・VRAM","",
 f'- 評価開始（JST）：{jst(v["started_at_utc"])}。',
 f'- 評価終了（JST）：{jst(v["finished_at_utc"])}。',
 f'- 評価全体：{dur(v["timing_seconds"]["evaluation_wall_seconds"])}。',
 f'- 事前確認・準備（指示受領10:44:48 JST→凍結）：{dur((datetime.datetime.fromisoformat(f["frozen_at_utc"])-datetime.datetime.fromisoformat("2026-10-11T01:44:48.969753+00:00")).total_seconds())}。',
 f'- 凍結→抽出・評価起動：{dur((datetime.datetime.fromisoformat(v["started_at_utc"])-datetime.datetime.fromisoformat(f["frozen_at_utc"])).total_seconds())}。',
 f'- 完了後の全件検査：{dur(v["validation_seconds"])}。hash一覧作成・報告整形は別。',
 f'- 経過時間の計測は開始時刻記録前の凍結・GPU確認も含むため、表示された開始/終了時刻の差より{before_start_record:.1f}秒長い。',
 "", "| 内訳 | 合計 |","|---|---:|"]
 labels={"env_build_and_reset_seconds":"環境構築・reset","model_load_seconds":"モデル読込","inference_and_simulation_seconds":"推論・simulation",
 "video_save_seconds":"動画・入力記録保存","intra_wave_other_seconds":"wave内のhash照合・記録等","process_startup_cleanup_other_seconds":"プロセス起動・終了等","orchestrator_between_waves_seconds":"wave間の監査・記録等"}
 for k,label in labels.items():L.append(f'| {label} | {dur(v["timing_seconds"][k])} |')
 L+=["",f'- policyのPyTorch最大割当{v["max_cuda_allocated_MiB"]:.1f} MiB、最大予約{v["max_cuda_reserved_MiB"]:.1f} MiB。',
 f'- nvidia-smiのGPU全体観測最大{v["peak_total_gpu_memory_MiB"]} MiB。描画とXorgを含む別指標。',"",
 "## 6. P4への持越し","",
 "- 主比較はN20B−N20。各組でdim・brightを同じ重みで平均し、R1〜R3の平均差と各組の差を表示する。P3では算出していない。",
 "- 参考区間を出す場合は同じ初期状態indexの全条件・全組をひとまとまりに再標本化する。20配置に条件付けた区間であり、別タスクや訓練乱数全般へ一般化しない。",
 "- R1の代表映像は条件ごとに①N20失敗・N20B成功の最小index、②N20B失敗の最小index。該当なしなら明記して最小indexの対応ペア。dim・bright、次にcleanの順を守る。選定・編集はP4で行う。",
 "- 同じ配置・同じ入力条件を左右比較する。raw80は20Hz基準で4倍速相当なので、step記録で等速へ戻してから動作だけ2倍速にし、図表等は等速にする。",
 "- 成功フラグだけから失敗原因を推測せず、動画で不明なら判別不能とする。",
 "- 最終評価後の学習・設定調整を行わない。P3では比較差・区間・図表・提出動画・PDF・公開を実施していない。","",
 "## 7. 生成物SHA-256","",
 f'- p3/freeze.json：{q(v["freeze_sha256"])}',
 f'- p3/results_all.csv：{q(v["results_all_csv_sha256"])}',
 f'- p3/cell_counts.csv：{q(v["cell_counts_csv_sha256"])}',
 f'- p3/validated_summary.json：{q(sha(P3/"validated_summary.json"))}',
 f'- SHA256SUMS：{q(manifest_sha)}。対象{len(records):,}件を作成後に全件再照合し一致。',
 "- 全件一覧は ~/projects/pai-final-smolvla/SHA256SUMS。P3前の一覧はp3/_evidence/SHA256SUMS.before_p3に保持。",
 "- 一覧自身と、一覧hashを本文に記載するP3_REPORT.mdは循環参照を避けるため除外。P3_REPORT.mdの独立hashはMac記録・JSON・完了報告に記載。cache/tmpと外部参照symlinkも対象外。p3/_evidence/hash_list_scope.jsonに除外理由を保存。",
 "- 原本・venv・HFキャッシュ・P2 checkpointは変更していない。HFオフラインで実行し、導入・Git変更・公開・投稿・Slack送信はなし。",""]
 (ROOT/"P3_REPORT.md").write_text("\n".join(L))
 print(json.dumps({"status":v["status"],"manifest_sha256":manifest_sha,"manifest_count":len(records),
 "P3_REPORT_sha256":sha(ROOT/"P3_REPORT.md"),"validated_summary_sha256":sha(P3/"validated_summary.json")}))
if __name__=="__main__":main()
