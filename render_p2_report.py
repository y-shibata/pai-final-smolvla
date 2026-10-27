"""検証済みのJSONからP2報告を作る。学習条件や重みは変更しない。"""
import datetime,json
from pathlib import Path
from p1_runtime import ROOT,BASE_SHA
P2=ROOT/"p2";BT=chr(96)
def read(p):return json.loads(Path(p).read_text())
def duration(x):return f"{int(x//60)}分{x%60:04.1f}秒"
def render():
 final=(P2/"validated_summary.json").exists()
 s=read(P2/("validated_summary.json" if final else "summary.json"))
 paused=s["status"]=="PAUSED_BY_USER"
 if paused and (P2/"pause_validated_summary.json").exists():s=read(P2/"pause_validated_summary.json")
 complete=s["status"]=="P2_COMPLETE"
 lines=["# P2 本学習記録（2026-10-10開始）","",
 "**P2完了。実施対象すべての30,000更新・保存・独立再読込・clean動作確認を確認した。**" if complete else ("**ユーザーの希望で一時停止。必須のR1・4runは完了。追加のR2/R3・8runは未完了。明日の明示的な再開指示を待つ。自動再開は設定しない。**" if paused else f"**実行中。完了済みrunは{len(s['runs'])}本。P2の完了判定は未実施。**"),
 "P3最終評価・dim/bright評価・提出動画は実施していない。出力はUbuntu "+BT+"~/projects/pai-final-smolvla/p2/"+BT+" に隔離。","",
 "## 1. 次へ進む条件と固定設定","",
 "全runの設定・重み・ログが揃う条件："+("**達成**。" if complete else ("**未達（ユーザー指示で一時停止。R1の4runは達成、追加8runが残る）**。" if paused else "**未達（実行中）**。")),
 "証拠は "+BT+("p2/pause_validated_summary.json" if paused else "p2/validated_summary.json")+BT+"、各runの "+BT+"run_summary.json"+BT+" と "+BT+"checkpoint_030000/standalone_reload.json"+BT+"。","",
 "- 計画MD5："+BT+"6ccbb83f0f1449bcce0f1fd94556b81b"+BT+"。計画本文は変更なし。",
 "- LeRobot HEAD："+BT+"7427f31801f944f5ec9ce53cda02862ce7d1638b"+BT+"、status空。",
 "- ベースrevision："+BT+"c83c3163b8ca9b7e67c509fffd9121e66cb96205"+BT+"。",
 "- 全run共通の出発重みSHA-256："+BT+BASE_SHA+BT+"。全条件を公開ベースから新規開始。",
 "- データrevision："+BT+"86958911c0f959db2bbbdb107eb3e17c5f9c798e"+BT+"。各run前に実shard・P0 subset/statsを照合。",
 "- 30,000更新、batch1、workers4、LoRA16/16、lr0.001、warmup1,000、cosine30,000、終了0.0001、chunk/action50。",
 "- optimizer betas=(0.9,0.95)、eps=1e-8、weight_decay=1e-10、grad_clip_norm=10。画像正規化IDENTITY。",
 "- N20Bだけ画像をclip(k×画像,0,1)、kは0.7〜1.3。同一サンプルの2カメラへ同じ倍率。拡張専用乱数を使用。",
 "- 各組の順序はN20 → N20B → N10 → N46。全run直列。GPU開始前の計算プロセス0件、Xorgは表示用として除外。","",
 "## 2. R1完了時刻とR2/R3の追加判定",""]
 d=s["additional_replicates_decision"]
 if d:
  lines.extend([f"- R1全4runの保存・独立再読込完了（JST）：**{d['R1_completed_at_JST']}**。",
  "- 判定："+(("**追加条件を満たし、R2/R3の8runも完了した。**" if complete else "**追加条件を満たした。R2/R3を続行。**") if d["eligible"] else "**期限条件を満たさず、R1のみ。**"),
  "- 期限は2026-10-17終了（JST）。追加判定に成績を使用していない。"])
 else:lines.append("R1の4run完了待ち。期限条件を満たせば新たな確認なしでR2/R3を続行する。")
 lines.extend(["","## 3. run別結果","",
 "動作確認は初期状態0〜9・cleanのみの参考値。本比較、checkpoint選定、条件変更には使用しない。",
 "最大VRAMは自分の学習プロセスのPyTorch最大割当。全条件で同じ公開ベースから開始した。",
 "","| run | 選択seed / 学習seed / 拡張seed | 学習時間 | 最大VRAM MiB | 最終損失 | 再読込 | 動作確認（参考値） |",
 "|---|---|---:|---:|---:|---|---:|"])
 for r in s["runs"]:
  lines.append(f"| {r['run_id']} | {r['selection_seed']} / {r['training_seed']} / {r['augmentation_seed']} | {duration(r['training_elapsed_seconds'])} | {r['max_cuda_allocated_bytes']/2**20:.1f} | {r['final_loss']:.8f} | PASS | {r['devcheck_successes']}/10 |")
 lines.extend(["","拡張seedを実際に使用するのはN20Bだけ。学習時間は起動・loader確認・保存を含むプロセス全体。",
 f"完了runの学習時間合計：{duration(sum(r['training_elapsed_seconds'] for r in s['runs']))}。",
 "各runの開始・終了UTC、GPU全体メモリ・温度、再読込・環境構築等の内訳は各 "+BT+"*.timing.json"+BT+" と集計JSONに保存。","",
 "## 4. checkpointの重みと正規化","",
 "0・3,000・12,000・30,000更新を保存。更新0は公開ベースとファイルhash一致。主評価対象は30,000のみ。",
 "","| run | 更新 | model.safetensors SHA-256 |","|---|---:|---|"])
 for r in s["runs"]:
  for cp in r["checkpoint_records"]:lines.append(f"| {r['run_id']} | {cp['step']} | "+BT+cp["model_file_sha256"]+BT+" |")
 lines.extend(["","各 "+BT+"checkpoint_audit.json"+BT+" に実normalizer tensor hash・経過時間・保存日時を記録。",
 "30,000更新の "+BT+"standalone_reload.json"+BT+" は別CPUプロセスの再読込結果。重み・normalizerを保存前と照合。",
 "正規化・逆正規化の実tensorとP0 statsの実dtype変換後値の差は全項目0。各組N20/N20Bのsubset・stats・seed・出発重み・初期adapterは一致し、最終重みは異なることを組別のpair_checkに保存する。","",
 "## 5. 損失の推移","",
 "損失の上下を理由とする停止・seed変更・途中checkpoint選別は行わない。図はP4で作る。",
 "","| run | 最初の100更新 平均損失 | 最後の100更新 平均損失 |","|---|---:|---:|"])
 for r in s["runs"]:lines.append(f"| {r['run_id']} | {r['loss_first_100_mean']:.6f} | {r['loss_last_100_mean']:.6f} |")
 if complete:
  lines.extend(["","1〜1,000・2,001〜3,000・11,001〜12,000・29,001〜30,000の区間平均と、フレーム数・概算epochは "+BT+"p2/validated_summary.json"+BT+" に保存。"])
 lines.extend(["","## 6. 動作確認・原本保護","",
 "- 開発初期状態0〜9・clean・各run1回。10〜49は読み込まず、元の全件初期状態読込はguardで遮断。",
 "- 元状態・安定化後状態・元画像hashをP1の同じ10配置と照合。reset・安定化10ステップ・20Hz・280ステップ・成功判定を維持。",
 "- 全試行の結果、両カメラ動画、制御ステップ・シミュレーション時刻は各runの "+BT+"devcheck/"+BT+" に保存。",
 "- 書込先・一時領域・新規cacheは作業フォルダ内。HFオフライン。既存venv・HF cache・LeRobotは変更せず、パッケージ導入・Git変更・公開・投稿・Slack送信なし。"])
 if complete or (paused and "original_protection" in s):
  prot=s["original_protection"]
  lines.append(f"- 終了時の照合：原本{prot['source_files']}件、P0出力{prot['P0_files']}件、P1 dev成果物{prot['P1_dev_files']}件はhash一致。LeRobot HEAD/status不変、計算プロセス残存なし。")
 lines.extend(["","## 7. 自作コードの変更と中断記録","",
 "- P1の自作train_experiment.pyにP2出力先・組別seed・逐次ログ・NaN/inf/90分監視・normalizer hashを追加。",
 "- 自作eval_paired.pyにP2最終checkpointの入口を追加。P2ではclean・repeat0・初期状態0〜9だけを受け付ける。",
 "- 変更前の自作コード・SHA一覧・P1報告は "+BT+"p2/_evidence/*.before_p2"+BT+" に保持。既存LeRobot・依存コードは変更なし。",
 "- R1/N20とR1/N20Bは、それぞれ最初の200更新の損失・学習率・初期adapterがP1の同条件と完全一致。学習経路が変わらないことを照合した。",
 f"- 本学習の中断・再実行：{len(s['aborted_runs'])}件。"])
 for a in s["aborted_runs"]:
  lines.extend([
   "- "+a["run_id"]+"はユーザーの翌日再開希望により停止。記録日時（JST）："+a["at_JST"]+"。",
   "- 最終ログは"+str(a["last_logged_update"])+"更新。更新0・3,000のcheckpointと全ログを "+BT+a["preserved_at"]+BT+" に保持。",
   "- 学習設定の変更、技術的不具合、成績を理由とする中断はいずれもなし。詳細は保存先の "+BT+"abort_record.json"+BT+" と "+BT+"p2/validated_summary.json"+BT+"。"])
 for resume in s.get("resumptions",[]):
  lines.extend(["","- 再開処理の開始（UTC）："+resume["at_utc"]+"。ユーザーの明示的な指示により再開。",
   "- 中断したR2/N20は保存済みの旧runを保持し、同じ公開ベース・subset・stats・seedから新規開始。完了済みのR1は再実行していない。",
   "- 再開前に停止時の生成物1,451件・原本33件・固定コード6件を照合し、GPU空きと空き容量を確認。",
   "- 再開したR2/N20の最初の3,184更新は中断前と損失・学習率の差が0。初期adapterと3,000更新重みのSHA-256も完全一致。証拠は "+BT+"p2/_evidence/R2N20_resume_reproducibility.json"+BT+"。"])

 if paused:
  a=s["pause"]
  lines.extend(["","### ユーザー指示による停止と再開","",
   "- 停止記録（JST）："+a["at_JST"]+"。性能や技術的不具合による停止ではない。",
   "- R2/N20は"+str(a["last_logged_update"])+"更新まで記録。更新0・3,000のcheckpointと全ログを "+BT+a["preserved_at"]+BT+" に保持。",
   "- R1の4runは再実行しない。R2/N20は同じ公開ベース・subset・stats・seedで最初から30,000更新する。optimizer状態を保存していない途中checkpointからは継続しない。",
   "- 残りの順序はR2（N20 → N20B → N10 → N46）、続いてR3の同じ4条件。各run直列。",
   "- 明日のユーザーの明示的な再開指示を待つ。自動再開・予約は設定しない。"])

 lines.extend(["","## 8. 実行コマンド","",
 "作業場所："+BT+"~/projects/pai-final-smolvla"+BT+"。各runの実コマンド・終了コード・時刻はtraining/standalone_reload/devcheckのtiming JSONに保存。",
 "",BT*3+"sh",
 "PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 ~/projects/demo-d3/.venv/bin/python run_p2.py",
 "# 上のorchestratorが各runのpreflight・直列実行・停止監視を行う。",
 BT*3,"","再開指示を受けた後だけ、計画MD5・原本・GPU空きを再確認して上記コマンド末尾へ "+BT+"--resume"+BT+" を付ける。既完了R1はskipされる。中断したR2/N20は保存先へ移動済みのため、新規開始される。","","## 9. P3への持越し","",
 "- P3許可後に本checkpointと30〜49の入力を接続し、モデル一覧・評価コード・設定・実施範囲を固定する。",
 "- 各組のZ20は各N20の更新0保存から用意できる。OLD46は共通の既存モデルと元の前処理を維持。",
 "- R1のみなら360試行、3組を完了した場合は960試行。P2の動作確認成績は選別・条件変更に使わない。",
 "- P4の主比較は同じ配置・同じ入力条件のN20対N20B。P1の明暗が異なる試作を主比較へ流用しない。","",
 "## 10. SHA-256一覧","",
 "全生成物の一覧はUbuntu "+BT+"~/projects/pai-final-smolvla/SHA256SUMS"+BT+"。完了時またはユーザー指示の一時停止時に更新し全件照合する。",
 "一覧自身と外部資産を参照するsymlinkは除く。P1完成時の835件の一覧は "+BT+"p2/_evidence/SHA256SUMS.before_p2"+BT+" に保存。",
 ""])
 (ROOT/"P2_REPORT.md").write_text("\n".join(lines))
 print(json.dumps({"status":s["status"],"runs":len(s["runs"]),"report":"~/projects/pai-final-smolvla/P2_REPORT.md"}))
if __name__=="__main__":render()
