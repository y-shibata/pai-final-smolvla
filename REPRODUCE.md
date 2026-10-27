# 再現手順とコードの前提

## 確認可能な範囲

CSVから今回の成否・成功数・対応差・図の値を追跡できる。P4aの分析方法とコードを同梱した。P4bでは再評価や数値の再計算を行っていない。

失敗分類は映像観察による。全97件の根拠stepと理由を提供するが、公開映像は12本のみ。全件の映像再レビューには非公開の全評価映像が必要。元の学習・評価コードも同梱するが、公開アーカイブ単独での再学習は未実行・未検証。元の保護・監査ファイルへの依存を残しており、単独ワンコマンドでの再現パッケージとはしていない。

OLD46は非公開の既存LoRAと配布前処理を使った参考値。再実行にはそれらが必要。主比較N20/N20Bの再学習と分ける。

## 固定環境と資産

Linux / Python 3.12。全依存版はenvironment/pip-freeze.txt。主な版：torch 2.11.0+cu128、NumPy 2.2.6、transformers 5.5.4、peft 0.20.0、hf_libero 0.1.4、mujoco 3.8.1、robosuite 1.4.0。

取得先：

- [LeRobot固定commit](https://github.com/huggingface/lerobot/tree/7427f31801f944f5ec9ce53cda02862ce7d1638b)
- [データ固定revision](https://huggingface.co/datasets/HuggingFaceVLA/libero/tree/86958911c0f959db2bbbdb107eb3e17c5f9c798e)
- [ベース固定revision](https://huggingface.co/lerobot/smolvla_base/tree/c83c3163b8ca9b7e67c509fffd9121e66cb96205)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- 内部VLM：HuggingFaceTB/SmolVLM2-500M-Video-Instruct。固定ベースに対応する公開キャッシュを別途準備する。

各資産の条件を確認して取得し、manifestのshard SHA、ベースSHA、初期状態ファイルSHAを照合する。今回これらの再取得・再配布は行っていない。

コードの想定配置：

~~~
~/projects/
├── demo-d3/
│   ├── .venv/
│   ├── lerobot/
│   ├── hf_cache/hub/
│   ├── lerobot_cache/hub/
│   ├── libero_config/config.yaml
│   ├── repro/phase2_selected_shards.sha256
│   └── outputs/phase2_lora_r16_s30000_46ep/checkpoints/030000/pretrained_model/
│       ├── config.json
│       └── train_config.json
└── pai-final-smolvla/
~~~

参照用アーカイブと実験作業先は別コピーとする。既存結果へ上書きしない。libero_config/config.yamlには環境のbddl_files・init_states・assets等の実パスが必要。p1_runtime.setup()はcache・tmpを作業先へ向け、作業先外への書込を拒否し、HFをオフラインにする。

## 元の実験の実行順

以下は元の入口と前提を示す。アーカイブ直下で無条件に実行する手順ではない。

1. prepare_subsets.py：固定教示の照合とsubset・統計の作成。出力はsubsets/に同梱。選択条件はexperiment_conditions.p0_input.json。
2. prepare_p1.py：evidence/initial_state_check.jsonのhashに合う初期状態から開発用0–9のみを抽出。P0当時のSHA一覧を保存する。公開アーカイブのSHAはP0当時の一覧と異なるため、新規実験では準備した資産の内容を照合して新たな初期一覧を作る。
3. 小規模確認の入口：

~~~sh
python train_experiment.py --scope p1 --replicate R1 --condition N20 --updates 200
~~~

4. P2はR1→R2→R3、各組N20→N20B→N10→N46。run_p2.pyがモデル・shard・subset・コードhash・GPU・容量を確認し、次の入口を呼ぶ。

~~~sh
python train_experiment.py --scope p2 --replicate R1 --condition N20 --updates 30000
python train_experiment.py --scope p2 --replicate R1 --condition N20B --updates 30000
~~~

各runは既存run_config.jsonの上書きを拒否し、先に検証したpreflight.jsonが必要。公開された過去のPASS記録を新しい環境の検証結果として流用しない。監視側はdev/evidence/P0_SHA256SUMS.before_p1、evidence/source_inventory.json、p2/_evidence/code_sha256.json等も使用する。run_p2.pyのpreflight()が必要条件の実装。

5. 学習完了後にモデルと条件をfreezeし、prepare_p3.pyで最終用30–49を抽出。run_p3.pyから各waveを評価する。

~~~sh
python eval_p3.py --model R1N20 --condition dim --wave A --attempt 1
python eval_p3.py --model R1N20 --condition dim --wave B --attempt 1
~~~

対応する新規freeze・checkpoint・初期状態がある場合の入口。元のfreezeは今回の実測モデルhashを持つ証拠であり、再学習した別の重みに合わせて書き換えない。新実験は新しいfreezeと記録を作る。成功判定・seed・配置・倍率・上限を変えた場合は同じ実験と称さない。

validate_p3.pyとfinalize_p3.pyは全件照合・集計の実装。全試行の内部出力と監査記録を要し、公開12本だけで全監査を実行することはできない。

## ホームパスの正規化

configs/legacy/train_config.jsonとphase2_selected_shards.sha256は、個人名を含むホーム絶対パスを~に正規化した。選択やデータ内容は変わらないがファイルのバイトhashは元と異なる。PUBLICATION_CHANGES.jsonに元hashを記録した。

元のprepare_subsets.pyは旧manifestのバイトhashを固定値で検証し、Path(original)で読む。このため公開した~版をそのまま渡すと保護条件で停止する。別環境でsubset作成を再実行する場合は、パス読込を.expanduser()へ変更し、準備したmanifestを内容（固定revision・32shardのpathとSHA）で照合してから、その公開用hashを採用する必要がある。episode選択・統計・評価方法を変更する理由にはしない。元コードは比較可能性のため未改変で保存した。

同梱の選択JSON・statsを使用するsubset_dataset.pyは既に.expanduser()に対応している。レガシーconfig.jsonは学習設定の出発点として参照するだけで、N20/N20Bの学習にOLD46のLoRA重みを読み込むわけではない。

## P4aの数値と図

解析用のコピーをコードの想定する~/projects/pai-final-smolvla/に置く。元の公開証拠は別途保存する。

~~~sh
python p4a/analyze.py
python p4a/aggregate_failures.py
python p4a/make_figures.py
~~~

analyze.pyは固定計画・結果・freeze・分類規則のhashを照合する。時間基準照合用のR1N20/dim/index44のframe metadataも同梱した。aggregate_failures.pyは保存ラベルの集計であり、自動分類ではない。P4bではこれらを実行せずP4aの値をそのまま使った。

make_figures.pyはP4a原版。図2の段間修正はpublication/render_fig2.py。入力はp4b/figures/data/、出力はp4b/figures/なので、その構成を作ってから実行する。変更は配置のみ。

## 動画の再編集

videos/の12本は原フレーム順を20 fpsで保持した等速素材。元raw80の4倍速相当の格納を補正した。フレーム数・理想2倍速秒・成功時の静止開始はpublication/clip_table.csvで追跡できる。終端PNGは最後の制御後の静止用で、raw80の最終保存フレームと区別する。

publication/make_video.pyは実際の編集コード。元P3のraw80パス、p4b/figures/、p4b/video/storyboard.json、p2/summary.jsonを参照する。公開12本から再編集する場合は等速素材から2倍速化するか、原フレーム順を80 fpsで再格納して元パスへ置く。提出MP4全体をさらに加速しない。

再編集後は180秒以内、30 fps、左右同速度、字幕3秒以上、成否・条件・速度表示、個人情報を再確認する。
