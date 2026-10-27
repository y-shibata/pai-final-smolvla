# SmolVLA：入力の明るさ変更と操作の比較

黒いボウルを皿へ運ぶLIBEROの単一タスクで、「お手本を増やさず、入力画像の明るさ変更で明暗への頑健性が改善するか」を調べた。

今回の固定条件では改善は見られず、N20B（20本・明るさ変更あり）とN20（同じ20本・通常学習）のdim/bright等重みの3組平均差は **-13.3ポイント、95%参考区間 [-24.2, -3.3]**。固定20配置・3学習組・単一タスクに条件付けた探索的結果。原因の考察は未検証であり、明るさ拡張一般の有害性は主張しない。

## 結果

|比較|確定値|
|---|---|
|主比較・明暗等重み|N20 95/120（79.2%）、N20B 79/120（65.8%）|
|dimの差| -16.7 pt、参考区間 [-31.7, -3.3]|
|brightの差| -10.0 pt、参考区間 [-25.0, +5.0]|
|cleanの差| -5.0 pt、参考区間 [-21.7, +11.7]|
|教示数とclean|N10 35/60、N20 46/60、N46 42/60。単調改善なし|
|更新0回との比較|Z20 0/180、N20 141/180|
|失敗の観察|主比較97件。brightの未把持はN20 1件、N20B 12件|

全組で同じ20配置を使うため、分母60/120は独立な配置数ではない。対応bootstrapはindex単位、100,000回、seed 20261011。p値・有意差表示なし。全評価は960試行、技術エラー0、除外0。OLD46と新N46は正規化統計が異なるため同条件比較ではない。

## 固定版と条件

- LeRobot commit: 7427f31801f944f5ec9ce53cda02862ce7d1638b
- Dataset: HuggingFaceVLA/libero revision 86958911c0f959db2bbbdb107eb3e17c5f9c798e
- Base: lerobot/smolvla_base revision c83c3163b8ca9b7e67c509fffd9121e66cb96205
- Task: pick up the black bowl from table center and place it on the plate
- N10 / N20 / N20B / N46、各30,000更新、3組。LoRA rank 16 / alpha 16、batch 1。
- N20Bは学習時に一様分布0.7–1.3の倍率、2カメラ共通、RGBを0–1へclip。状態・行動は変更しない。
- 評価はindex 30–49、clean 1.00 / dim 0.65 / bright 1.35、20 Hz、最大280ステップ。固定版LIBEROの成功判定。

## 自作した部分

教示subsetと統計の固定、明るさ変更wrapper、対応配置での比較、学習・評価の保護と監査、全成否記録、事前固定の対応分析、失敗分類、図表・動画編集。SmolVLA、LeRobot、LIBERO本体は公開成果を利用した。

## 確認と再現

[REPRODUCE.md](REPRODUCE.md)に取得先・実行順・前提を記載。まずSHA256SUMSで配布物を照合し、結果CSVと図用CSVを読む。コードの元hashはCODE_PROVENANCE.json、公開準備上の変更はPUBLICATION_CHANGES.json。

**重みは非公開**。ベース・LoRA・checkpoint・HFキャッシュ・原データ・環境assetは同梱しない。再学習には固定資産を別途取得する必要がある。OLD46の参考値は非公開の既存重み・前処理を要し、公開物だけでは再実行できない。公開アーカイブ単独での再学習は未検証。元の保護用ファイルへの依存を残しており、前提を省略して一括実行しない。

|場所|内容|
|---|---|
|ルートのPython|subset、学習、評価と関連自作コード|
|subsets/|選択JSON・stats・manifest|
|p2/|保存済み学習設定・事前照合・完了時間|
|p3/|freeze、960行の結果、48セル集計、選定例のframe metadata・終端画像・入力例|
|p4a/|分析コード、図用CSV、4図、失敗分類と規則|
|videos/|選定6対のagentview等速20 fps・12本|
|publication/|storyboard・clip_table・時間対応・編集用コード|
|docs/|説明PDF|
|configs/legacy/|元の設定（重みなし）。ホームパスは~へ正規化|
|environment/|実験時pip-freeze。今回環境構築を実行した記録ではない|

選定はR1、各条件で「N20失敗/N20B成功」の最小indexと「N20B失敗」の最小index。dim→bright→cleanの順。raw80保存は20 Hz動作の4倍速相当なので、videos/では原フレームを20 fpsで保持して等速へ戻した。提出動画は動作のみ2倍速・図表と字幕は等速。全1,920本の評価映像は含めない。

結果CSVやfreeze内の~パスは元の監査用参照であり、全参照ファイルの同梱を意味しない。公開動画名との対応はpublication/clip_table.csv。

## ライセンス

自作コードは[Apache-2.0](LICENSE)。第三者資産は[THIRD_PARTY.md](THIRD_PARTY.md)。コードのライセンスをデータや重みに自動適用しない。図2は段間のみ修正し、確定CSVの値は変更していない。

## 提出資料

- [説明資料（PDF・9ページ）](docs/説明資料_PAI最終課題_y-shibata.pdf)
- [提出動画（YouTube・限定公開）](https://youtu.be/i8rYx3zfnrA)
- [Public GitHubリポジトリ](https://github.com/y-shibata/pai-final-smolvla)

公開日：2026-10-27。動画タイトル：PAI最終課題_y-shibata。
