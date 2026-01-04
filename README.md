# CTC TrOCR

## 直近のリファクタ概要

`train.py` を `utils/` 配下に機能別で分割しました。動作はそのままで、エントリポイントは `train.py` のままです。

## ファイル構成と役割

- `train.py`: CLIの入口、引数解析、学習ループ、推論フローの呼び出し。
- `utils/text.py`: ラベル読込、語彙生成、テキストエンコード、CER算出、CTCのgreedy decode。
- `utils/image.py`: 画像ファイル解決、リサイズ/パディング前処理。
- `utils/dataset.py`: Datasetとcollate関数。
- `utils/model.py`: TrOCR encoder + CTC head モデル定義。
- `utils/train_helpers.py`: 評価/推論ヘルパー、分割ディレクトリ解決。

