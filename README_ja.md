# ikaclipper

## 概要
- 音声を含んだメディア(動画/音声)ファイルから特定の効果音(マーカー/`markers`)を検出し、次を実現する
    - 指定したマーカー毎にファイル分割
    - 指定したマーカーから一定時間の画面キャプチャをまとめたタイル画像の生成
- 作者の環境では上記が実現したが、汎用性は未検証

## 使い方
### 必要なツール
- FFmpeg
- FFprobe
    - FFmpegに同梱される
- Python 3
    - 外部モジュールとしてnumpy, scipyを使用

### インストール
- `pip install git+https://github.com/waseriin/ikaclipper`

### 実行前の準備
- マーカーとなる音声ファイルを用意する
    - `ffmpeg -vn -ss (START:TIME) -to (END:TIME) -i input.mkv -ca copy out.flac`と実行して音声を切り出すなど
- `markers_rootdir`の(サブ)ディレクトリに設置する

### 実行
- `ikaclipper` or `python -m ikaclipper`
- `ikaclipper -h`で実行時の引数が参照可能

## 設定
- `config.ini`を特定ディレクトリに設置 (この順に参照される)
    - 実行時のカレントディレクトリ
    - `$HOME/.config/ikaclipper/` (for Linux/macOS)
    - `%USERPROFILE%/ikaclipper/` (for Windows)
- ユーザが設定していない項目はデフォルトが参照される
    - [./src/default_config.ini](./src/ikaclipper/default_config.ini)

---

## やること (やらないかも)
- [x] 基本機能の実装
- [ ] Provide README.md in English 
- [ ] Docstringを詳細に書く
    - ~~本人も忘れるので~~
- [x] 設定ファイルの中のカテゴリを整理する
- [ ] 実装済みのJSONに加えて、FFmetadataのエクスポートを実装する
    - そうすれば、チャプターを動画に埋め込めるようになる
- [ ] 実際の使用例

## 参考
- [audio-detect](https://github.com/craigfrancis/audio-detect)
