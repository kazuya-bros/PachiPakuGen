/**
 * 素材ローダー
 * PachiPakuGen / See-Through 出力フォルダを読み込む。
 * 準拠元: SpriTalk src/main/settings/layered-sprite-importer.ts のフォルダ規約
 *
 *   root/
 *     body.png (必須) / hair.png / hair_back.png
 *     eye/ frame_001.png ...      （連番。数値部分でソート）
 *     mouth_a/ 〜 mouth_o/
 *     axis_<name>/                （拡張: 汎用パラメータ軸）
 *
 * Base64は使わず createImageBitmap + ObjectURL（本体より軽量な読込方式）。
 */
(function (NS) {
  'use strict';

  const ROOT_FILES = {
    'body.png': 'body',
    'hair.png': 'hair',
    'hair_back.png': 'hair_back',
    'arm_l.png': 'arm_l',
    'arm_r.png': 'arm_r',
  };
  const FRAME_DIRS = ['eye', 'mouth_a', 'mouth_i', 'mouth_u', 'mouth_e', 'mouth_o'];

  /** ファイル名から連番番号を抽出（frame_001.png / 001.png 両対応） */
  function frameNumber(name) {
    const m = name.match(/(\d+)\.(png|webp|jpg|jpeg)$/i);
    return m ? parseInt(m[1], 10) : null;
  }

  async function fileToFrame(file) {
    const bitmap = await createImageBitmap(file);
    return { bitmap, url: URL.createObjectURL(file) };
  }

  /**
   * パスマップ（相対パス→File）から CharacterAssets を構築
   * @param {Map<string, File>} pathMap 例: 'body.png', 'mouth_a/frame_001.png'
   */
  async function buildFromPathMap(pathMap) {
    const layers = [];
    const axes = new Map();

    // ルート直下の単体ファイル
    for (const [fname, layerName] of Object.entries(ROOT_FILES)) {
      const file = pathMap.get(fname);
      if (file) {
        layers.push({
          name: layerName,
          zIndex: NS.LAYER_Z_INDEX[layerName],
          frames: [await fileToFrame(file)],
        });
      }
    }
    if (!layers.some((l) => l.name === 'body')) {
      throw new Error('body.png が見つかりません。フォルダ構造を確認してください。');
    }

    // 連番フォルダ収集: dir名 → [{num, file}]
    const dirFiles = new Map();
    for (const [path, file] of pathMap) {
      const slash = path.indexOf('/');
      if (slash < 0) continue;
      const dir = path.slice(0, slash);
      const base = path.slice(slash + 1);
      if (base.indexOf('/') >= 0) continue; // 深い階層は無視
      const num = frameNumber(base);
      if (num === null) continue;
      if (!dirFiles.has(dir)) dirFiles.set(dir, []);
      dirFiles.get(dir).push({ num, file });
    }

    for (const [dir, entries] of dirFiles) {
      entries.sort((a, b) => a.num - b.num);
      const frames = [];
      for (const e of entries) frames.push(await fileToFrame(e.file));
      if (FRAME_DIRS.includes(dir)) {
        layers.push({ name: dir, zIndex: NS.LAYER_Z_INDEX[dir], frames });
      } else if (dir.startsWith('axis_')) {
        axes.set(dir, { name: dir, zIndex: 0, frames });
      }
      // それ以外のフォルダは無視（README参照）
    }

    const body = layers.find((l) => l.name === 'body');
    const w = body.frames[0].bitmap.width;
    const h = body.frames[0].bitmap.height;
    return NS.finalizeAssets(layers, axes, w, h);
  }

  /** <input type="file" webkitdirectory> の FileList から読込 */
  async function loadFromFileList(fileList) {
    const pathMap = new Map();
    for (const file of fileList) {
      // webkitRelativePath は「選択フォルダ名/…」なので先頭セグメントを除去
      const rel = file.webkitRelativePath || file.name;
      const parts = rel.split('/');
      const path = parts.length > 1 ? parts.slice(1).join('/') : parts[0];
      pathMap.set(path, file);
    }
    return buildFromPathMap(pathMap);
  }

  /** DnD の DataTransferItemList から読込（webkitGetAsEntry 再帰走査） */
  async function loadFromDataTransfer(items) {
    const pathMap = new Map();

    async function walkEntry(entry, prefix) {
      if (entry.isFile) {
        const file = await new Promise((res, rej) => entry.file(res, rej));
        pathMap.set(prefix + entry.name, file);
      } else if (entry.isDirectory) {
        const reader = entry.createReader();
        // readEntries は100件ずつしか返さないため繰り返し呼ぶ
        let batch;
        do {
          batch = await new Promise((res, rej) => reader.readEntries(res, rej));
          for (const child of batch) {
            await walkEntry(child, prefix + entry.name + '/');
          }
        } while (batch.length > 0);
      }
    }

    const entries = [];
    for (const item of items) {
      const entry = item.webkitGetAsEntry && item.webkitGetAsEntry();
      if (entry) entries.push(entry);
    }
    if (entries.length === 1 && entries[0].isDirectory) {
      // 単一フォルダをドロップ → そのフォルダをルートとして展開
      const reader = entries[0].createReader();
      let batch;
      do {
        batch = await new Promise((res, rej) => reader.readEntries(res, rej));
        for (const child of batch) await walkEntry(child, '');
      } while (batch.length > 0);
    } else {
      for (const e of entries) await walkEntry(e, '');
    }
    return buildFromPathMap(pathMap);
  }

  /** 既存アセットの ObjectURL / ImageBitmap を解放 */
  function disposeAssets(assets) {
    if (!assets) return;
    const disposeLayer = (l) => {
      for (const f of l.frames) {
        if (f.url && f.url.startsWith('blob:')) URL.revokeObjectURL(f.url);
        if (f.bitmap && f.bitmap.close) f.bitmap.close();
      }
    };
    for (const l of assets.layers) disposeLayer(l);
    for (const [, ax] of assets.axes) disposeLayer(ax);
  }

  NS.Loader = { loadFromFileList, loadFromDataTransfer, disposeAssets };
})(window.AnimLab);
