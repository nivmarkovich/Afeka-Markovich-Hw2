/**
 * Pure Vanilla JS CNN — forward pass, backpropagation, training.
 * No external libraries.
 */

(function () {
  'use strict';

  const INPUT_SIZE = 28;
  const NUM_CLASSES = 3;
  const CLASS_NAMES = ['Circle', 'Square', 'Triangle'];
  const STORAGE_KEY = 'vanilla_cnn_model_v1';
  const POOL_SIZE = 2;

  /* ---------- Matrix / tensor utilities ---------- */

  function create2D(rows, cols, fill) {
    const a = new Array(rows);
    for (let r = 0; r < rows; r++) {
      a[r] = new Array(cols);
      for (let c = 0; c < cols; c++) {
        a[r][c] = fill !== undefined ? fill : 0;
      }
    }
    return a;
  }

  function create3D(h, w, d, fill) {
    const a = new Array(h);
    for (let i = 0; i < h; i++) {
      a[i] = new Array(w);
      for (let j = 0; j < w; j++) {
        a[i][j] = new Array(d);
        for (let k = 0; k < d; k++) {
          a[i][j][k] = fill !== undefined ? fill : 0;
        }
      }
    }
    return a;
  }

  function clone3D(tensor) {
    const h = tensor.length;
    const w = tensor[0].length;
    const d = tensor[0][0].length;
    const out = create3D(h, w, d, 0);
    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++) {
        for (let k = 0; k < d; k++) {
          out[i][j][k] = tensor[i][j][k];
        }
      }
    }
    return out;
  }

  function convOutputSize(size, filterSize) {
    return size - filterSize + 1;
  }

  function poolOutputSize(size, poolSize) {
    return Math.floor(size / poolSize);
  }

  function randomNormal(mean, std) {
    let u = 0;
    let v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    return mean + std * z;
  }

  function initFilters(count, fSize, inChannels) {
    const scale = Math.sqrt(2 / (fSize * fSize * inChannels));
    const filters = [];
    for (let f = 0; f < count; f++) {
      const filter = create3D(fSize, fSize, inChannels, 0);
      for (let i = 0; i < fSize; i++) {
        for (let j = 0; j < fSize; j++) {
          for (let c = 0; c < inChannels; c++) {
            filter[i][j][c] = randomNormal(0, scale);
          }
        }
      }
      filters.push(filter);
    }
    return filters;
  }

  function initBiases(count) {
    const b = new Array(count);
    for (let i = 0; i < count; i++) b[i] = 0;
    return b;
  }

  /* ---------- Convolution (valid, stride 1) ---------- */

  function conv2dForward(input, filters, biases) {
    const inH = input.length;
    const inW = input[0].length;
    const inC = input[0][0].length;
    const fSize = filters[0].length;
    const numFilters = filters.length;
    const outH = convOutputSize(inH, fSize);
    const outW = convOutputSize(inW, fSize);
    const output = create3D(outH, outW, numFilters, 0);

    for (let f = 0; f < numFilters; f++) {
      const kernel = filters[f];
      for (let i = 0; i < outH; i++) {
        for (let j = 0; j < outW; j++) {
          let sum = biases[f];
          for (let ki = 0; ki < fSize; ki++) {
            for (let kj = 0; kj < fSize; kj++) {
              for (let c = 0; c < inC; c++) {
                sum += input[i + ki][j + kj][c] * kernel[ki][kj][c];
              }
            }
          }
          output[i][j][f] = sum;
        }
      }
    }
    return output;
  }

  function relu3D(tensor) {
    const h = tensor.length;
    const w = tensor[0].length;
    const d = tensor[0][0].length;
    const out = create3D(h, w, d, 0);
    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++) {
        for (let k = 0; k < d; k++) {
          out[i][j][k] = tensor[i][j][k] > 0 ? tensor[i][j][k] : 0;
        }
      }
    }
    return out;
  }

  function reluBackward(gradOut, preActivation) {
    const h = gradOut.length;
    const w = gradOut[0].length;
    const d = gradOut[0][0].length;
    const grad = create3D(h, w, d, 0);
    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++) {
        for (let k = 0; k < d; k++) {
          grad[i][j][k] = preActivation[i][j][k] > 0 ? gradOut[i][j][k] : 0;
        }
      }
    }
    return grad;
  }

  function maxPool2dForward(input, poolSize) {
    const inH = input.length;
    const inW = input[0].length;
    const depth = input[0][0].length;
    const outH = poolOutputSize(inH, poolSize);
    const outW = poolOutputSize(inW, poolSize);
    const output = create3D(outH, outW, depth, 0);
    const argmaxI = create3D(outH, outW, depth, 0);
    const argmaxJ = create3D(outH, outW, depth, 0);

    for (let c = 0; c < depth; c++) {
      for (let i = 0; i < outH; i++) {
        for (let j = 0; j < outW; j++) {
          let maxVal = -Infinity;
          let maxIi = 0;
          let maxJj = 0;
          for (let pi = 0; pi < poolSize; pi++) {
            for (let pj = 0; pj < poolSize; pj++) {
              const ii = i * poolSize + pi;
              const jj = j * poolSize + pj;
              const v = input[ii][jj][c];
              if (v > maxVal) {
                maxVal = v;
                maxIi = ii;
                maxJj = jj;
              }
            }
          }
          output[i][j][c] = maxVal;
          argmaxI[i][j][c] = maxIi;
          argmaxJ[i][j][c] = maxJj;
        }
      }
    }
    return { output: output, argmaxI: argmaxI, argmaxJ: argmaxJ, poolSize: poolSize };
  }

  function maxPool2dBackward(gradOut, argmaxI, argmaxJ, inH, inW) {
    const outH = gradOut.length;
    const outW = gradOut[0].length;
    const depth = gradOut[0][0].length;
    const gradIn = create3D(inH, inW, depth, 0);

    for (let c = 0; c < depth; c++) {
      for (let i = 0; i < outH; i++) {
        for (let j = 0; j < outW; j++) {
          const g = gradOut[i][j][c];
          const ii = argmaxI[i][j][c];
          const jj = argmaxJ[i][j][c];
          gradIn[ii][jj][c] += g;
        }
      }
    }
    return gradIn;
  }

  function conv2dBackward(gradOut, input, filters) {
    const inH = input.length;
    const inW = input[0].length;
    const inC = input[0][0].length;
    const fSize = filters[0].length;
    const numFilters = filters.length;
    const outH = gradOut.length;
    const outW = gradOut[0].length;

    const gradInput = create3D(inH, inW, inC, 0);
    const gradFilters = [];
    const gradBiases = new Array(numFilters).fill(0);

    for (let f = 0; f < numFilters; f++) {
      const kernel = filters[f];
      const gradKernel = create3D(fSize, fSize, inC, 0);
      gradFilters.push(gradKernel);

      for (let i = 0; i < outH; i++) {
        for (let j = 0; j < outW; j++) {
          const g = gradOut[i][j][f];
          gradBiases[f] += g;
          for (let ki = 0; ki < fSize; ki++) {
            for (let kj = 0; kj < fSize; kj++) {
              for (let c = 0; c < inC; c++) {
                const inVal = input[i + ki][j + kj][c];
                gradKernel[ki][kj][c] += g * inVal;
                gradInput[i + ki][j + kj][c] += g * kernel[ki][kj][c];
              }
            }
          }
        }
      }
    }
    return { gradInput, gradFilters, gradBiases };
  }

  function flatten3D(tensor) {
    const h = tensor.length;
    const w = tensor[0].length;
    const d = tensor[0][0].length;
    const vec = new Array(h * w * d);
    let idx = 0;
    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++) {
        for (let k = 0; k < d; k++) {
          vec[idx++] = tensor[i][j][k];
        }
      }
    }
    return { vec, shape: { h, w, d } };
  }

  function unflatten(vec, shape) {
    const { h, w, d } = shape;
    const tensor = create3D(h, w, d, 0);
    let idx = 0;
    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++) {
        for (let k = 0; k < d; k++) {
          tensor[i][j][k] = vec[idx++];
        }
      }
    }
    return tensor;
  }

  function denseForward(vec, weights, bias) {
    const out = new Array(weights.length);
    for (let i = 0; i < weights.length; i++) {
      let sum = bias[i];
      for (let j = 0; j < vec.length; j++) {
        sum += weights[i][j] * vec[j];
      }
      out[i] = sum;
    }
    return out;
  }

  function softmax(logits) {
    // Softmax implemented with plain loops (no external libs; no helper utilities).
    let max = logits[0];
    for (let i = 1; i < logits.length; i++) {
      if (logits[i] > max) max = logits[i];
    }

    const exps = new Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
      const e = Math.exp(logits[i] - max);
      exps[i] = e;
      sum += e;
    }

    for (let i = 0; i < logits.length; i++) {
      exps[i] = exps[i] / sum;
    }
    return exps;
  }

  function crossEntropyLoss(probs, label) {
    const p = Math.max(probs[label], 1e-15);
    return -Math.log(p);
  }

  function oneHot(label, n) {
    const v = new Array(n).fill(0);
    v[label] = 1;
    return v;
  }

  /* ---------- CNN model ---------- */

  function computeFlatSize(numLayers, numFilters, filterSize) {
    let h = INPUT_SIZE;
    let w = INPUT_SIZE;
    let channels = 1;
    for (let l = 0; l < numLayers; l++) {
      h = convOutputSize(h, filterSize);
      w = convOutputSize(w, filterSize);
      channels = numFilters;
      h = poolOutputSize(h, POOL_SIZE);
      w = poolOutputSize(w, POOL_SIZE);
    }
    return h * w * channels;
  }

  function CNN(config) {
    this.numLayers = config.numLayers;
    this.numFilters = config.numFilters;
    this.filterSize = config.filterSize;
    this.learningRate = config.learningRate;
    this.flatSize = computeFlatSize(
      this.numLayers,
      this.numFilters,
      this.filterSize
    );
    this.initWeights();
  }

  CNN.prototype.initWeights = function () {
    this.convLayers = [];
    let inChannels = 1;
    for (let l = 0; l < this.numLayers; l++) {
      this.convLayers.push({
        filters: initFilters(this.numFilters, this.filterSize, inChannels),
        biases: initBiases(this.numFilters),
      });
      inChannels = this.numFilters;
    }
    const outDim = NUM_CLASSES;
    const inDim = this.flatSize;
    this.denseWeights = [];
    for (let i = 0; i < outDim; i++) {
      const row = new Array(inDim);
      const scale = Math.sqrt(2 / inDim);
      for (let j = 0; j < inDim; j++) {
        row[j] = randomNormal(0, scale);
      }
      this.denseWeights.push(row);
    }
    this.denseBias = initBiases(outDim);
  };

  CNN.prototype.forward = function (input2d) {
    const cache = { input: input2d, layers: [] };
    let x = input2d;

    for (let l = 0; l < this.numLayers; l++) {
      const layer = this.convLayers[l];
      const conv = conv2dForward(x, layer.filters, layer.biases);
      const relu = relu3D(conv);
      const pool = maxPool2dForward(relu, POOL_SIZE);
      cache.layers.push({
        input: x,
        conv: conv,
        relu: relu,
        pool: pool,
        filters: layer.filters,
      });
      x = pool.output;
    }

    const flat = flatten3D(x);
    const logits = denseForward(flat.vec, this.denseWeights, this.denseBias);
    const probs = softmax(logits);

    cache.flatVec = flat.vec;
    cache.flatShape = flat.shape;
    cache.logits = logits;
    cache.probs = probs;

    return { probs: probs, cache: cache };
  };

  CNN.prototype.backward = function (cache, label) {
    const lr = this.learningRate;
    const probs = cache.probs;
    const gradLogits = probs.slice();
    gradLogits[label] -= 1;

    const inDim = this.flatSize;
    const gradDenseW = [];
    for (let i = 0; i < NUM_CLASSES; i++) {
      gradDenseW[i] = new Array(inDim).fill(0);
    }
    const gradDenseB = new Array(NUM_CLASSES).fill(0);

    for (let i = 0; i < NUM_CLASSES; i++) {
      gradDenseB[i] = gradLogits[i];
      for (let j = 0; j < inDim; j++) {
        gradDenseW[i][j] = gradLogits[i] * cache.flatVec[j];
      }
    }

    let gradFlat = new Array(inDim).fill(0);
    for (let j = 0; j < inDim; j++) {
      let s = 0;
      for (let i = 0; i < NUM_CLASSES; i++) {
        s += this.denseWeights[i][j] * gradLogits[i];
      }
      gradFlat[j] = s;
    }

    let gradTensor = unflatten(gradFlat, cache.flatShape);

    for (let l = this.numLayers - 1; l >= 0; l--) {
      const layerCache = cache.layers[l];
      const poolInH = layerCache.relu.length;
      const poolInW = layerCache.relu[0].length;

      gradTensor = maxPool2dBackward(
        gradTensor,
        layerCache.pool.argmaxI,
        layerCache.pool.argmaxJ,
        poolInH,
        poolInW
      );
      gradTensor = reluBackward(gradTensor, layerCache.conv);

      const convGrad = conv2dBackward(
        gradTensor,
        layerCache.input,
        layerCache.filters
      );
      gradTensor = convGrad.gradInput;

      const layer = this.convLayers[l];
      for (let f = 0; f < this.numFilters; f++) {
        layer.biases[f] -= lr * convGrad.gradBiases[f];
        for (let ki = 0; ki < this.filterSize; ki++) {
          for (let kj = 0; kj < this.filterSize; kj++) {
            for (let c = 0; c < layer.filters[f][ki][kj].length; c++) {
              layer.filters[f][ki][kj][c] -=
                lr * convGrad.gradFilters[f][ki][kj][c];
            }
          }
        }
      }
    }

    for (let i = 0; i < NUM_CLASSES; i++) {
      this.denseBias[i] -= lr * gradDenseB[i];
      for (let j = 0; j < inDim; j++) {
        this.denseWeights[i][j] -= lr * gradDenseW[i][j];
      }
    }
  };

  CNN.prototype.trainSample = function (input2d, label) {
    const result = this.forward(input2d);
    const loss = crossEntropyLoss(result.probs, label);
    this.backward(result.cache, label);
    return { loss: loss, probs: result.probs };
  };

  CNN.prototype.predict = function (input2d) {
    const result = this.forward(input2d);
    let best = 0;
    for (let i = 1; i < NUM_CLASSES; i++) {
      if (result.probs[i] > result.probs[best]) best = i;
    }
    return { classIndex: best, probs: result.probs, name: CLASS_NAMES[best] };
  };

  CNN.prototype.toJSON = function () {
    return {
      numLayers: this.numLayers,
      numFilters: this.numFilters,
      filterSize: this.filterSize,
      learningRate: this.learningRate,
      flatSize: this.flatSize,
      convLayers: this.convLayers,
      denseWeights: this.denseWeights,
      denseBias: this.denseBias,
    };
  };

  CNN.fromJSON = function (data) {
    const cnn = Object.create(CNN.prototype);
    cnn.numLayers = data.numLayers;
    cnn.numFilters = data.numFilters;
    cnn.filterSize = data.filterSize;
    cnn.learningRate = data.learningRate;
    cnn.flatSize = data.flatSize;
    cnn.convLayers = data.convLayers;
    cnn.denseWeights = data.denseWeights;
    cnn.denseBias = data.denseBias;
    return cnn;
  };

  /* ---------- Synthetic training data (28x28) ---------- */

  function rasterToTensor(grid) {
    const tensor = create3D(INPUT_SIZE, INPUT_SIZE, 1, 0);
    for (let r = 0; r < INPUT_SIZE; r++) {
      for (let c = 0; c < INPUT_SIZE; c++) {
        tensor[r][c][0] = grid[r][c];
      }
    }
    return tensor;
  }

  function drawCircleGrid() {
    const g = create2D(INPUT_SIZE, INPUT_SIZE, 0);
    const cx = 14;
    const cy = 14;
    const rad = 9;
    for (let r = 0; r < INPUT_SIZE; r++) {
      for (let c = 0; c < INPUT_SIZE; c++) {
        const d = Math.sqrt((r - cx) * (r - cx) + (c - cy) * (c - cy));
        if (d <= rad && d >= rad - 2.5) g[r][c] = 1;
      }
    }
    return g;
  }

  function drawSquareGrid() {
    const g = create2D(INPUT_SIZE, INPUT_SIZE, 0);
    for (let r = 6; r < 22; r++) {
      for (let c = 6; c < 22; c++) {
        if (r === 6 || r === 21 || c === 6 || c === 21) g[r][c] = 1;
      }
    }
    return g;
  }

  function distToSegment(pr, pc, r1, c1, r2, c2) {
    const dr = r2 - r1;
    const dc = c2 - c1;
    const lenSq = dr * dr + dc * dc;
    if (lenSq === 0) {
      return Math.sqrt((pr - r1) * (pr - r1) + (pc - c1) * (pc - c1));
    }
    let t = ((pr - r1) * dr + (pc - c1) * dc) / lenSq;
    t = Math.max(0, Math.min(1, t));
    const projR = r1 + t * dr;
    const projC = c1 + t * dc;
    return Math.sqrt((pr - projR) * (pr - projR) + (pc - projC) * (pc - projC));
  }

  function drawTriangleGrid() {
    const g = create2D(INPUT_SIZE, INPUT_SIZE, 0);
    const ax = 14;
    const ay = 5;
    const bx = 5;
    const by = 22;
    const cx = 23;
    const cy = 22;
    for (let r = 0; r < INPUT_SIZE; r++) {
      for (let c = 0; c < INPUT_SIZE; c++) {
        const d1 = distToSegment(r, c, ax, ay, bx, by);
        const d2 = distToSegment(r, c, bx, by, cx, cy);
        const d3 = distToSegment(r, c, cx, cy, ax, ay);
        const minD = Math.min(d1, d2, d3);
        if (minD < 1.6) g[r][c] = 1;
      }
    }
    return g;
  }

  function generateSample(classIndex) {
    let grid;
    if (classIndex === 0) grid = drawCircleGrid();
    else if (classIndex === 1) grid = drawSquareGrid();
    else grid = drawTriangleGrid();

    const jitter = (Math.random() - 0.5) * 0.15;
    for (let r = 0; r < INPUT_SIZE; r++) {
      for (let c = 0; c < INPUT_SIZE; c++) {
        if (grid[r][c] > 0) {
          grid[r][c] = Math.min(1, Math.max(0, grid[r][c] + jitter));
        }
      }
    }
    return { tensor: rasterToTensor(grid), label: classIndex };
  }

  function augmentGrid(grid, shiftR, shiftC) {
    const out = create2D(INPUT_SIZE, INPUT_SIZE, 0);
    for (let r = 0; r < INPUT_SIZE; r++) {
      for (let c = 0; c < INPUT_SIZE; c++) {
        const nr = r + shiftR;
        const nc = c + shiftC;
        if (nr >= 0 && nr < INPUT_SIZE && nc >= 0 && nc < INPUT_SIZE) {
          out[nr][nc] = grid[r][c];
        }
      }
    }
    return out;
  }

  /* ---------- Canvas capture ---------- */

  function captureCanvasToGrid(canvas) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const imageData = ctx.getImageData(0, 0, w, h);
    const grid = create2D(INPUT_SIZE, INPUT_SIZE, 0);

    for (let r = 0; r < INPUT_SIZE; r++) {
      for (let c = 0; c < INPUT_SIZE; c++) {
        const sx = Math.floor((c / INPUT_SIZE) * w);
        const sy = Math.floor((r / INPUT_SIZE) * h);
        let sum = 0;
        let count = 0;
        const blockW = Math.max(1, Math.floor(w / INPUT_SIZE));
        const blockH = Math.max(1, Math.floor(h / INPUT_SIZE));
        for (let dy = 0; dy < blockH; dy++) {
          for (let dx = 0; dx < blockW; dx++) {
            const px = Math.min(w - 1, sx + dx);
            const py = Math.min(h - 1, sy + dy);
            const idx = (py * w + px) * 4;
            const gray =
              (imageData.data[idx] +
                imageData.data[idx + 1] +
                imageData.data[idx + 2]) /
              3;
            sum += 1 - gray / 255;
            count++;
          }
        }
        grid[r][c] = count ? sum / count : 0;
      }
    }
    return grid;
  }

  /* ---------- Persistence ---------- */

  function saveModel(model) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(model.toJSON()));
      return true;
    } catch (e) {
      return false;
    }
  }

  function loadModel() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return null;
      return CNN.fromJSON(JSON.parse(raw));
    } catch (e) {
      return null;
    }
  }

  /* ---------- UI wiring ---------- */

  let model = null;
  let isTraining = false;

  const canvas = document.getElementById('drawCanvas');
  const ctx = canvas ? canvas.getContext('2d') : null;
  const statusEl = document.getElementById('status');
  const lossEl = document.getElementById('lossValue');
  const accEl = document.getElementById('accValue');
  const predEl = document.getElementById('prediction');
  const probBars = document.getElementById('probBars');

  const elLayers = document.getElementById('numLayers');
  const elFilters = document.getElementById('numFilters');
  const elFilterSize = document.getElementById('filterSize');
  const elLR = document.getElementById('learningRate');
  const elEpochs = document.getElementById('epochs');

  function getConfig() {
    return {
      numLayers: parseInt(elLayers.value, 10),
      numFilters: parseInt(elFilters.value, 10),
      filterSize: parseInt(elFilterSize.value, 10),
      learningRate: parseFloat(elLR.value),
    };
  }

  function setStatus(msg) {
    if (statusEl) statusEl.textContent = msg;
  }

  function validateArchitecture(cfg) {
    let h = INPUT_SIZE;
    let w = INPUT_SIZE;
    for (let l = 0; l < cfg.numLayers; l++) {
      h = convOutputSize(h, cfg.filterSize);
      w = convOutputSize(w, cfg.filterSize);
      if (h < 1 || w < 1) return false;
      h = poolOutputSize(h, POOL_SIZE);
      w = poolOutputSize(w, POOL_SIZE);
      if (h < 1 || w < 1) return false;
    }
    return computeFlatSize(cfg.numLayers, cfg.numFilters, cfg.filterSize) > 0;
  }

  function buildModelFromUI() {
    const cfg = getConfig();
    if (!validateArchitecture(cfg)) {
      setStatus(
        'Architecture too deep/large for 28×28 input. Reduce layers or filter size.'
      );
      return null;
    }
    const epochs = parseInt(elEpochs.value, 10);
    model = new CNN(cfg);
    return { cfg: cfg, epochs: epochs };
  }

  function configMatchesSaved(m) {
    const cfg = getConfig();
    return (
      m.numLayers === cfg.numLayers &&
      m.numFilters === cfg.numFilters &&
      m.filterSize === cfg.filterSize
    );
  }

  function initCanvas() {
    if (!ctx) return;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#111';
    ctx.lineWidth = 14;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
  }

  let drawing = false;

  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: ((clientX - rect.left) / rect.width) * canvas.width,
      y: ((clientY - rect.top) / rect.height) * canvas.height,
    };
  }

  function startDraw(e) {
    e.preventDefault();
    drawing = true;
    const p = getPos(e);
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
  }

  function moveDraw(e) {
    if (!drawing) return;
    e.preventDefault();
    const p = getPos(e);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
  }

  function endDraw(e) {
    if (e) e.preventDefault();
    drawing = false;
  }

  function clearCanvas() {
    initCanvas();
  }

  function updateProbBars(probs) {
    if (!probBars) return;
    probBars.innerHTML = '';
    for (let i = 0; i < NUM_CLASSES; i++) {
      const row = document.createElement('div');
      row.className = 'prob-row';
      const label = document.createElement('span');
      label.textContent = CLASS_NAMES[i];
      const barWrap = document.createElement('div');
      barWrap.className = 'prob-bar-wrap';
      const bar = document.createElement('div');
      bar.className = 'prob-bar';
      bar.style.width = Math.round(probs[i] * 100) + '%';
      const pct = document.createElement('span');
      pct.className = 'prob-pct';
      pct.textContent = (probs[i] * 100).toFixed(1) + '%';
      barWrap.appendChild(bar);
      row.appendChild(label);
      row.appendChild(barWrap);
      row.appendChild(pct);
      probBars.appendChild(row);
    }
  }

  async function runTraining() {
    if (isTraining) return;
    const cfg = getConfig();
    if (!validateArchitecture(cfg)) {
      setStatus(
        'Architecture too deep/large for 28×28 input. Reduce layers or filter size.'
      );
      return;
    }
    const saved = model && configMatchesSaved(model);
    if (!saved) {
      if (!buildModelFromUI()) return;
    } else {
      model.learningRate = getConfig().learningRate;
    }

    const epochs = parseInt(elEpochs.value, 10);
    const samplesPerClass = 24;
    isTraining = true;
    setStatus('Training…');
    document.getElementById('btnTrain').disabled = true;

    let totalLoss = 0;
    let correct = 0;
    let total = 0;

    for (let ep = 0; ep < epochs; ep++) {
      totalLoss = 0;
      correct = 0;
      total = 0;

      for (let cls = 0; cls < NUM_CLASSES; cls++) {
        for (let s = 0; s < samplesPerClass; s++) {
          const sample = generateSample(cls);
          const shiftR = Math.floor(Math.random() * 5) - 2;
          const shiftC = Math.floor(Math.random() * 5) - 2;
          const grid2d = create2D(INPUT_SIZE, INPUT_SIZE, 0);
          for (let r = 0; r < INPUT_SIZE; r++) {
            for (let c = 0; c < INPUT_SIZE; c++) {
              grid2d[r][c] = sample.tensor[r][c][0];
            }
          }
          const shifted = augmentGrid(grid2d, shiftR, shiftC);
          const tensor = rasterToTensor(shifted);
          const out = model.trainSample(tensor, cls);
          totalLoss += out.loss;
          const pred = model.predict(tensor);
          if (pred.classIndex === cls) correct++;
          total++;
        }
      }

      const userGrid = captureCanvasToGrid(canvas);
      const userTensor = rasterToTensor(userGrid);
      const selected = parseInt(
        document.getElementById('trainClass').value,
        10
      );
      const userOut = model.trainSample(userTensor, selected);
      totalLoss += userOut.loss;
      const userPred = model.predict(userTensor);
      if (userPred.classIndex === selected) correct++;
      total++;

      if (lossEl) lossEl.textContent = (totalLoss / total).toFixed(4);
      if (accEl) accEl.textContent = ((100 * correct) / total).toFixed(1) + '%';
      setStatus('Epoch ' + (ep + 1) + ' / ' + epochs);

      await new Promise(function (resolve) {
        setTimeout(resolve, 0);
      });
    }

    saveModel(model);
    setStatus('Training complete. Weights saved to localStorage.');
    document.getElementById('btnTrain').disabled = false;
    isTraining = false;

    const grid = captureCanvasToGrid(canvas);
    const pred = model.predict(rasterToTensor(grid));
    updateProbBars(pred.probs);
    if (predEl) {
      predEl.textContent = pred.name + ' (' + (pred.probs[pred.classIndex] * 100).toFixed(1) + '%)';
    }
  }

  function runPredict() {
    if (!model) {
      setStatus('Train first or load saved weights.');
      return;
    }
    const grid = captureCanvasToGrid(canvas);
    const pred = model.predict(rasterToTensor(grid));
    updateProbBars(pred.probs);
    if (predEl) {
      predEl.textContent =
        pred.name + ' (' + (pred.probs[pred.classIndex] * 100).toFixed(1) + '%)';
    }
    setStatus('Prediction: ' + pred.name);
  }

  function runReset() {
    localStorage.removeItem(STORAGE_KEY);
    buildModelFromUI();
    if (lossEl) lossEl.textContent = '—';
    if (accEl) accEl.textContent = '—';
    if (predEl) predEl.textContent = '—';
    if (probBars) probBars.innerHTML = '';
    clearCanvas();
    setStatus('Model reset. Draw a shape and train.');
  }

  function bindControls() {
    document.getElementById('btnTrain').addEventListener('click', runTraining);
    document.getElementById('btnPredict').addEventListener('click', runPredict);
    document.getElementById('btnReset').addEventListener('click', runReset);
    document.getElementById('btnClear').addEventListener('click', clearCanvas);

    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mousemove', moveDraw);
    canvas.addEventListener('mouseup', endDraw);
    canvas.addEventListener('mouseleave', endDraw);
    canvas.addEventListener('touchstart', startDraw, { passive: false });
    canvas.addEventListener('touchmove', moveDraw, { passive: false });
    canvas.addEventListener('touchend', endDraw, { passive: false });
  }

  function startup() {
    initCanvas();
    bindControls();

    const loaded = loadModel();
    if (loaded) {
      model = loaded;
      model.learningRate = getConfig().learningRate;
      elLayers.value = String(model.numLayers);
      elFilters.value = String(model.numFilters);
      elFilterSize.value = String(model.filterSize);
      setStatus('Loaded saved weights from localStorage.');
    } else {
      model = null;
      setStatus('No saved model. Adjust hyperparameters and Train.');
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', startup);
  } else {
    startup();
  }
})();
