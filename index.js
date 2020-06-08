// The image classifier
let model;

// The image classifier used for creating the activation maps
let activationModel;

// The object detection model
let odModel;
let imageTensor;
let sourceImage;
let ctx;

const RADIX = 10;
const BBCOLOR = '#008000';
const ODSIZE = 500;


async function drawActivation(activations, layerIndex, filterNumber, plotId, size) {
  // Use tf.tidy to remove garbage collect the intermediate tensors.
  const activationToDraw = tf.tidy(() => {
    // Get the activation map of the given layer (layerIndex) and filter (filterNumber)
    const activation = activations[layerIndex].slice([0, 0, 0, filterNumber], [1, size, size, 1]);
    // TypedArray to Array and reserve it on the axis #1.
    const activationArray = Array.from(activation.reverse(1).dataSync());

    const reshapedActivation = [];
    // Reshape array to 2D
    while (activationArray.length) reshapedActivation.push(activationArray.splice(0, size));
    return reshapedActivation;
  });

  const data = [
    {
      z: activationToDraw,
      type: 'heatmap',
    },
  ];

  const layout = {
    autosize: false,
    width: 500,
    height: 500,
  };

  Plotly.newPlot(plotId, data, layout);
}


function setODCanvas() {
  const canvas = document.getElementById('canvas');
  ctx = canvas.getContext('2d');
  sourceImage = document.getElementById('source');

  canvas.width = ODSIZE;
  canvas.height = ODSIZE;
}

const layersInformation = [
  {
    layer: 'one',
    index: 0,
    size: 150,
  },
  {
    layer: 'two',
    index: 1,
    size: 75,
  },
  {
    layer: 'three',
    index: 2,
    size: 75,
  },
  {
    layer: 'four',
    index: 3,
    size: 37,
  },
  {
    layer: 'five',
    index: 4,
    size: 37,
  },
  {
    layer: 'six',
    index: 5,
    size: 18,
  },
];


async function init() {
  setODCanvas();

  initSliders('one', (value) => {
    const activations = activationModel.predict(imageTensor);
    drawActivation(activations, 0, parseInt(value, RADIX), 'plot-activations-one', 150);
  });

  initSliders('two', (value) => {
    const activations = activationModel.predict(imageTensor);
    drawActivation(activations, 1, parseInt(value, RADIX), 'plot-activations-two', 75);
  });

  initSliders('three', (value) => {
    const activations = activationModel.predict(imageTensor);
    drawActivation(activations, 2, parseInt(value, RADIX), 'plot-activations-three', 75);
  });

  initSliders('four', (value) => {
    const activations = activationModel.predict(imageTensor);
    drawActivation(activations, 3, parseInt(value, RADIX), 'plot-activations-four', 37);
  });

  initSliders('five', (value) => {
    const activations = activationModel.predict(imageTensor);
    drawActivation(activations, 4, parseInt(value, RADIX), 'plot-activations-five', 37);
  });

  initSliders('six', (value) => {
    const activations = activationModel.predict(imageTensor);
    drawActivation(activations, 5, parseInt(value, RADIX), 'plot-activations-six', 18);
  });


  model = await tf.loadLayersModel('http://127.0.0.1:8080/model/tfjs-2/model.json');
  model.summary();
  odModel = await tf.automl.loadObjectDetection('model/object-detection/model.json');
  console.log(odModel);
  const outputLayers = [];

  model.layers.slice(0, 6).forEach((layer) => {
    outputLayers.push(layer.output);
  });

  console.log();
  activationModel = tf.model({ inputs: model.inputs, outputs: outputLayers });
  activationModel.summary();
  const inputImage = document.getElementById('input-image');

  inputImage.onchange = async (file) => {
    const input = file.target;
    const reader = new FileReader();
    const output = document.getElementById('output');

    reader.onload = () => {
      const dataURL = reader.result;
      output.src = dataURL;
      sourceImage.src = dataURL;
    };

    output.onload = async () => {
      imageTensor = tf.browser.fromPixels(output)
        .expandDims()
        .toFloat()
        .div(255.0);

      const surface = { name: 'Values Distribution', tab: 'Model Inspection' };
      // tfvis.show.valuesDistribution(surface, imageTensor);

      const prediction = model.predict(imageTensor).dataSync();
      const activations = activationModel.predict(imageTensor);
      const options = { score: 0.95, iou: 0.95, topk: 5 };
      const obPredictions = await odModel.detect(sourceImage, options);
      console.log(obPredictions);

      ctx.save();
      // ctx.scale(-1, 1);
      // ctx.translate(-150, 0);
      ctx.drawImage(output, 0, 0, ODSIZE, ODSIZE);
      ctx.restore();

      obPredictions.forEach((obPrediction) => {
        drawBoundingBoxes(obPrediction);
      });

      // model.layers[0].batchInputShape[1]
      drawActivation(activations, 0, 0, 'plot-activations-one', 150);
      drawActivation(activations, 1, 0, 'plot-activations-two', 75);
      drawActivation(activations, 2, 0, 'plot-activations-three', 75);
      drawActivation(activations, 3, 0, 'plot-activations-four', 37);
      drawActivation(activations, 4, 0, 'plot-activations-five', 37);
      drawActivation(activations, 5, 0, 'plot-activations-six', 18);

      document.getElementById('p-prediction').innerHTML = prediction;
      // argMax returns a tensor with the arg max index. So,
      // we need dataSync to convert it to array and indexing to
      // get the value.
      const label = tf.argMax(prediction).dataSync()[0];
      console.log(label);
      if (label === 0) {
        console.log('hey its James!');
      }
      document.getElementById('p-label').innerHTML = ((label === 0) ? 'James' : 'Jessie');
    };

    reader.readAsDataURL(input.files[0]);
  };
}

function drawBoundingBoxes(prediction) {
  ctx.font = '20px Arial';
  const {
    left, top, width, height,
  } = prediction.box;

  // Draw the box
  ctx.strokeStyle = BBCOLOR;
  ctx.lineWidth = 1;
  ctx.strokeRect(left, top, width, height);

  // Draw the label background
  ctx.fillStyle = BBCOLOR;
  const textWidth = ctx.measureText(prediction.label).width;
  const textHeight = parseInt(ctx.font, 10);

  // Top left rectangle
  ctx.fillRect(left, top, textWidth + textHeight, textHeight * 2);
  // Bottom left rectangle
  ctx.fillRect(left, top + height - textHeight * 2, textWidth + textHeight, textHeight * 2);

  // Draw labels and score
  ctx.fillStyle = '#000000';
  ctx.fillText(prediction.label, left, top + textHeight);
  ctx.fillText(prediction.score.toFixed(2), left, top + height - textHeight);
}

function initSliders(layerNumber, cb) {
  const slider = document.getElementById(`activations-layer-${layerNumber}-range`);
  const output = document.getElementById(`activations-layer-${layerNumber}-value`);
  output.innerHTML = slider.value;

  slider.oninput = function onInputCb() {
    output.innerHTML = this.value;
    cb(this.value);
  };
}

init();
