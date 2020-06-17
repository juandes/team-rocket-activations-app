// The image classifier
let classifierModel;

// The image classifier used for creating the activation maps
let activationModel;

// The object detection model
let odModel;
const odModelOptions = {
  score: 0.80,
  iou: 0.80,
  topk: 5,
};


let imageTensor;
let ctx;

const BBCOLOR = '#008000';
const ODSIZE = 300;

const layersInformation = [
  {
    layer: 'one',
    size: 150,
  },
  {
    layer: 'two',
    size: 75,
  },
  {
    layer: 'three',
    size: 75,
  },
  {
    layer: 'four',
    size: 37,
  },
  {
    layer: 'five',
    size: 37,
  },
  {
    layer: 'six',
    size: 18,
  },
];


function setupODCanvas() {
  const canvas = document.getElementById('output-detection-canvas');
  ctx = canvas.getContext('2d');
  canvas.width = ODSIZE;
  canvas.height = ODSIZE;
}

function initSliders(layerNumber, onInputCb) {
  const slider = document.getElementById(`activations-layer-${layerNumber}-range`);
  const output = document.getElementById(`activations-layer-${layerNumber}-value`);
  console.log(slider);
  output.innerHTML = slider.value;

  slider.oninput = function getSliderValue() {
    output.innerHTML = this.value;
    onInputCb(this.value);
  };
}

async function drawActivation(activations, layerIndex, filterNumber, plotId, size) {
  // Use tf.tidy to remove garbage collect the intermediate tensors.
  const activationToDraw = tf.tidy(() => {
    // Get the activation map of the given layer (layerIndex) and filter (filterNumber).
    const activation = activations[layerIndex].slice([0, 0, 0, filterNumber], [1, size, size, 1]);

    // TypedArray to Array and reverse it on the axis #1.
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


function setupSliders() {
  layersInformation.forEach((layerInfo, i) => {
    initSliders(layerInfo.layer, (filterNumber) => {
      const activations = activationModel.predict(imageTensor);
      drawActivation(activations, i, parseInt(filterNumber), `plot-activations-${layerInfo.layer}`, layerInfo.size);
    });
  });
}

async function setupModels() {
  classifierModel = await tf.loadLayersModel('http://127.0.0.1:8080/models/tfjs-3/model.json');
  odModel = await tf.automl.loadObjectDetection('models/object-detection/model.json');

  const outputLayers = [];

  // Iterate over first six layers of the image classification model
  // and push their output to outputLayers
  classifierModel.layers.slice(0, 6).forEach((layer) => {
    outputLayers.push(layer.output);
  });


  activationModel = tf.model({ inputs: classifierModel.inputs, outputs: outputLayers });
  activationModel.summary();
}

function drawBoundingBoxes(prediction) {
  ctx.font = '20px Arial';
  const {
    left, top, width, height,
  } = prediction.box;

  // Draw the box.
  ctx.strokeStyle = BBCOLOR;
  ctx.lineWidth = 1;
  ctx.strokeRect(left, top, width, height);

  // Draw the label background.
  ctx.fillStyle = BBCOLOR;
  const textWidth = ctx.measureText(prediction.label).width;
  const textHeight = parseInt(ctx.font);

  // Top left rectangle.
  ctx.fillRect(left, top, textWidth + textHeight, textHeight * 2);
  // Bottom left rectangle.
  ctx.fillRect(left, top + height - textHeight * 2, textWidth + textHeight, textHeight * 2);

  // Draw labels and score.
  ctx.fillStyle = '#000000';
  ctx.fillText(prediction.label, left + 10, top + textHeight);
  ctx.fillText(prediction.score.toFixed(2), left + 10, top + height - textHeight);
}


async function predictWithObjectDetector(outputDetections, outputImage) {
  // Predict with the object detector.
  const obPredictions = await odModel.detect(outputDetections, odModelOptions);

  ctx.drawImage(outputImage, 0, 0, ODSIZE, ODSIZE);
  obPredictions.forEach((obPrediction) => {
    drawBoundingBoxes(obPrediction);
  });
}

function predictWithClassifier(image) {
  // Convert the image to a tensor.
  imageTensor = tf.browser.fromPixels(image)
    .expandDims()
    .toFloat()
    .div(255.0);

  const prediction = classifierModel.predict(imageTensor).dataSync();
  const activations = activationModel.predict(imageTensor);

  // Draw the activation maps of the first filter.
  layersInformation.forEach((layerInfo, i) => {
    drawActivation(activations, i, 0, `plot-activations-${layerInfo.layer}`, layerInfo.size);
  });

  document.getElementById('p-prediction').innerHTML = prediction;
  // argMax returns a tensor with the arg max index. So,
  // we need dataSync to convert it to array and indexing to
  // get the value.
  const label = tf.argMax(prediction).dataSync()[0];

  document.getElementById('p-label').innerHTML = ((label === 0) ? 'James' : 'Jessie');
}


function processInput() {
  const inputImage = document.getElementById('input-image');
  const outputDetections = document.getElementById('output-detections');

  // Fired when the user selects an image.
  inputImage.onchange = async (file) => {
    const input = file.target;
    const reader = new FileReader();
    const outputImage = document.getElementById('output-image');

    // Fired when the selected image is loaded.
    reader.onload = () => {
      const dataURL = reader.result;

      // Set the the image to the output img element
      outputImage.src = dataURL;
      // Set the the image to the canvas element
      outputDetections.src = dataURL;
    };

    // Fired when the image is loaded to the HTML.
    outputImage.onload = async () => {
      predictWithObjectDetector(outputDetections, outputImage);
      predictWithClassifier(outputImage);
    };

    reader.readAsDataURL(input.files[0]);
  };
}


async function init() {
  setupODCanvas();
  setupSliders();
  await setupModels();
  processInput();
}


init();
