let model;
let activationModel;
let imageTensor;


async function drawActivation(activations, layerIndex, filterNumber, plotId, size) {
  const activationToDraw = tf.tidy(() => {
    // TODO: Choose the filter
    const b = activations[layerIndex].slice([0, 0, 0, filterNumber], [1, size, size, 1]);
    // TypedArray to Array and reserve it on the axis #1.
    const newB = Array.from(b.reverse(1).dataSync());

    const newArr = [];
    while (newB.length) newArr.push(newB.splice(0, size));
    return newArr;
  });

  console.log(activationToDraw);
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

async function init() {
  initSliders('one', (value) => {
    console.log(value);
    const prediction = model.predict(imageTensor).dataSync();
    const activations = activationModel.predict(imageTensor);

    // model.layers[0].batchInputShape[1]
    drawActivation(activations, 0, parseInt(value, 10), 'plot-activations-one', 150);
  });


  model = await tf.loadLayersModel('http://127.0.0.1:8080/model/tfjs-2/model.json');
  model.summary();
  const outputLayers = [];

  model.layers.slice(0, 6).forEach((layer, i) => {
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
    };

    output.onload = () => {
      imageTensor = tf.browser.fromPixels(output)
        .expandDims()
        .toFloat()
        .div(255.0);

      const surface = { name: 'Values Distribution', tab: 'Model Inspection' };
      // tfvis.show.valuesDistribution(surface, imageTensor);

      const prediction = model.predict(imageTensor).dataSync();
      const activations = activationModel.predict(imageTensor);

      // model.layers[0].batchInputShape[1]
      drawActivation(activations, 0, 0, 'plot-activations-one', 150);
      drawActivation(activations, 1, 0, 'plot-activations-two', 75);
      drawActivation(activations, 2, 0, 'plot-activations-three', 75);
      drawActivation(activations, 3, 0, 'plot-activations-four', 37);
      drawActivation(activations, 4, 0, 'plot-activations-five', 37);
      drawActivation(activations, 5, 0, 'plot-activations-six', 18);

      document.getElementById('p-prediction').innerHTML = prediction;
    };

    reader.readAsDataURL(input.files[0]);
  };
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
