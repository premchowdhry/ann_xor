public class ANN {

  /* Represents a N layer artificial neural network. */

  /* The head and tail of layers doubly linked list. */
  Layer inputLayer;
  Layer outputLayer;

  public ANN() {
    inputLayer = new Layer();
    outputLayer = new Layer();
  }

  /* Creates and returns a new ann. */
  void annInitialize(int numLayers, int[] layerOutputs) {
    assert (layerOutputs != null);

    Layer iter = new Layer();

    if (!iter.initializeLayer(layerOutputs[0], null)) {
      this.inputLayer = iter;
    }

    for (int i = 1; i < numLayers; i++) {
      Layer prev = iter;
      Layer layer = new Layer();

      if (layer.initializeLayer(layerOutputs[i], iter)) {
        return;
      }
      iter.next = layer;
      iter = layer;
      iter.prev = prev;
    }

    this.outputLayer= iter;
  }

  /* Forward run of given ann with inputs. */
  void annPredict(double[] inputs) {
    assert(inputs != null);

    this.inputLayer.outputs = inputs;
    Layer iter = this.inputLayer.next;
    while (iter != null) {
      iter.computeOutputs();
      iter = iter.next;
    }
  }

  /* Trains the ann with single backprop update. */
  void annTrain(double[] inputs, double[] targets, double learnRate) {
    /* Sanity checks. */
    assert(inputs != null);
    assert(targets != null);
    assert(learnRate > 0);

    /* Run forward pass. */
    annPredict(inputs);

    /* Compute output layer deltas */
    int numOutputDeltas = this.outputLayer.deltas.length;

    for (int j = 0; j < numOutputDeltas; j++) {
      double output = this.outputLayer.outputs[j];
      this.outputLayer.deltas[j] = LayerUtils.sigmoidPrime(output) * (targets[j] - output);
    }

    /* Compute hidden layer deltas */
    Layer deltaIter = this.inputLayer.next;

    while (deltaIter != this.outputLayer) {
      deltaIter.computeDeltas();
      deltaIter = deltaIter.next;
    }

    /* Update all weights except the input layer */
    Layer weightIter = this.inputLayer.next;

    while (weightIter != null) {
      weightIter.updateLayer(learnRate);
      weightIter = weightIter.next;
    }
  }


}
