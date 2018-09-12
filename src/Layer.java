public class Layer {

  /* Number of inputs and outputs (neurons).*/
  int numInputs, numOutputs;

  /* Output of EACH neuron. */
  double[] outputs;

  /* Pointers to previous and next layer if any. */
  Layer prev;
  Layer next;

  /* Incoming weights of EACH neuron. */
  double[][] weights;

  /* Biases of EACH neuron. */
  double[] biases;

  /* Delta errors of EACH neuron. */
  double[] deltas;

  public Layer() {
    numInputs = 0;
  }

  public boolean initializeLayer(int numOutputs, Layer prev) {
    if (prev == null) {
      numInputs = 0;
    }
    else {
      int arrSize = prev.numOutputs;
      double[] emptyArr = new double[arrSize];

      this.numInputs = prev.numOutputs;
      this.prev = prev;
      this.numOutputs = numOutputs;
      this.outputs = emptyArr;
      this.deltas = new double[arrSize];
      this.biases = new double[arrSize];

      int height = prev.weights.length;
      double[][] weights = new double[height][];

      for (int i = 0; i < height; i++) {
        int width = weights[i].length;
        weights[i] = new double[width];

        for (int j = 0; j < width; j++) {
          weights[i][j] = LayerUtils.annRandom();
        }
      }
      this.weights = weights;
    }
  return false;
  }

  /* Computes the outputs of the current and all subsequent layers given inputs. */
  public void computeOutputs() {
    Layer prev = this.prev;
    if (prev == null) {return;} /* do nothing as layer is an input layer */

    else {
      int currNumOutputs = this.numOutputs;
      int prevNumOutputs = this.numOutputs;

      for (int j = 0; j < currNumOutputs; j++) {
        double partialSum = 0;

        for (int i = 0; i < prevNumOutputs; i++) {
          partialSum += this.weights[i][j] * prev.outputs[i];
        }

        this.outputs[j] = LayerUtils.sigmoid(this.biases[j] + partialSum);
      }
    }
  }

  /* Computes the delta errors for this layer and all previous layers (backpropagate). */
  public void computeDeltas() {
    Layer next = this.next;

    if (next == null) {return;} /* do nothing as layer is the final layer */

    else {
      int currNumDeltas = this.deltas.length;
      int nextNumDeltas = this.deltas.length;

      for (int i = 0; i < currNumDeltas; i++) {
        double partialSum = 0;

        for (int j = 0; j < nextNumDeltas; j++) {
          partialSum += next.weights[i][j] * next.deltas[j];
        }

        this.deltas[i] = LayerUtils.sigmoidPrime(LayerUtils.sigmoid(this.outputs[i]))
                          * partialSum;
      }
    }
  }

  /* Updates weights and biases according to the delta errors given learning rate. */
  public void updateLayer(double learnRate) {
    int height = this.weights.length;

    for (int i = 0; i < height; i++) {
      int width = this.weights[i].length;

      for (int j = 0; j < width; j++) {
        double weight = this.weights[i][j];
        this.weights[i][j] = weight + (learnRate * this.outputs[i] * this.deltas[j]);
      }
    }

    int bound = this.biases.length;

    for (int j = 0; j < bound; j++) {
      double bias = this.biases[j];
      this.biases[j] = bias + (learnRate * this.deltas[j]);
    }
  }

}
