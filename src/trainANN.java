import java.util.Random;

public class trainANN {

  /* Creates and trains a simple ann for XOR. */
  public static void main(String[] args) {
    System.out.println("Big data machine learning.\n");
    System.out.println("--------------------------");

    /* Intializes random number generator */
    Random rand = new Random(42);

    /* Here is some BIG DATA to train, XOR function. */
    double[][] inputs = new double[][]{{0, 0},
                                       {0, 1},
                                       {1, 0},
                                       {1, 1}};
    double[] targets = {0, 1, 1, 0};

    System.out.println("PART I - Creating a layer.\n");
    System.out.println("Trying to construct first layer.");
    Layer firstLayer = new Layer();

    System.out.println("Running intializeLayer.");
    if (firstLayer.initializeLayer(2, null)) {
      System.out.println("Couldn't initialize first layer...");
      System.exit(-1);
    }

    System.out.println("Here are some of the properties:");
    System.out.print("  num_outputs: " + firstLayer.numOutputs);
    System.out.print("   num_inputs: " + firstLayer.numInputs);
    System.out.print("   outputs[0]: " + firstLayer.outputs[0]);
    System.out.print("   outputs[1]: " + firstLayer.outputs[1]);

    System.out.println("Creating second layer.\n");
    Layer secondLayer = new Layer();

    System.out.println("Running layer_init on second layer");
    if (secondLayer.initializeLayer(1, firstLayer)) {
      System.out.println("Couldn't layer_init second layer...");
      System.exit(-1);
    }

    System.out.print("Here are some of the properties:\n");
    System.out.print("  num_outputs: " + secondLayer.numOutputs);
    System.out.print("   num_inputs: " + secondLayer.numInputs);
    System.out.print("   weights[0]: " + secondLayer.weights[0][0]);
    System.out.print("   weights[1]: " + secondLayer.weights[1][0]);
    System.out.print("    biases[0]: " + secondLayer.biases[0]);
    System.out.print("   outputs[0]: " + secondLayer.outputs[0]);

    System.out.print("\nComputing second layer outputs:\n");
    secondLayer.computeOutputs();
    System.out.print("Here is the new output:\n");
    System.out.print("   outputs[0]: " + secondLayer.outputs[0]);

    /* Create neural network. */
    System.out.print("\n--------------------------\n");
    System.out.print("PART II - Creating a neural network.\n");
    System.out.print("2 inputs, 2 hidden neurons and 1 output.\n\n");
    System.out.print(" * - * \\ \n");
    System.out.print("         * - \n");
    System.out.print(" * - * / \n\n");

    int layerOutputs[] = {2, 2, 1};
    ANN xorAnn = new ANN();
    xorAnn.annInitialize(3, layerOutputs);

    /* Initialise weights to random. */
    System.out.print("Initialising network with random weights...\n");

    /* Print hidden layer weights, biases and outputs. */
    System.out.print("The current state of the hidden layer:\n");
    for(int i=0; i < layerOutputs[0]; ++i) {
      for(int j=0; j < layerOutputs[1]; ++j)
        System.out.print("  weights[" + i + "][" + j + "]: " +
            xorAnn.inputLayer.next.weights[i][j] + "\n");
    }

    for(int i=0; i < layerOutputs[1]; ++i)
      System.out.print("  biases[" + i + "]: " + xorAnn.inputLayer.next.biases[i] + "\n");
    for(int i=0; i < layerOutputs[1]; ++i)
      System.out.print("  outputs[" + i + "]: " + xorAnn.inputLayer.next.outputs[i] + "\n");

    /* Dummy run to see random network output. */
    System.out.print("Current random outputs of the network:\n");
    for(int i = 0; i < 4; ++i) {
      xorAnn.annPredict(inputs[i]);
      System.out.print("  [" + inputs[i][0] + ", " + inputs[i][1] + "] -> " +
          xorAnn.outputLayer.outputs[0] + "\n");
    }

    /* Train the network. */
    System.out.print("\nTraining the network...\n");
    for(int i = 0; i < 25000; ++i) {
    /* This is an epoch, running through the entire data. */
      for(int j = 0; j < 4; ++j) {
      /* Training at batch size 1, ie updating weights after every data point. */
        xorAnn.annTrain(inputs[j], targets, 1.0);
      }
    }

    /* Print hidden layer weights, biases and outputs. */
    System.out.print("The current state of the hidden layer:\n");
    for(int i=0; i < layerOutputs[0]; ++i) {
      for(int j=0; j < layerOutputs[1]; ++j)
        System.out.print("  weights[" + i + "][" + j + "]: " +
            xorAnn.inputLayer.next.weights[i][j] + "\n");
    }

    for(int i=0; i < layerOutputs[1]; ++i)
      System.out.print("  biases[" + i + "]: " + xorAnn.inputLayer.next.biases[i] + "\n");
    for(int i=0; i < layerOutputs[1]; ++i)
      System.out.print("  outputs[" + i + "]: " + xorAnn.inputLayer.next.outputs[i] + "\n");

    /* Let's see the results. */
    System.out.print("\nAfter training magic happened the outputs are:\n");
    for(int i = 0; i < 4; ++i) {
      xorAnn.annPredict(inputs[i]);
      System.out.print("  [" + inputs[i][0] + ", " + inputs[i][1] + "] -> " +
          xorAnn.outputLayer.outputs[0] + "\n");
    }

    System.exit(1);
  }

}
