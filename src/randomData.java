import java.util.Random;

public class randomData {

  /* Creates and displays random data for training. */
  public static void main(String[] args) {
    /* Check argument count. */
    if (args.length != 3) {
      System.out.println("Usage, randomData rows columns");
      System.exit(-1);
    }

    /* Intializes random number generator */
    Random rand = new Random(System.currentTimeMillis());

    /* Extract arguments. */
    int inputRows = Integer.parseInt(args[1]);
    int inputCols = Integer.parseInt(args[2]);

    /* Create dynamic arrays. */
    System.out.println("Creating (" + inputRows + ", " + inputCols + ") -> " + inputRows + ".");
    double[][] inputs = new double[inputRows][];
    double[] targets = new double[inputRows];

    /* Initialise with random numbers. */
      for(int i = 0; i < inputRows; ++i) {
        inputs[i] = new double[inputCols];

        for(int j = 0; j < inputCols; ++j) {
          inputs[i][j] = rand.nextDouble();
        }
        targets[i] = rand.nextDouble();
      }

    /* Display the random data. */
      for(int i = 0; i < inputRows; ++i) {
        for(int j = 0; j < inputCols; ++j) {
          System.out.print(" " + inputs[i][j] + " ");
        }
        System.out.println("-> " + targets[i]);
      }

    System.exit(1);
  }


}
