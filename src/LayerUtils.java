import static java.lang.Math.exp;

import java.util.Random;

public class LayerUtils {

  static double annRandom() {
    Random rand = new Random();
    return rand.nextDouble() - 0.5;
  }

  /* The sigmoid function and derivative. */
  static double sigmoid(double x) {
    return 1 / (1 + exp(-x));
  }

  static double sigmoidPrime(double x) {
    return x * (1 - x);
  }

}
