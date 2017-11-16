package com.fw3e11.bp;

public final class Activation {

  private Activation() {
  }

  public static double sigmoid(double x) {
    return 1 / (1 + Math.exp(-x));
  }

  // y' = y * (1-y)
  public static double sigmoidDerivative(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
  }
}
