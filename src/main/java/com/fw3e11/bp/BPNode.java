package com.fw3e11.bp;

public class BPNode {

  private Type type;

  private double forwardInput;
  private double forwardOutput;

  private double backwardInput;
  private double backwardOutput;

  public BPNode(Type type) {
    this.type = type;
  }

  public enum Type {
    INPUT,
    HIDDEN,
    OUTPUT,
  }

  public void setForwardInput(double forwardInput) {
    this.forwardInput = forwardInput;
    switch (type) {
      case INPUT:
        this.forwardOutput = forwardInput;
      case HIDDEN:
      case OUTPUT:
        this.forwardOutput = Activation.sigmoid(forwardInput);
    }
  }

  public void setBackwardInput(double backwardInput) {
    this.backwardInput = backwardInput;
    switch (type) {
      case INPUT:
        this.backwardOutput = backwardInput;
      case HIDDEN:
      case OUTPUT:
        this.backwardOutput = Activation.sigmoidDerivative(backwardInput);
    }
  }

  public double getForwardOutput() {
    return forwardOutput;
  }

  public double getBackwardOutput() {
    return backwardOutput;
  }
}
