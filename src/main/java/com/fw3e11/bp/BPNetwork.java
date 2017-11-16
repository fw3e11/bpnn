package com.fw3e11.bp;

import com.fw3e11.bp.BPNode.Type;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class BPNetwork {

  private double alpha = 0.01;

  private int inputCount;
  private int hiddenCount;
  private int outputCount;

  private List<BPNode> inputNodes;
  private List<BPNode> hiddenNodes;
  private List<BPNode> outputNodes;

  private double[][] inputHiddenWeights;
  private double[][] hiddenOutputWeights;

  public BPNetwork(int inputCount, int hiddenCount, int outputCount) {
    this.inputCount = inputCount;
    this.hiddenCount = hiddenCount;
    this.outputCount = outputCount;

    inputNodes = initNodes(inputCount, Type.INPUT);
    hiddenNodes = initNodes(hiddenCount, Type.HIDDEN);
    outputNodes = initNodes(outputCount, Type.OUTPUT);

    inputHiddenWeights = initWeights(inputCount, hiddenCount);
    hiddenOutputWeights = initWeights(hiddenCount, outputCount);
  }

  public void setAlpha(double alpha) {
    this.alpha = alpha;
  }

  public void predict(BPData data) {
    forward(data.getDataXs());
    System.out.println("the fact is:      " + data.getDataYs());
    System.out.println("what we predict:  " + outputNodes.stream()
        .map(BPNode::getForwardOutput).collect(Collectors.toList()));
  }

  public void train(List<BPData> dataList, int n) {
    int dataSize = dataList.size();
    for (int i = 0; i < n; i++) {
      BPData data = dataList.get(i % dataSize);
      forward(data.getDataXs());
      backward(data.getDataYs());
      updateWeights();
    }
  }

  private List<BPNode> initNodes(int length, BPNode.Type type) {
    List<BPNode> nodeList = new ArrayList<>(length);
    for (int i = 0; i < length; i++) {
      nodeList.add(new BPNode(type));
    }
    return nodeList;
  }

  private double[][] initWeights(int x, int y) {
    return new double[x][y];
  }

  private void forward(List<Double> inputXs) {
    if (inputXs.size() != inputCount) {
      throw new IllegalArgumentException();
    }
    // Type.INPUT
    for (int i = 0; i < inputCount; i++) {
      inputNodes.get(i).setForwardInput(inputXs.get(i));
    }

    // Type.HIDDEN
    doForward(inputNodes, hiddenNodes, inputHiddenWeights);

    // Type.OUTPUT
    doForward(hiddenNodes, outputNodes, hiddenOutputWeights);
  }

  private void doForward(List<BPNode> l1, List<BPNode> l2, double[][] weights) {
    for (int j = 0; j < l2.size(); j++) {
      double sigma = 0;
      for (int i = 0; i < l1.size(); i++) {
        sigma += weights[i][j] * l1.get(i).getForwardOutput();
      }
      l2.get(j).setForwardInput(sigma);
    }
  }

  private void backward(List<Double> inputYs) {
    if (inputYs.size() != outputCount) {
      throw new IllegalArgumentException();
    }
    // Type.OUTPUT
    for (int i = 0; i < outputCount; i++) {
      BPNode outputNode = outputNodes.get(i);
      outputNode.setBackwardInput(outputNode.getForwardOutput() - inputYs.get(i));
    }

    // Type.HIDDEN
    doBackward(hiddenNodes, outputNodes, hiddenOutputWeights);

    // Type.INPUT
    doBackward(inputNodes, hiddenNodes, inputHiddenWeights);
  }

  private void doBackward(List<BPNode> l1, List<BPNode> l2, double[][] weights) {
    for (int i = 0; i < l1.size(); i++) {
      double sigma = 0;
      for (int j = 0; j < l2.size(); j++) {
        sigma += weights[i][j] * l2.get(j).getBackwardOutput();
      }
      l1.get(i).setBackwardInput(sigma);
    }
  }

  // gradient descent
  private void updateWeights() {
    for (int i = 0; i < inputCount; i++) {
      for (int j = 0; j < hiddenCount; j++) {
        inputHiddenWeights[i][j] -= alpha
            * inputNodes.get(i).getForwardOutput()
            * hiddenNodes.get(j).getBackwardOutput();
      }
    }
    for (int j = 0; j < hiddenCount; j++) {
      for (int k = 0; k < outputCount; k++) {
        hiddenOutputWeights[j][k] -= alpha
            * hiddenNodes.get(j).getForwardOutput()
            * outputNodes.get(k).getBackwardOutput();
      }
    }
  }
}
