package com.fw3e11.bp;

import java.util.List;

public class BPData {

  private List<Double> dataXs;
  private List<Double> dataYs;

  public BPData(List<Double> dataXs, List<Double> dataYs) {
    this.dataXs = dataXs;
    this.dataYs = dataYs;
  }

  public List<Double> getDataXs() {
    return dataXs;
  }

  public List<Double> getDataYs() {
    return dataYs;
  }
}
