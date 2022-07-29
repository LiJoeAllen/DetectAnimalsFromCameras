package cn.heavenlybook.utils;

import ai.djl.modality.Classifications;

import java.util.ArrayList;
import java.util.List;

/**
 * @author JoeAllen
 * @date 2022/7/29 9:16 上午
 */
public class ClassificationUtil {
  public static Classifications get(List<Classifications.Classification> items) {
    double sum = 0;
    double max = 0;
    double[] probArr = new double[items.size()];

    List<String> names = new ArrayList<>();
    List<Double> probs = new ArrayList<>();

    for (int i = 0; i < items.size(); i++) {
      Classifications.Classification item = items.get(i);
      double prob = item.getProbability();
      probArr[i] = prob;
      if (prob > max) {
        max = prob;
      }
    }

    for (int i = 0; i < items.size(); i++) {
      probArr[i] = Math.exp(probArr[i] - max);
      sum = sum + probArr[i];
    }

    for (int i = 0; i < items.size(); i++) {
      Classifications.Classification item = items.get(i);
      names.add(item.getClassName());
      probs.add(probArr[i]);
    }

    return new Classifications(names, probs);
  }
}
