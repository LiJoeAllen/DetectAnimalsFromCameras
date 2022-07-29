package cn.heavenlybook.utils;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.util.List;

// https://www.paddlepaddle.org.cn/hubdetail?name=mobilenet_v2_animals&en_category=ImageClassification
//
// /Users/calvin/Desktop/Download/browser/PaddleHub-release-v2.1/modules/image/classification/mobilenet_v2_animals

public final class LightAnimalsClassification {

  private LightAnimalsClassification() {}

  public static Classifications predict(Image img)
      throws IOException, ModelException, TranslateException {
    Classifications classifications = LightAnimalsClassification.classfier(img);
    List<Classifications.Classification> items = classifications.items();
    return ClassificationUtil.get(items);
  }

  public static Classifications classfier(Image img)
      throws IOException, ModelException, TranslateException {

    Criteria<Image, Classifications> criteria =
        Criteria.builder()
            .optEngine("PaddlePaddle")
            .setTypes(Image.class, Classifications.class)
            .optModelUrls(
                "https://aias-home.oss-cn-beijing.aliyuncs.com/models/mobilenet_animals.zip")
            // .optModelUrls("/Users/calvin/model/mobilenet_animals/")
            .optModelName("inference")
            .optTranslator(new AnimalTranslator())
            .optProgress(new ProgressBar())
            .build();

    try (ZooModel<Image, Classifications> rotateModel = ModelZoo.loadModel(criteria)) {
      try (Predictor<Image, Classifications> classifier = rotateModel.newPredictor()) {
        Classifications classifications = classifier.predict(img);
        return classifications;
      }
    }
  }
}
