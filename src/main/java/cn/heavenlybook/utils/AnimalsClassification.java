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

// https://www.paddlepaddle.org.cn/hubdetail?name=resnet50_vd_animals&en_category=ImageClassification
//
// Users/calvin/Desktop/Download/browser/PaddleHub-release-v2.1/modules/image/classification/resnet50_vd_animals

/**
 * @author ht
 */
public final class AnimalsClassification {

  private AnimalsClassification() {}

  public static Classifications predict(Image img)
      throws IOException, ModelException, TranslateException {
    Classifications classifications = AnimalsClassification.classifier(img);
    List<Classifications.Classification> items = classifications.items();
    return ClassificationUtil.get(items);
  }

  static Criteria<Image, Classifications> criteria;
  static ZooModel<Image, Classifications> rotateModel;
  static Predictor<Image, Classifications> classifier;

  public static Classifications classifier(Image img)
      throws IOException, ModelException, TranslateException {

    criteria =
        criteria == null
            ? Criteria.builder()
                .optEngine("PaddlePaddle")
                .setTypes(Image.class, Classifications.class)
                .optModelUrls("https://aias-home.oss-cn-beijing.aliyuncs.com/models/animals.zip")
                //            .optModelUrls("/Users/calvin/model/animals/")
                .optModelName("inference")
                .optTranslator(new AnimalTranslator())
                .optProgress(new ProgressBar())
                .build()
            : criteria;
    rotateModel = rotateModel == null ? ModelZoo.loadModel(criteria) : rotateModel;
    classifier = classifier == null ? rotateModel.newPredictor() : classifier;
    return classifier.predict(img);
  }
}
