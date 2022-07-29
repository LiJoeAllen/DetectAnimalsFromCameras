package cn.heavenlybook;

import ai.djl.ModelException;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.translate.TranslateException;
import cn.heavenlybook.utils.AnimalsClassification;
import cn.heavenlybook.utils.OpenCVImageUtil;
import lombok.extern.java.Log;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.opencv.opencv_core.Mat;

import java.awt.image.BufferedImage;
import java.io.IOException;

/**
 * @author ht
 */
@Log
public class Main {
  public static void main(String[] args) throws IOException, ModelException, TranslateException {
    // 开启摄像头，获取图像（得到的图像为frame类型，需要转换为mat类型进行检测和识别）
    OpenCVFrameGrabber grabber = new OpenCVFrameGrabber(0);
    grabber.start();
    // Frame与Mat转换
    OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
    Frame frame;
    // 获取图像帧
    while ((frame = grabber.grab()) != null) {
      // 将获取的frame转化成mat数据类型
      Mat img = converter.convert(frame);
      BufferedImage buffImg = OpenCVImageUtil.mat2BufferedImage(img);
      Image image = ImageFactory.getInstance().fromImage(buffImg);
      Classifications classifications = AnimalsClassification.predict(image);
      Classifications.Classification bestItem = classifications.best();
      log.info(bestItem.getClassName() + " : " + bestItem.getProbability());
      log.info(classifications.toJson());
    }
  }
}
