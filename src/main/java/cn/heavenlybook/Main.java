package cn.heavenlybook;

import ai.djl.ModelException;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.translate.TranslateException;
import cn.heavenlybook.utils.AnimalsClassification;
import cn.heavenlybook.utils.OpenCvImageUtil;
import lombok.extern.java.Log;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.opencv.opencv_core.Mat;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;

/**
 * @author ht
 */
@Log
public class Main {
  static int i = 0;

  public static void main(String[] args) throws IOException, ModelException, TranslateException {
    // 开启摄像头，获取图像（得到的图像为frame类型，需要转换为mat类型进行检测和识别）
    OpenCVFrameGrabber grabber = new OpenCVFrameGrabber(0);
    grabber.start();
    // Frame与Mat转换
    OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
    Frame frame;
    // 新建一个预览窗口
    CanvasFrame canvas = new CanvasFrame("动物检测");
    canvas.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    canvas.setVisible(true);
    canvas.setFocusable(true);
    // 窗口置顶
    if (canvas.isAlwaysOnTopSupported()) {
      canvas.setAlwaysOnTop(true);
    }
    // 获取图像帧
    while (canvas.isVisible() && (frame = grabber.grab()) != null) {
      // 将获取的frame转化成mat数据类型
      Mat img = converter.convert(frame);
      BufferedImage buffImg = OpenCvImageUtil.mat2BufferedImage(img);
      Image image = ImageFactory.getInstance().fromImage(buffImg);
      Classifications classifications = AnimalsClassification.predict(image);
      Classifications.Classification bestItem = classifications.best();
      log.info(bestItem.getClassName() + " : " + bestItem.getProbability());
      log.info(classifications.toJson());
      // 显示视频图像
      canvas.showImage(frame);
      if (bestItem.getProbability() < 0.00001 && !"非动物".equals(bestItem.getClassName())) {
        File outPutFile = Paths.get("src/test/resources/coffee" + i++ + ".png").toFile();
        if (buffImg != null) {
          if (!outPutFile.exists()) {
            log.info(String.valueOf(outPutFile.createNewFile()));
          }
          ImageIO.write(buffImg, "png", outPutFile);
        }
      }
    }
    canvas.dispose();
    converter.close();
    grabber.close();
  }
}
