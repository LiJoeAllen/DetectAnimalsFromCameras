package cn.heavenlybook;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import cn.heavenlybook.utils.AnimalsClassification;
import cn.heavenlybook.utils.FaceDetection;
import cn.heavenlybook.utils.OpenCVImageUtil;
import lombok.extern.java.Log;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;

import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_imgproc.*;

/**
 * @author ht
 */
@Log
public class Main {
  public static void main(String[] args) throws IOException, ModelException, TranslateException {
    float shrink = 0.5f;
    float threshold = 0.7f;

    Criteria<Image, DetectedObjects> criteria = new FaceDetection().criteria(shrink, threshold);

    // 开启摄像头，获取图像（得到的图像为frame类型，需要转换为mat类型进行检测和识别）
    OpenCVFrameGrabber grabber = new OpenCVFrameGrabber(0);

    grabber.start();

    // Frame与Mat转换
    OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
    // 新建一个预览窗口
    CanvasFrame canvas = new CanvasFrame("人脸检测");
    canvas.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    canvas.setVisible(true);
    canvas.setFocusable(true);
    // 窗口置顶
    if (canvas.isAlwaysOnTopSupported()) {
      canvas.setAlwaysOnTop(true);
    }
    Frame frame;

    try (ZooModel<Image, DetectedObjects> model = ModelZoo.loadModel(criteria);
        Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
      // 获取图像帧
      while (canvas.isVisible() && (frame = grabber.grab()) != null) {

        // 将获取的frame转化成mat数据类型
        Mat img = converter.convert(frame);
        BufferedImage buffImg = OpenCVImageUtil.mat2BufferedImage(img);

        Image image = ImageFactory.getInstance().fromImage(buffImg);

        // region 动物识别
        Classifications classifications = AnimalsClassification.predict(image);

        Classifications.Classification bestItem = classifications.best();
        System.out.println(bestItem.getClassName() + " : " + bestItem.getProbability());
        //    List<Classifications.Classification> items = classifications.items();
        //    List<String> names = new ArrayList<>();
        //    List<Double> probs = new ArrayList<>();
        //    for (int i = 0; i < items.size(); i++) {
        //      Classifications.Classification item = items.get(i);
        //      names.add(item.getClassName());
        //      probs.add(item.getProbability());
        //    }

        log.info(classifications.toJson());
        // endregion

        int imageWidth = image.getWidth();
        int imageHeight = image.getHeight();

        DetectedObjects detections = predictor.predict(image);
        List<DetectedObjects.DetectedObject> items = detections.items();

        // 遍历人脸
        for (DetectedObjects.DetectedObject item : items) {
          BoundingBox box = item.getBoundingBox();
          Rectangle rectangle = box.getBounds();
          int x = (int) (rectangle.getX() * imageWidth);
          int y = (int) (rectangle.getY() * imageHeight);
          Rect face =
              new Rect(
                  x,
                  y,
                  (int) (rectangle.getWidth() * imageWidth),
                  (int) (rectangle.getHeight() * imageHeight));

          // 绘制人脸矩形区域，scalar色彩顺序：BGR(蓝绿红)
          rectangle(img, face, new Scalar(0, 0, 255, 1));

          int posX = Math.max(face.tl().x() - 10, 0);
          int posY = Math.max(face.tl().y() - 10, 0);
          // 在人脸矩形上面绘制文字
          putText(
              img,
              "Face",
              new Point(posX, posY),
              FONT_HERSHEY_COMPLEX,
              1.0,
              new Scalar(0, 0, 255, 2.0));
        }

        // 显示视频图像
        canvas.showImage(frame);
      }
    }

    canvas.dispose();
    grabber.close();
  }
}
