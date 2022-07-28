## 目录：

# 动物分类识别

动物识别模型，支持7978种动物的分类识别。

### SDK功能

- 支持7978种动物的分类识别，并给出置信度。
- 提供两个可用模型例子 1). 大模型(resnet50)https://aias-home.oss-cn-beijing.aliyuncs.com/models/animals.zip
  2). 小模型(mobilenet_v2)https://aias-home.oss-cn-beijing.aliyuncs.com/models/mobilenet_animals.zip

[动物分类](https://aias-home.oss-cn-beijing.aliyuncs.com/AIAS/animal_sdk/animals.txt)

## 运行例子

- 测试图片
  ![tiger](https://aias-home.oss-cn-beijing.aliyuncs.com/AIAS/animal_sdk/tiger.jpeg)

运行成功后，命令行应该看到下面的信息:

```text
西伯利亚虎 : 1.0
[INFO ] - [
	class: "西伯利亚虎", probability: 1.00000
	class: "孟加拉虎", probability: 0.02022
	class: "华南虎", probability: 0.00948
	class: "苏门答腊虎", probability: 0.00397
	class: "印度支那虎", probability: 0.00279
]
```

### 帮助

### 官网：

### Git地址：