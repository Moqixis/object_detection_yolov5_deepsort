### 1.参考

基于YOLOv5和DeepSort的目标跟踪 [🔗](https://xugaoxiang.com/2020/10/17/yolov5-deepsort-pytorch/#DeepSort)

YOLOv5_deepsort实时多目标跟踪模型 [🔗](https://openbayes.com/console/open-tutorials/containers/BvxvYMbdefV/overview)

☑【YOLOv5】yolov5目标识别+DeepSort目标追踪 [🔗](https://blog.csdn.net/qq_44703886/article/details/121327643)

Yolov5_DeepSort_Pytorch-master 调用流程 [🔗](http://blog.chinaunix.net/uid-20901038-id-5861488.html)

### 2.文件夹 torchreid

视觉任务-目标跟踪检测DeepSort 与yolov5目标检测部署实践 [🔗](https://blog.csdn.net/weixin_46019162/article/details/124523609)

https://www.zhihu.com/question/511584675/answer/2445602236

### 3.保存检测后的视频

加上这个 --save-vid

### 4.训练自己的跟踪模型

【目标跟踪】Yolov5_DeepSort_Pytorch训练自己的数据 [🔗](https://zhuanlan.zhihu.com/p/354945895)

☑YOLOv5_DeepSORT_Pytorch训练自己的多目标跟踪模型 [🔗](https://blog.csdn.net/weixin_50008473/article/details/122347582)

PS：没有train.py把Yolov5_DeepSort_Pytorch的v3版本里面的复制出来一份用

Yolov5 + Deepsort 重新训练自己的数据（保姆级超详细）[🔗](https://blog.csdn.net/weixin_53711236/article/details/123762215)

根据这个：更新版yolov5_deepsort_pytorch实现目标检测和跟踪  [🔗](https://blog.csdn.net/weixin_44238733/article/details/123805195)

改了一下，还是不能跑，把Yolov5_DeepSort_Pytorch的v3版本里面的track.py复制过来用，库文件显示缺失的根据最新版的yolov5目录格式改改就好了

### 5.问题

训练跟踪模型感觉没什么效果，最后测试视频跟不加deep sort结果一样。。。不懂
