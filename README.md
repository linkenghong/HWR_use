# HA-AILab 手写签名识别
基于卷积神经网络的中文手写签名识别

## 依赖包
* PIL
* numpy
* torch
* torchvision

## 使用说明
1. 调用ha_SignCompare中 ha_SignCompare函数，该函数使用如下，其接受两个输入，一个是图片，另一个是对应文字；
```
ha_SignCompare(r'./data/边寿博.jpg', '边寿博')
```
2. 函数会根据输入图片的宽高比，自动识别是横向签名还是竖向签名；


3. 针对竖向签字，先使用投影法进行切割，对于投影法切割不理想的使用连通域法进行切割；


4. 针对横向签字，则直接使用高斯平滑投影直方图取极值点作为备选切割点再进一步得到最优切割点。


5. 输出参数有4个，分别识别成功与否的代码code，返回描述message，识别出的名字字数char_count，识别明细char_list。
