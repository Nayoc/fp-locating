# 无线定位深度学习Pytorch实现



## 目录
### /data
1.存放数据集，original为原始数据，format为标准格式化的数据,dataset为tensor打包后的pth文件

2.format层数据集shape默认支持(m,n)的二维数据，切默认m的最后两位为坐标/经纬，其他的维度的数据集例如带时间或channel更多的情况需要自行编写方法处理

3.生成的dataset下的pth文件应该满足data,label分离，以供支持CnnDataBuilder类直接统一加载

### /src
主要程序文件夹
#### /data
进行数据处理转化，格式化等操作
#### /exec
实际训练方法
#### /net
神经网络文件夹、可以自己增加调整需要的cnn或rnn网络算法
#### /util
工具类
#### /view
图表展示工具，可视化展示损失、精确度等定位训练情况

## 启动流程
1.在/data/format文件下添加数据集csv文件，csv格式参考syl_data.csv。或者添加到original文件夹下，自行开发转换方法写入format文件夹下

2.在data_translator文件下使用**build_dataset()**方法从format文件夹下将标准数据集重构成pth文件写入dataset文件夹

3.在start文件中将build.CnnDataBuilder('syl')方法指向的目录修改为自己数据集所在的目录名称

4.执行/调试 start.py 文件

5.每**record_term**次训练后会打印训练损失图、记录网络参数params文件到/models文件夹下，同名的训练后续会自动读取文件基于上次的训练继续

6.其他自行调整或使用默认：CNN类、日志输出、绘图周期、epoch轮次

## GPU支持
代码支持使用多GPU训练，需要电脑是英伟达显卡并开启cuda，
run_train函数会自动搜索可用的GPU并调用pytorch的并行训练函数，需要打印数据在GPU的存放情况自行打印日志

