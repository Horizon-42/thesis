# FAF to threshold (target)
final glide, make sure the start altitude is >= given altitude
最终滑行段 保证起始点在给定的高度限制之上
make sure the path follow a constant slide degree
保证滑行的垂直方向下降在给定的角度之内
make sure the horizental movement fit for the RNP 

或许这里可以约束航迹的角度和3d的 FAF to target 的角度一直小于给定的角度，优雅地解决问题

# IF to FAF
- altitude above given height of FAF
- horizental movement fit for RNP
也就是通过IF可以给一个横向偏差 x m
但是FAF必须精确通过
对于段中所有的state 和IF到FAF之间的直线之间的垂直距离，小于横向偏差约束
而且，横向的切入角度也要保持小于30度

# IAF to IF
- altitude above given height of IF
- horizental movement fit for RNP
针对给定的多个IAF点，只对段起始点和所有IAF点中的最小值做一个横向偏差约束
对于段终点 也给定一个横向偏差约束

然后对于段中所有state 其和 IAF 到 IF的直线的垂直距离也要小于这个横向偏差约束

# Start to IAF
对段终点和所有IAF的最小距离给一个横向偏差约束


# 控制点分布
以及每段的起点作为一个控制点，每段中再额外增加一个控制点

# Step down 点处理
对于段中出现step down 那么就利用段中的控制点，用这个控制点处的state来约束其altitude above given height


