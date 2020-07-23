# Working with Real Data

# Look at the Big Picture

这里以预测房价为例

## Frame the Problem

明确模型的输出数据内容及用途
是一个监督学习，回归任务，单变量回归，batch learning

## Select a Performance Measure

对于回归问题，典型的衡量标准是Root Mean Square Error (RMSE)，它给出了预测会包含多少误差（对于大误差有更大的权重）
![EQU 2-1](PIC/EQU_2_1.png)

尽管RMSE很常用，也可以选择其他方法来评价。比如有离群点较多，可能会选用MAE(mean absolute error)

![EQU 2-2](PIC/EQU_2_2.png)

> The higher the norm index, the more it focuses on large values and neglects small ones. This is why the RMSE is more sensitive to outliers than the MAE. But when outliers are exponentially rare (like in a bell-shaped curve), the RMSE performs very well and is generally preferred.

## Check the Assumptions


# Get the Data


## Create a Test Set
