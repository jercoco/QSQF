# Nonparametric Probabilistic Forecasting for Wind Power Generation using Quadratic Spline Quantile Function and Autoregressive Recurrent Neural Network
This is an implementation of the paper "Nonparametric Probabilistic Forecasting for Wind Power Generation using Quadratic Spline Quantile Function and Autoregressive Recurrent Neural Network".

This implementation is mainly based on the [TimeSeries repository](https://github.com/zhykoties/TimeSeries). We deeply thank to the work by Yunkai Zhang, Qiao Jiang and Xueying Ma. We also thank to [GEFCom2014](https://www.sciencedirect.com/science/article/abs/pii/S0169207016000133#ec000005) to provide open source data.

## Authors:
* **Ke Wang**(<w110k120@stu.xjtu.edu.cn>) - *Xi'an Jiaotong University, Xi'an* 

* **Yao Zhang**(<yaozhang_ee@ieee.org>) - *Xi'an Jiaotong University, Xi'an* 

* **Fan Lin**(<lf1206@stu.xjtu.edu.cn>) - *Xi'an Jiaotong University, Xi'an* 

* **Jianxue Wang**(<jxwang@mail.xjtu.edu.cn>) - *Xi'an Jiaotong University, Xi'an* 

* **Morun Zhu**(<1491974695@qq.com>) - *Xi'an Jiaotong University, Xi'an* 

## Introduction
We propose a non-parametric and flexible method for probabilistic wind power forecasting. First, the distribution of wind power output is specified by spline quantile function, which avoids assuming a parametric form and also provides flexible shape for wind power density. Then, auto-regressive recurrent neural network is used to build the non-linear mapping from input features to the parameters of quadratic spline quantile function. A novel loss function based on continuous ranked probability score (CRPS) is designed to train the forecasting model. We also derive the closed-form solution of the integral required in computing the CRPS-based loss function in order to improve the computational efficiency in training.


## Results
The main results of four QSQF models for reliabilty, sharpness and CRPS are listed in the following table.
| Criterion | QSQF-A | QSQF-B | QSQF-AB | QSQF-C |
| :-------: | :----: | :----: | :-----: | :----: |
| MRAE      | 0.0401 | 0.0506 | 0.0397  | 0.0286 |
| NAPS      | 0.3423 | 0.3530 | 0.3577  | 0.3698 |
| CRPS      | 0.0811 | 0.0832 | 0.0868  | 0.0764 |

