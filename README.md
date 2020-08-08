# LapCSNet-Pytorch

* Officical code of paper "An efficient deep convolutional laplacian pyramid architecture for CS reconstruction at low sampling ratios" ICASSP2018
* Download the paper: https://arxiv.org/pdf/1804.04970.pdf

## Framework of LapCSNet

![image](https://github.com/WenxueCui/LapCSNet/raw/master/images/framework.jpg)

## Requirements

* Windows10
* Matlab R2015b
* MatconvNet 1.0-beta23

## How to Run

### Training

* Preparing the training data. (T91 and BSDS200 are included in our repo)
* Train the LapCSNet, `train_LapCSN(0.1, 2, 0);`

### Testing

* Preparing the testing data. (Set5 and Set14 are included in our repo)
* Test the LapCSNet, `test_LapCSN_main(100, 200)`

## Experimental Results

### Subjective results

![image](https://github.com/WenxueCui/LapCSNet/raw/master/images/results.jpg)

### Objective results

![image](https://github.com/WenxueCui/LapCSNet/raw/master/images/table.jpg)

## Additional instructions

* For training data, you can choose any dataset by yourself.
* When subrate<=0.25, the laplacian structure can be used.
* If you have any problem, please email wenxuecui@stu.hit.edu.cn

## Citation

```
@article{Cui2018An,
  title={An efficient deep convolutional laplacian pyramid architecture for CS reconstruction at low sampling ratios},
  author={Cui, Wenxue and Xu, Heyao and Gao, Xinwei and Zhang, Shengping and Jiang, Feng and Zhao, Debin},
  year={2018},
}
```

## Acknowledgments


