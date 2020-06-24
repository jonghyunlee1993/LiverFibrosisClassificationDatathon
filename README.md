[Queens Deep Learning Challenge]    
Deep Learning Challenge for Liver Fibrosis Classification   
======================
# Liver Fibrosis 

간 질환의 종류는 간염, 간경변증, 간암 순으로 그 중증도가 심화되며 다양한 원인으로 초래된다. 간 섬유화는 간 내 염증으로 인한 세포와 기질의 과다한 침착으로 정의되며 만성 간질환이 지속될 경우 간내 구조의 변형과 간세포수의 감소로 간경변으로 진행된다. 또한, 질환 특성상 초기에 이상증세가 발견되지 않아 임상적으로 ‘침묵성 간질환’으로 불리우기도 한다. 중증도가 심화될수록 합병증 및 증상이 순차적으로 발현되며 증상 발현 시에는 이미 그 중증도가 많이 심화되어 있는 경우가 대다수다. 이를 판별하기 위해서는 비침습적 방법을 이용하게 되는데 이는 간경변과 고도 간 섬유증을 진단하고 간 부패와 이의 생명을 예측하기 위해 광범위적으로 사용되는 방법이다. 현재는 간섬유화의 정도에 따라서 등급을 F0에서 F4로 나누고 있다. 

![structure](/Description_Image/lsn_table.png)

그러나 침습적 방법을 사용할 경우 이로 인해 합병증이 발병할 수 있으며, 조직 채취가 간 전체를 대표하지 못하는 문제점이 제기되고 있다. 이러한 임상적 문제를 극복하기 위하여MRI 혹은 CT 영상에 딥러닝 모델을 적용 후 중증도를 예측하고 판단할 수 있는 대안을 제시하고자 한다.

## Framework [  Tensorflow - Keras ]
![structure](/Description_Image/tensorflow.png)
케라스는 파이썬으로 작성된 고수준 신경망 API이며 특히, Tensorflow는 빠른 실험에 특히 중점을 두고 있습니다. 아이디어를 결과물로 최대한 빠르게 구현할수 있어 해당 플랫폼을 선택했습니다.


## Deep Learning Model [  ALEX_NET ]
ALEX NET은 ImageNet 이미지 데이터베이스를 기반으로 이미지를 인식하는데 매우 적합한 모델입니다. ALEX NET은 5개의 Convolution layers와 3개의 Fully-Connected layers로 구성되으며 매우 방대한 CNN 구조를 가지고 있어 사용하였습니다.
![structure](/Description_Image/alexnet.png)

## Deep Learnig Model - VGG16
VGG16은 옥스포드 대학 (University of Oxford)의 K. Simonyan과 A. Zisserman이 제안한 "대규모 이미지 인식을위한 매우 깊은 컨볼 루션 네트워크"논문에서 제시 한 컨볼 루션 신경망 모델입니다. 그리고 대규모 이미지 데이터를 기반으로 CNN 모델 이며 대규모 이미지 데이터를 적용하기에 매우 적합하여 사용하였습니다.
![structure](/Description_Image/vgg16.png)

## Process

- 처음 기존의 F1~ F3 의 데이터를 VGG16 학습결과 F1 ~ F3 영역을 잘 학습할수가 없었음
  : 특히, validation , test set의 있던 데이터셋을 확인한 결과 Liver 영역을 명확하게 확인할 수 없는 데이터가 있음. 
- Liver에 대한 학습이 필요하여 Liver에 대한 레이블링 작업 진행.

## Main Function 
- Generated

## Result

======================
# Conclusion


