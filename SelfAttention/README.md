# SelfAttention model
센서간의 관계성을 SelfAttention model을 통해 학습시키고 가중치를 부여한다.<br/>
그 후 Machine prediction을 진행하려는 목적.<br/>

기본적으로 SelfAttetnion을 진행하면<br/>
Machine 끼리의 관계성을 학습시키는 모델이 생성된다.<br/>
따라서 Data를 transpose하여 sensor x machine 으로 sensor 데이터를 학습시키려고 한다.<br/>

하지만 이 과정에서 machine의 수가 Linear layer의 파라미터로 입력된다.<br/>
이는 train-test split 과정에서 machine의 수가 달라질 수 있기 때문에<br/>
학습 >> 검증 과정의 model parameter가 달라지는 문제가 발생한다.<br/>
현재 이 문제를 해결하는 과정 중에 있다.<br/>
