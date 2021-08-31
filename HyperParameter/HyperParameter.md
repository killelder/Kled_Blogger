# Hyperparameter Optimization(HPO) 超參數優化  
  
> Preface (廢言) : 原先要做RL自動找參數, Survey與親自試驗過後, 發現RL真的是一個大坑, 在與組員討論過後, 決定使用HPO的方式來做找參數的Algorithm, 想到了當初AWS主打的功能之一就是HPO, 所以這邊就參考了AWS的HPO工作原理介紹  
  
- [AWS HPO 工作原理](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html)  
- [Scikit-optimize for LightGBM Tutorial with Luca Massaron | Kaggle's #30daysofML 2021](https://www.youtube.com/watch?v=AFtjWuwqpSQ&list=PLqFaTIg4myu9uAPsqXBBZRr8kcj9IvAIf&index=3)  
- [How to Implement Bayesian Optimization from Scratch in Python](https://machinelearningmastery.com/what-is-bayesian-optimization/)  

上面三篇是我開始HPO的起點  
  
## Reinforcement Learning vs Hyperparameter Optimization Dochi?  
### What's the difference between RL and HPO?  
最近接到一個題目是自動查找參數, 這類型的問題在一開始的時候我是想到RL, 一方面是因為已經接觸過幾次RL的演算法Q Learning, 蒙地卡羅, DQN等等, 但是實際執行RL以後發現並不適合我們, 也不好做, 這裡歸納一下自己對於RL以及HPO的認知  
> **RL is for "Agent interact with env" (different action sequence will affect env)**  
> **HPO is for "Surrogate choice will not affect objective distribution**  
  
RL跟HPO一樣都是一個不斷迭代的過程, 主要的差別就在於RL要解決的問題是迭代過程會影響到主環境的state distribution, 這是甚麼意思? 舉個例子, 你玩圍棋的時候每下一步, 會導致這個位置不能夠再下, 並且會對於你接下來的action產生影響, 或者是打磚塊, 你打完一個磚塊, 之後就不會需要再打這個磚塊了, 你的環境會不斷被Agent給出的action改變, 如果還是有點抽象, 我給你一個方式去判斷  
  
你先做Action A, 再做Action B是否會導致不同的結果? 如果會, 那就是RL的範疇  
  
那麼我們回來看HPO, HPO的surrogate因為只是找最佳參數, 他並不會影響到objective function的distribution, 所以如果是這類型的問題, 就不見意花時間使用RL去解決  
  
### HPO的好處?  
其實硬要說的話RL也還是可以應用在HPO的題目上面的, 那為什麼我們還是要使用HPO呢?  
RL因為原先設計就是要應對變化的各種state, 所以他必須學習各種state, 因此需要大量的Huge data to learn env policy, 我們想像如果要查找2組參數各自的範圍為int : 0~255, 那麼RL很可能就需要滾過所有的可能性256^2次的嘗試, 才有辦法學習到, 這裡還不論到RL需要幾次loop去學習同一個場景, 並且有很大的可能性會受到local max/min的限制走不出圈子  
  
相比之下HPO是利用機率模型, 針對目的性去查找, 例如我要找最大值, 透過先前每次的嘗試給出後面每組參數有可能是最大值的可能性, 由於不需要fitting出全部的objective function, 上面RL的範例, 在這裡只需要100~200次的嘗試就會有很不錯的效果, 這對於如果取得資料cost很高的情況下來說, 更是一大利多, 並且HPO因為不像RL受制於action, 可以在全局的parameter space做搜尋, 較不易遇到local max/min的問題  
  
## HPO的經典三種方法  
- Grid Search  
- Random Search  
- **Bayesian Search(本文主要重點)**  
  
## 隨機搜索(Random Search)  
在Random Search中, 因為調整不依照先前的作業結果, 因此可以運行multithreah, 而不影響搜索的性能  
  
**Random Search效果可能會比Grid Search來的好**  
因為Grid 有可能沒有指定到想要的參數, 就會造成local minimum而非global minimum, 而Random Search又會比全部掃描來的快  
  
## 貝葉斯搜索(Bayesian Search)  
![Exploration-oriented and Exploitation-oriented](https://github.com/killelder/Kled_Blogger/blob/Hyperparameter/HyperParameter/BO-1.png)  
上面左圖顯示Exploration-oriented, 右圖顯示Exploitation-oriented, shade indicates uncertainty  
比grid跟random search好的原因有三個  
1. 不需要了解distribution of hyper-parameters  
2. 使用先驗機率推測下一步最可能的位置 (BS的核心)  
3. 不論objective function is stochastic or discrete, or convex or nonconvex都可以用  
----  
  
### BS process  
1. build a prior distribution of the surrogate model  
2. obtain the hyper-parameter set that performs best on the surrogate model  
3. compute the acquisition function with the current surrogate model  
4. apply the hyper-parameter set to the objective function  
5. update the surrogate model with new results  
6. Loop 1~5 until the optimal configuration is found or the resource limit is reached  
  
#### Surrogate function(代理模型)  
使用沒有參數的surrogate function去模擬objective function的行為, 我們可以根據prior distribution, 來推測出objective function的posterior distribution, 這個posterior probability 就是我們的surrogate objective function.  
因為我們不需要了解objective function, surrogate只是給出每個input對應output的機率並藉此找出目標(例如最大最小值), objective function不需要連續, 但連續的時候surrogate 優化效果最好  
  
從網路上找到的範例, 可以看到它們採用GPR (Gaussian Process Regressor)當作Surrogate function  
```python
# surrogate or approximation for the objective function
def surrogate(model, X):
    # catch any warning generated when making a prediction
    with catch_warnings():
        # ignore generated warnings
        simplefilter("ignore")
        return model.predict(X, return_std=True)
model = GaussianProcessRegressor()
ysamples, _ = surrogate(model, Xsamples)
```
這裡簡化一下GP的過程, 簡單來說GP就是利用檢驗過的點的distribution來推測尚未檢驗的sample點的機率, 離檢驗點越近越確定答案(std越小), mean值也越接近檢驗點的值, 根據這個過程找出其他點的mean以及std, 並依此推論哪個點是最有可能的candidate, 持續loop的過程  
而GP就是透過Gaussian 來定義點跟點之間的相關性  
  
#### Acquisition function  
在我的觀點來看, Acquisition function類似RL的Reward function, 這邊主要是用來balance exploration跟exploitation用的, 用來minimize loss function 來選擇optimal candidate points, 可以使用BO, 但是Gaussian Process(GP)是最廣泛使用的, 不同的acquisition function被提出來, 像是**probability of improvement, GP upper confidence bound(GP-UCB), predictive entropy search, protfolio containing multiple acquisition strategies. 其中expected improvement algorithm 是最常用的**  
  
下面是節錄自[modAL](https://modal-python.readthedocs.io/en/latest/content/query_strategies/Acquisition-functions.html)  
##### Probability of improvement  
![PI](https://github.com/killelder/Kled_Blogger/blob/Hyperparameter/HyperParameter/AcquisitionFunction-1.png)  
從左圖開始, 上圖白點代表量測點, 陰影部分代表不確定性是跟據GP模型算出來的, 離量測點越遠, 標準差越大(不確定性越高), 黑色實線代表實際的objective function, 下圖代表了PI, 我們可以看到幾個特性  
1. 離量測點越遠不確定性越大(上圖陰影部分)  
2. 透過計算可能是最大值的機率而選擇, 機率受到先驗值的mean以及 std影響  
  
##### Expected improvement  
![EI](https://github.com/killelder/Kled_Blogger/blob/Hyperparameter/HyperParameter/AcquisitionFunction-2.png)  
對於EI一樣, 只是EI的下圖改成機率期望值  
  
##### Upper confidence bound  
![UCB](https://github.com/killelder/Kled_Blogger/blob/Hyperparameter/HyperParameter/AcquisitionFunction-3.png)  
與上面一樣, 只是採用UCB作為下圖  
這邊解釋為什麼這個式子可以代表UCB, 因為在該點量測越多次的時候std越小, 量測越少的點std越大, 也就導致UCB越大, 使得UCB會去探索不確定的點, 與蒙地卡羅中的UCB是同樣的概念  
  
回到Acquisition function來看, 下面的sample code是用PI來做, 可以再幫助我們了解整個acquisition function  
下面的例子是要找objective 最大值, 所以他將檢測過的X(input)送進去surrogate  
surrogate會output mean跟std  
選擇觀測到最大的yhat(mean)  
並且利用Xsamples(建議可以越大越好, 避免沒有選取到空間中所有的點), 來代表整個Space的其他點, 根據上面PI的公式把所有的點的PI算出來  
選取PI最大的人做為下一個要選取的目標  
[High Dimension HPO](https://arxiv.org/pdf/2010.03955.pdf)  
[Optimization Modeling in Python: Multiple Objectives Optimization(MOO)](https://medium.com/analytics-vidhya/optimization-modelling-in-python-multiple-objectives-760b9f1f26ee)  
  
```python
def acquisition(X, Xsamples, model):
    # calculate the best surrogate score found so far
    yhat, _ = surrogate(model, X)
    best = max(yhat)
    # calculate mean and stdev via surrogate function
    mu, std = surrogate(model, Xsamples)
    mu = mu[:, 0]
    # calculate the probability of improvement
    probs = norm.cdf((mu - best) / (std+1E-9))
    return probs
    
# optimize the acquisition function
def opt_acquisition(X, y, model):
    # random search, generate random samples
    Xsamples = random(100)
    Xsamples = Xsamples.reshape(len(Xsamples), 1)
    # calculate the acquisition function for each sample
    scores = acquisition(X, Xsamples, model)
    # locate the index of the largest scores
    ix = argmax(scores)
    return Xsamples[ix, 0]
```
#### Gausian Process  
![Gaussian process with 29 samplings](https://github.com/killelder/Kled_Blogger/blob/Hyperparameter/HyperParameter/BO-1.png)  
從上圖可以更清楚的了解BO的背後原理, GP根據前面的sampling來決定還沒sampling點的uncertainty(probability)  
**一般函數給X(input)會希望求出Y(output), 而GP是希望給X(input)求出Y(output)的分布**  
對於集合  
  
假設x與y的關係符合線性回歸  
![formula 0](https://render.githubusercontent.com/render/math?math=y_i=f%28x_i%29=w_ix_i&mode=inline)  
  
將需要預測的![x_i](https://render.githubusercontent.com/render/math?math=x_i&mode=inline)的集合定義為![x_prime](https://render.githubusercontent.com/render/math?math=X\prime&mode=inline)  
我們需要利用![formula 1](https://render.githubusercontent.com/render/math?math=X&mode=inline)跟![formula 2](https://render.githubusercontent.com/render/math?math=f&mode=inline)來預測![formula 2](https://render.githubusercontent.com/render/math?math=f\prime&mode=inline)  
對應的預測為![f_prime](https://render.githubusercontent.com/render/math?math=f\prime&mode=inline)  
根據貝氏定理可以知道下面的公式  
  
![formula 2](https://render.githubusercontent.com/render/math?math=p%28f\prime|f%29=\frac{p%28f|f\prime%29p%28f\prime%29}{p%28f%29}=\frac{p%28f,f\prime%29}{p%28f%29}&mode=inline)
  
假設樣本在每個點是獨立分布的, 所以我們可以知道在每個點的樣本分布  
  
![formula 3](https://render.githubusercontent.com/render/math?math=f=N%28\mu,%20K%29&mode=inline)
  
其中  
![mu](https://render.githubusercontent.com/render/math?math=\mu&mode=inline)為樣本在各sampling點![fset](https://render.githubusercontent.com/render/math?math=[f%28x_1%29,f%28x_2%29,..]&mode=inline)的均值  
![K](https://render.githubusercontent.com/render/math?math=K&mode=inline)為斜方差矩陣(covariance matrix)  
再根據![f_prime](https://render.githubusercontent.com/render/math?math=f\prime&mode=inline)的先驗機率分布  
  
![formula 4](https://render.githubusercontent.com/render/math?math=f\prime=N%28\mu\prime,K\prime%29&mode=inline)
  
就可以推出![pf](https://render.githubusercontent.com/render/math?math=p%28f\prime|f%29&mode=inline)  
其中有兩個核心問題  
1. 如何計算covariance matrix  
2. 如何計算f'的機率分布  
  
計算covariance之前可以先對covariance有點感覺  
![Covariance](https://github.com/killelder/Kled_Blogger/blob/Hyperparameter/HyperParameter/Covariance.png)  
左圖很明顯y跟x關係較小, 右圖很明顯x跟y呈現線性關係  
因為我們不想要真的去計算covariance公式  
我們透過covariance的特性找出kernel作替代  
選擇了[RBF kernel](https://zh.wikipedia.org/wiki/%E5%BE%84%E5%90%91%E5%9F%BA%E5%87%BD%E6%95%B0)  

對於GP來說, 任何參數的mean跟std都是確定的, 我們把參數看成是無限的高斯分布, 對於一個p維的高斯分布而言, 決定分布的是他的  

std反應了高維分布中自己的std, 以及不同維度之間的std  
  
![formula_5](https://render.githubusercontent.com/render/math?math=k%28s,t%29=\sigma%20exp%28-\frac{||s-t||^2}{2t^2}%29&mode=inline)
  
從這個式子可以解讀出, s跟t是兩個不同的sampling點, |s-t|平方可以看做是距離, 這個函數代表的就是s和t兩個sampling點各自代表的高斯分布之間的std差值, 是一個與距離負相關的函數, 當距離越大, 兩個std差越小, 即相關性越小, 反之越靠近的兩個點對應的分布std差值就越大  
  

## Code  
以下是節錄自- [How to Implement Bayesian Optimization from Scratch in Python](https://machinelearningmastery.com/what-is-bayesian-optimization/)的sample code  
下面是採用Probability Improvement的方式  
  
  
```python
# example of bayesian optimization for a 1d function from scratch
from math import sin
from math import pi
from numpy import arange
from numpy import vstack
from numpy import argmax
from numpy import asarray
from numpy.random import normal
from numpy.random import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot

# objective function
def objective(x, noise=0.1):
    noise = normal(loc=0, scale=noise)
    return (x**2 * sin(5 * pi * x)**6.0) + noise

# surrogate or approximation for the objective function
def surrogate(model, X):
    # catch any warning generated when making a prediction
    with catch_warnings():
        # ignore generated warnings
        simplefilter("ignore")
        return model.predict(X, return_std=True)

# probability of improvement acquisition function
def acquisition(X, Xsamples, model):
    # calculate the best surrogate score found so far
    yhat, _ = surrogate(model, X)
    best = max(yhat)
    # calculate mean and stdev via surrogate function
    mu, std = surrogate(model, Xsamples)
    mu = mu[:, 0]
    # calculate the probability of improvement
    probs = norm.cdf((mu - best) / (std+1E-9))
    return probs
    
# optimize the acquisition function
def opt_acquisition(X, y, model):
    # random search, generate random samples
    Xsamples = random(100)
    Xsamples = Xsamples.reshape(len(Xsamples), 1)
    # calculate the acquisition function for each sample
    scores = acquisition(X, Xsamples, model)
    # locate the index of the largest scores
    ix = argmax(scores)
    return Xsamples[ix, 0]

# plot real observations vs surrogate function
def plot(X, y, model):
    # scatter plot of inputs and real objective function
    pyplot.scatter(X, y)
    # line plot of surrogate function across domain
    Xsamples = asarray(arange(0, 1, 0.001))
    Xsamples = Xsamples.reshape(len(Xsamples), 1)
    ysamples, _ = surrogate(model, Xsamples)
    pyplot.plot(Xsamples, ysamples)
    # show the plot
    pyplot.show()

# sample the domain sparsely with noise
X = random(100)
y = asarray([objective(x) for x in X])
# reshape into rows and cols
X = X.reshape(len(X), 1)
y = y.reshape(len(y), 1)
# define the model
model = GaussianProcessRegressor()
# fit the model
model.fit(X, y)
# plot before hand
plot(X, y, model)
# perform the optimization process
for i in range(100):
    # select the next point to sample
    x = opt_acquisition(X, y, model)
    # sample the point
    actual = objective(x)
    # summarize the finding
    est, _ = surrogate(model, [[x]])
    print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
    # add the data to the dataset
    X = vstack((X, [[x]]))
    y = vstack((y, [[actual]]))
    # update the model
    model.fit(X, y)

# plot all samples and the final surrogate function
plot(X, y, model)
# best result
ix = argmax(y)
print('Best Result: x=%.3f, y=%.3f' % (X[ix], y[ix]))
```
  
# 結論  
HPO實作起來比RL親民很多, 速度快又有效, 很多問題其實是不需要用到RL的, 如同前言所說, 如果不會對環境有影響的情況下, 建議可以採取HPO的方式, 實際HPO會遇到的問題可能還要更深一步的模擬實驗才可以看的出來, 例如遇到環境給的reward值會很不連續的時候, 這時候可能就會對global minimum的地方產生障蔽, 以及嘗試了把整個sample space送進去surrogate得到的結果並不一定比random sampling來的更好, 目前還不得而知這樣的效應為何, 但整體來說已經比RL準確許多  
  
## Reference  
- [A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning](https://arxiv.org/abs/1012.2599)  
- [Practical Bayesian Optimization of Machine Learning Algorithms](https://arxiv.org/abs/1206.2944)  
- [Taking the Human Out of the Loop: A Review of Bayesian Optimization](https://ieeexplore.ieee.org/document/7352306?reload=true)  
- [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://liamcli.com/assets/pdf/hyperband_jmlr.pdf)  
- [Google Vizier: A Service for Black-Box Optimization](https://dl.acm.org/doi/10.1145/3097983.3098043)  
- [Learning Curve Prediction with Bayesian Neural Networks](https://openreview.net/forum?id=S11KBYclx)  
- [Speeding up automatic hyperparameter optimization of deep neural networks by extrapolation of learning curves](https://dl.acm.org/doi/10.5555/2832581.2832731)  
- [Scalable Hyperparameter Transfer Learning](https://papers.nips.cc/paper/2018/hash/14c879f3f5d8ed93a09f6090d77c2cc3-Abstract.html)  
- [Bayesian Optimization with Tree-structured Dependencies](http://proceedings.mlr.press/v70/jenatton17a.html)  
- [Proximal Stochastic Methods for Nonsmooth Nonconvex Finite-Sum Optimization](https://papers.nips.cc/paper/2016/hash/291597a100aadd814d197af4f4bab3a7-Abstract.html)  
- [Scalable Bayesian Optimization Using Deep Neural Networks](http://proceedings.mlr.press/v37/snoek15.pdf)  
- [Input Warping for Bayesian Optimization of Non-stationary Functions](https://arxiv.org/abs/1402.0929)  
- [Hyperparameter Optimization](https://www.automl.org/automl/hpo-overview/)  
  
**如果喜歡這篇文章歡迎打賞**  
BTC (BTC) : 1LFRBqvWR9GGizBzoFddkb5xRpAzKRVoDC  
BTC (BSC) : 0xe1cda3eb778d1751af17feac796a4bbe4dbca815  
BTC (ERC20) : 0xe1cda3eb778d1751af17feac796a4bbe4dbca815  
USDT (TRC20) : TT7wgKuYoHwfRy3vCr38Qy3gnS3JJ1aKvn  

