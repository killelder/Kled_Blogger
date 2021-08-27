# Hyperparameter Optimization(HPO) 超參數優化  
  
> Preface (廢言) : 原先要做RL自動找參數, Survey與親自試驗過後, 發現RL真的是一個大坑, 在與組員討論過後, 決定使用HPO的方式來做找參數的Algorithm, 想到了當初AWS主打的功能之一就是HPO, 所以這邊就參考了AWS的HPO工作原理介紹  
  
- [AWS HPO 工作原理](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html)  
- [Scikit-optimize for LightGBM Tutorial with Luca Massaron | Kaggle's #30daysofML 2021](https://www.youtube.com/watch?v=AFtjWuwqpSQ&list=PLqFaTIg4myu9uAPsqXBBZRr8kcj9IvAIf&index=3)  
- [How to Implement Bayesian Optimization from Scratch in Python](https://machinelearningmastery.com/what-is-bayesian-optimization/)  

上面三篇是我開始HPO的起點  
## HPO的經典三種方法  
- Grid Search  
- Random Search  
- Bayesian Search  
  
## 隨機搜索(Random Search)  
在Random Search中, 因為調整不依照先前的作業結果, 因此可以運行multithreah, 而不影響搜索的性能, Random Search效果可能會比Grid Search來的好, 因為Grid 有可能沒有指定到想要的參數, 就會造成local minimum而非global minimum  
而Random Search又會比全部掃描來的快  
  
## 貝葉斯搜索(Bayesian Search)  
![Exploration-oriented and Exploitation-oriented](https://github.com/killelder/Kled_Blogger/blob/Hyperparameter/HyperParameter/BO-1.png)  
上面左圖顯示Exploration-oriented, 右圖顯示Exploitation-oriented, shade indicates uncertainty  
比grid跟random search好的原因有兩個  
1. 不需要了解distribution of hyper-parameters  
2. 使用後驗機率 (BO的核心)  
3. 不論objective function is stochastic or discrete, or convex or nonconvex都可以用  
----
### BO process  
1. build a prior distribution of the surrogate model  
2. obtain the hyper-parameter set that performs best on the surrogate model  
3. compute the acquisition function with the current surrogate model  
4. apply the hyper-parameter set to the objective function  
5. update the surrogate model with new results
6. Loop 1~5 until the optimal configuration is found or the resource limit is reached  
  
#### Surrogate function(代理模型)  
使用沒有參數的surrogate function去模擬objective function, 我們可以根據prior distribution, 來推測出objective function的posterior distribution, 這個posterior probability 就是我們的surrogate objective function.  
  
#### Acquisition function  
是用來balance exploration跟exploitation用的, 用來minimize loss function 來選擇optimal candidate points, 可以使用BO, 但是Gaussian Process(GP)是最廣泛使用的, 不同的acquisition function被提出來, 像是probability of imporvement, GP upper confidence bound(GP-UCB), predictive entropy search, protfolio containing multiple acquisition strategies. 其中expected improvement algorithm 是最常用的  
  
#### Gausian Process  
![Gaussian process with 29 samplings](https://github.com/killelder/Kled_Blogger/blob/Hyperparameter/HyperParameter/BO-1.png)  
從上圖可以更清楚的了解BO的背後原理, GP根據前面的sampling來決定還沒sampling點的uncertainty(probability)  
對於GP來說, 任何參數的mean跟std都是確定的, 我們把參數看成是無限的高斯分布, 對於一個p維的高斯分布而言, 決定分布的是他的  
mean = $$\mu_p$$  
  
std = $$\sum_{pxp}$$  
std反應了高維分布中自己的std, 以及不同維度之間的std  
std function = $$k(s, t) = \sigma exp(- \frac{||s-t||^2}{2t^2})$$  
  
從這個式子可以解讀出, s跟t是兩個不同的sampling點, |s-t|平方可以看做是距離, 這個函數代表的就是s和t兩個sampling點各自代表的高斯分布之間的std差值, 是一個與距離負相關的函數, 當距離越大, 兩個std差越小, 即相關性越小, 反之越靠近的兩個點對應的分布std差值就越大  
  
### 更簡化的來討論GP
假設我們可以取樣空間中所有點, 理論上可以得到function的樣子, 但如果我們沒有辦法得到所有的點又想要推論function, 假設有三個點ABC, AB比AC更近, 那麼AB有更高的相關性, AC有較低的相關性, 如果在A得到了一組數據, 想根據A推論B跟C, 這樣推論B應該有比較小的不確定性, 如果我們把採樣點都套用這個方式去推論, 就可以得到整個function的大概樣貌  
假設我們知道x=4跟x=4.5相關係數是0.8, 因為我們每次採樣都是有誤差的, (該sampling點自己的std), 所以我們用gaussian error來描述該點, 假設我們知道觀測x=4得到3 std=1, 那麼推測x=4.5的數據, 只需要找到那些滿足x=4的點的條件以及相關係數的條件, 就可以推斷出來x=4.5了  
而GP就是透過Gaussian 來定義點跟點之間的相關性  
  
**二維的gp叫gaussian random field**  
  
大家跑Gaussian process經驗是只有參數和kernel選的非常好fit才會好, 不然会失之千里。 相比之下 **RBF radio basis function fit surface**的效果更穩健  
  
  

  
```python
import matplotlib.pyplot as plt
import numpy as np

def gaussian_kernel(x1, x2, l=0.5, sigma_f=0.2):
    m, n = x1.shape[0], x2.shape[0]
    dis_matrix = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            dist_matrix[i][j] = np.sum((x1[i] - x2[j]) ** 2)
    return sigma_f ** 2 * np.exp( -0.5 / l ** 2 * dist_matrix)
def getY(X):
    X = np.asarray(X)
    Y = np.sin(X)*0.4 + np.random.normal(0, 0.05, size=X.shape)
    return Y.tolist()
def update(X, X_star):
    X = np.asarray(X)
    X_star = np.asarray(X_star)
    K_YY = gaussian_kernel(X, X)
    K_ff = gaussian_kernel(X_star, X_star)
    K_Yf = gaussian_kernel(X, X_star)
    K_fY = gaussian_kernel(X_star, X)
    K_YY_inv = np.linalg.inv(K_YY + 1e-8 * np.eye(len(X)))
    
    mu_star = K_fY.dot(K_YY_inv).dot(Y)
    cov_star = K_ff - K_fY.dot(K_YY_inv).dot(K_Yf)
    return mu_star, cov_star

f, ax = plt.subplots(2, 1, sharex=True, sharey=True)
X_pre = np.arange(0, 10, 0.1)
mu_pre = np.array([0]*len(X_pre))
Y_pre = mu_pre
cov_pre = gaussian_kernel(X_pre, X_pre)
uncertainty = 1.96 * np.sqrt(np.diag(cov_pre)) #95% believe area
ax[0].fill_between(X_pre, Y_pre + uncertainty, Y_pre - uncertianty, alpha=0.1)
ax[0].plot(X_pre, Y_pre, label='expection')
ax[0].legend()

X= np.array([1, 3, 7, 9].reshape(-1, 1))
Y = getY(X)
X_star = np.arange(0, 10, 0.1).reshape(-1, 1)
mu_star, cov_star = update(X, X_star)
Y_star = mu_star.ravel()
uncertainty = 1.96 * np.sqrt(np.diag(cov_pre)) #95% believe area
ax[1].fill_between(X_star.ravel(), Y_star + uncertainty, Y_star - uncertianty, alpha=0.1)
ax[1].plot(X_star, Y_star, label='expection')
ax[1].scatter(X, Y, label='observation point', c='red', markr='x')
ax[1].legend()
plt.show()
```

### What is Bayesian Optimization  
### Sequential model-based optimization (SMBO)  

## SMAC  
## TPE  

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

