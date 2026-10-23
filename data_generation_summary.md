혼합모형 시뮬레이션 데이터 생성 정리
================

- [0. 공통 생성 절차](#0-공통-생성-절차)
- [1. t-분포 군집 + 정규분포 군집](#1-t-분포-군집--정규분포-군집)
  - [1.1 모형](#11-모형)
  - [1.2 t-분포의 생성 원리](#12-t-분포의-생성-원리)
  - [1.3 모수 설정](#13-모수-설정)
  - [1.4 코드](#14-코드)
- [2. 2성분 가우시안 데이터 + 1성분 가우시안
  데이터](#2-2성분-가우시안-데이터--1성분-가우시안-데이터)
  - [2.1 구조](#21-구조)
  - [2.2 모수 설정](#22-모수-설정)
  - [2.3 코드](#23-코드)
- [3. 다양한 유형의 변수가 결합된
  데이터](#3-다양한-유형의-변수가-결합된-데이터)
  - [3.1 변수 구성](#31-변수-구성)
  - [3.2 결합분포](#32-결합분포)
  - [3.3 모수 설정](#33-모수-설정)
  - [3.4 코드](#34-코드)
- [4. CWM을 이용한 군집 분류](#4-cwm을-이용한-군집-분류)
  - [4.1 설정](#41-설정)
  - [4.2 코드](#42-코드)

# 0. 공통 생성 절차

모든 예제는 다음 4단계로 데이터를 생성한다.

1.  **군집 라벨 생성**: 각 관측의 소속 군집 $Z_i$를 먼저 생성한다.
2.  **모수 설정**: 군집별 모수를 리스트(또는 벡터)에 `k`번째 원소로
    저장한다.
3.  **군집별 블록 생성**: `lapply(1:K, ...)`로 군집 $k$에 속한 $n_k$개의
    관측을 한 번에 생성한다.
4.  **결합**: `do.call(rbind, gen_block)`으로 블록을 행 방향으로 합친다.

**라벨 생성**

``` r
z <- 1 + rbinom(n, size = 1, prob = 0.7)
```

`rbinom(n, 1, 0.7)`은 0 또는 1을 반환하므로 `z`는 1 또는 2이다.

$$P(Z_i = 2) = 0.7, \qquad P(Z_i = 1) = 0.3$$

즉 혼합비율은 $\pi_1 = 0.3$, $\pi_2 = 0.7$이다. 군집이 3개 이상이면
`sample(1:K, n, replace = TRUE, prob = c(...))`를 사용한다.

**결합 시 주의점**

`do.call(rbind, gen_block)`의 결과는 군집 1의 행이 모두 나온 뒤 군집 2의
행이 나오도록 정렬된다. 따라서 원래 `z`의 순서와 행 순서가 다르다. 각
블록에 라벨 열 `k`를 함께 붙여 두면(`cbind(..., k)`) 이 문제가 없다.

------------------------------------------------------------------------

# 1. t-분포 군집 + 정규분포 군집

## 1.1 모형

$$Z \sim \text{Categorical}(\pi_1 = 0.3, \ \pi_2 = 0.7)$$

$$\mathbf{X} \mid Z = 1 \sim t_2(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1, \nu), \qquad \mathbf{X} \mid Z = 2 \sim N_2(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)$$

$$f(\mathbf{x}) = \pi_1 \, t_2(\mathbf{x}; \boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1, \nu) + \pi_2 \, \phi_2(\mathbf{x}; \boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)$$

$p$변량 t-분포의 밀도는 다음과 같다. 여기서
$\delta(\mathbf{x}) = (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$이다.

$$t_p(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}, \nu) = \frac{\Gamma\left(\frac{\nu + p}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right) (\nu \pi)^{p/2} |\boldsymbol{\Sigma}|^{1/2}} \left[ 1 + \frac{\delta(\mathbf{x})}{\nu} \right]^{-\frac{\nu + p}{2}}$$

## 1.2 t-분포의 생성 원리

$$W \sim \text{Gamma}\left(\tfrac{\nu}{2}, \ \text{rate} = \tfrac{\nu}{2}\right), \quad \mathbf{U} \sim N_p(\mathbf{0}, \boldsymbol{\Sigma}), \quad \mathbf{X} = \boldsymbol{\mu} + \frac{\mathbf{U}}{\sqrt{W}} \ \Longrightarrow \ \mathbf{X} \sim t_p(\boldsymbol{\mu}, \boldsymbol{\Sigma}, \nu)$$

- 한 관측벡터의 모든 성분이 같은 $W$를 공유한다.
- `mvtnorm::rmvt(n, sigma, df, delta, type = "shifted")`가 이 구성으로
  생성한다. `type = "shifted"`일 때 `delta`는 위치벡터
  $\boldsymbol{\mu}$이다.
- `sigma`는 공분산이 아니라 척도행렬이다. $\nu > 2$이면
  $\text{Cov}(\mathbf{X}) = \frac{\nu}{\nu - 2} \boldsymbol{\Sigma}$이다.
  아래 설정($\nu = 3$)에서는 $3 \boldsymbol{\Sigma}_1$이다.

## 1.3 모수 설정

| 군집 | 분포 | $\boldsymbol{\mu}_k$ | $\boldsymbol{\Sigma}_k$ | $\nu$ |
|----|----|----|----|----|
| 1 | $t_2$ | $(2, 5)$ | $\begin{pmatrix} 1 & 0.5 \\ 0.5 & 2 \end{pmatrix}$ | 3 |
| 2 | $N_2$ | $(4, 5)$ | $\begin{pmatrix} 1 & 0.7 \\ 0.7 & 4 \end{pmatrix}$ | — |

$n = 1000$, seed = 1

## 1.4 코드

``` r
library(mvtnorm)
set.seed(1)

# 1) 군집 라벨
z <- 1 + rbinom(1000, size = 1, prob = 0.7)

# 2) 모수
t.mu    <- c(2, 5)
t.sigma <- matrix(c(1, 0.5,
                    0.5, 2), nrow = 2, byrow = TRUE)
t.df    <- 3

n.mu    <- c(4, 5)
n.sigma <- matrix(c(1, 0.7,
                    0.7, 4), nrow = 2, byrow = TRUE)

mu    <- list(t.mu, n.mu)
sigma <- list(t.sigma, n.sigma)

# 3) 군집별 블록 생성: 군집마다 분포가 다르므로 if로 분기
gen_block <- lapply(1:2, function(k){
  nk <- sum(z == k)
  if (k == 1) {
    cbind(rmvt(nk, sigma = sigma[[k]], df = t.df, delta = mu[[k]], type = "shifted"), k)
  } else {
    cbind(rmvnorm(nk, mean = mu[[k]], sigma = sigma[[k]]), k)
  }
})

# 4) 결합
dat_t <- do.call(rbind, gen_block)
colnames(dat_t) <- c("X1", "X2", "z")

head(dat_t)
```

    ##              X1        X2 z
    ## [1,]  2.0164457  4.499303 1
    ## [2,]  0.1984095  4.636055 1
    ## [3,]  3.1811536  7.206739 1
    ## [4,]  0.5371120  4.330781 1
    ## [5,]  2.8651271  3.735286 1
    ## [6,] -8.9484664 -2.552043 1

``` r
table(dat_t[, "z"])
```

    ## 
    ##   1   2 
    ## 304 696

``` r
plot(dat_t[, "X1"], dat_t[, "X2"], col = dat_t[, "z"], xlab = "X1", ylab = "X2")
```

![](data_generation_summary_files/figure-gfm/t-mixture-1.png)<!-- -->

------------------------------------------------------------------------

# 2. 2성분 가우시안 데이터 + 1성분 가우시안 데이터

## 2.1 구조

서로 다른 두 데이터를 따로 생성한 뒤 합친다.

- **데이터 A** ($n_A = 700$): 2성분 가우시안 혼합
  $$\mathbf{X} \sim 0.3 \, \phi_2(\mathbf{x}; \boldsymbol{\mu}_{A1}, \boldsymbol{\Sigma}_{A1}) + 0.7 \, \phi_2(\mathbf{x}; \boldsymbol{\mu}_{A2}, \boldsymbol{\Sigma}_{A2})$$
- **데이터 B** ($n_B = 300$): 단일 가우시안
  $$\mathbf{X} \sim N_2(\boldsymbol{\mu}_B, \boldsymbol{\Sigma}_B)$$

최종 라벨 `z`는 **출처 라벨**이다(A = 1, B = 2). 데이터 A 내부의 성분
라벨 `z1`은 최종 데이터에 저장되지 않는다.

전체 데이터를 성분 기준으로 보면 3개의 가우시안 성분으로 이루어진다. A의
성분 개수는 확률적으로 정해지고(기댓값 210, 490), B의 개수는 300으로
고정이다.

## 2.2 모수 설정

| 데이터 | 성분 | 비율 | $\boldsymbol{\mu}$ | $\boldsymbol{\Sigma}$ |
|----|----|----|----|----|
| A | 1 | 0.3 | $(1, 2)$ | $\begin{pmatrix} 5 & 4 \\ 4 & 5 \end{pmatrix}$ |
| A | 2 | 0.7 | $(5, 7)$ | $\begin{pmatrix} 3 & 2 \\ 2 & 3 \end{pmatrix}$ |
| B | — | — | $(9, 9)$ | $\begin{pmatrix} 3 & 2 \\ 2 & 2 \end{pmatrix}$ |

seed = 1557

## 2.3 코드

``` r
library(MASS)   # mvrnorm()
set.seed(1557)

# 데이터 A: 2성분 가우시안 혼합
pi1   <- 0.3
mu    <- list(c(1, 2), c(5, 7))
sigma <- list(matrix(c(5, 4, 4, 5), 2, byrow = TRUE),
              matrix(c(3, 2, 2, 3), 2, byrow = TRUE))

z1 <- sample(1:2, 700, replace = TRUE, prob = c(pi1, 1 - pi1))
X1 <- rbind(mvrnorm(sum(z1 == 1), mu[[1]], sigma[[1]]),
            mvrnorm(sum(z1 == 2), mu[[2]], sigma[[2]]))

# 데이터 B: 단일 가우시안
mu2    <- c(9, 9)
sigma2 <- matrix(c(3, 2, 2, 2), 2, byrow = TRUE)
X2 <- mvrnorm(300, mu = mu2, Sigma = sigma2)

# 결합: z는 출처 라벨 (A = 1, B = 2)
colnames(X1) <- c("V1", "V2")
colnames(X2) <- c("V1", "V2")
z <- c(rep(1, 700), rep(2, 300))
dat_g <- cbind(rbind(X1, X2), z)

head(dat_g)
```

    ##              V1         V2 z
    ## [1,]  0.5134377  2.0035996 1
    ## [2,] -2.7056459 -2.0881409 1
    ## [3,]  1.3685687  0.8186640 1
    ## [4,]  0.8605438  0.9697562 1
    ## [5,]  0.5960169  1.3931561 1
    ## [6,]  1.0642312  2.8661738 1

``` r
plot(dat_g[, "V1"], dat_g[, "V2"], col = dat_g[, "z"], xlab = "V1", ylab = "V2")
```

![](data_generation_summary_files/figure-gfm/gaussian-combine-1.png)<!-- -->

------------------------------------------------------------------------

# 3. 다양한 유형의 변수가 결합된 데이터

## 3.1 변수 구성

한 관측은 연속형 2개, 개수형 1개, 이진형 1개로 이루어진다.

$$\mathbf{X} = (\mathbf{X}_N, X_P, X_B), \qquad \mathbf{X}_N = (X_{N1}, X_{N2})$$

| 변수 | 유형 | 군집 $k$에서의 조건부 분포 |
|----|----|----|
| $\mathbf{X}_N$ | 연속형 | $N_2(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ |
| $X_P$ | 개수형 | $\text{Poisson}(\lambda_k)$ |
| $X_B$ | 이진형 | $\text{Bin}(m, p_k)$, $m = 1$이므로 $\text{Bernoulli}(p_k)$ |

## 3.2 결합분포

**군집 안에서의 조건부 독립**

코드에서 세 유형의 변수는 같은 군집 안에서 서로 다른 함수(`mvrnorm`,
`rpois`, `rbinom`)로 따로 생성된다. 따라서 $Z = k$가 주어지면
$\mathbf{X}_N$, $X_P$, $X_B$는 서로 독립이다.

$$f(\mathbf{x} \mid Z = k) = \phi_2(\mathbf{x}_N; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \cdot \text{Pois}(x_P; \lambda_k) \cdot \text{Bin}(x_B; m, p_k)$$

**혼합밀도**

$$f(\mathbf{x}) = \sum_{k=1}^{2} \pi_k \, \phi_2(\mathbf{x}_N; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \, \text{Pois}(x_P; \lambda_k) \, \text{Bin}(x_B; m, p_k)$$

각 구성요소는 다음과 같다.

$$\phi_2(\mathbf{x}_N; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{2\pi |\boldsymbol{\Sigma}|^{1/2}} \exp\left\lbrace -\frac{1}{2} (\mathbf{x}_N - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x}_N - \boldsymbol{\mu}) \right\rbrace$$

$$\text{Pois}(x_P; \lambda) = \frac{\lambda^{x_P} e^{-\lambda}}{x_P!}, \quad x_P = 0, 1, 2, \ldots$$

$$\text{Bin}(x_B; m, p) = \binom{m}{x_B} p^{x_B} (1 - p)^{m - x_B}, \quad x_B = 0, 1, \ldots, m$$

**변수 간 의존성**

- 군집 안: $X_{N1}$과 $X_{N2}$는 $\boldsymbol{\Sigma}_k$의 비대각 원소로
  상관을 갖는다. 서로 다른 유형의 변수 사이에는 의존성이 없다.
- 군집을 무시한 전체(주변분포): 모든 변수가 같은 $Z$에 의존하므로 유형이
  다른 변수 사이에도 의존성이 생긴다. 예를 들어 $Z = 2$이면 $X_P$와
  $X_B$가 함께 커지는 경향이 있다.

## 3.3 모수 설정

| 군집 | $\pi_k$ | $\boldsymbol{\mu}_k$ | $\boldsymbol{\Sigma}_k$ | $\lambda_k$ | $p_k$ |
|----|----|----|----|----|----|
| 1 | 0.3 | $(15, 20)$ | $\begin{pmatrix} 1 & 0.1 \\ 0.1 & 1 \end{pmatrix}$ | 2 | 0.2 |
| 2 | 0.7 | $(20, 25)$ | $\begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$ | 3 | 0.8 |

$m = 1$, $n = 1000$, seed = 1557

## 3.4 코드

원본 코드의 `pi <- c(pi1, pi2)`는 R 내장 상수 `pi`(원주율)를 덮어쓰므로
이름을 `p`로 바꾸었다.

``` r
library(MASS)
set.seed(1557)

# 1) 군집 라벨
z <- 1 + rbinom(1000, size = 1, prob = 0.7)
table(z)
```

    ## z
    ##   1   2 
    ## 293 707

``` r
# 2) 모수
mu    <- list(c(15, 20), c(20, 25))
sigma <- list(matrix(c(1, 0.1,
                       0.1, 1), nrow = 2, byrow = TRUE),
              matrix(c(2, 1,
                       1, 2), nrow = 2, byrow = TRUE))
lambda <- c(2, 3)     # Poisson 모수
p      <- c(0.2, 0.8) # Binomial 성공확률 (원본의 pi)
m      <- 1           # Binomial 시행 횟수

# 3) 군집별 블록 생성: 같은 군집 안에서 유형별로 따로 생성 -> 조건부 독립
gen_block <- lapply(1:2, function(k){
  nk <- sum(z == k)
  cbind(mvrnorm(nk, mu[[k]], sigma[[k]]),  # 연속형 2열
        rpois(nk, lambda[k]),              # 개수형 1열
        rbinom(nk, m, p[k]),               # 이진형 1열
        k)                                 # 라벨
})

# 4) 결합
dat_mix <- do.call(rbind, gen_block)
colnames(dat_mix) <- c("y_norm1", "y_norm2", "y_pois", "y_bin", "z")

head(dat_mix)
```

    ##       y_norm1  y_norm2 y_pois y_bin z
    ## [1,] 14.36683 19.13546      3     0 1
    ## [2,] 15.82956 18.54474      2     0 1
    ## [3,] 15.61566 20.05153      1     1 1
    ## [4,] 14.81384 21.56082      3     0 1
    ## [5,] 15.47207 20.39677      1     0 1
    ## [6,] 15.32041 20.57271      0     1 1

**다른 유형을 추가하는 방법**

`cbind()` 안에 해당 유형의 생성 함수를 한 열씩 추가하고, 그 모수를
군집별 벡터나 리스트로 만들면 된다. 예를 들어 지수분포 변수는
`rexp(nk, rate[k])`를 추가한다.

------------------------------------------------------------------------

# 4. CWM을 이용한 군집 분류

## 4.1 설정

- **데이터**: 3절에서 생성한 `dat_mix`
- **모형**: `flexCWM::cwm()`으로 $K = 2$ 군집을 적합한다. 반응변수 없이
  공변량의 분포만 사용한다(`formulaY` 미지정).
- 이 경우 적합하는 모형은 3.2절의 혼합밀도와 같은 형태이다. 연속형은
  다변량 정규, 개수형은 Poisson, 이진형은 Binomial이며, 군집 안에서 유형
  간 조건부 독립을 가정한다.

## 4.2 코드

``` r
# install.packages("flexCWM")
library(flexCWM)

res <- cwm(
  Xnorm    = dat_mix[, c("y_norm1", "y_norm2")],
  Xpois    = dat_mix[, "y_pois", drop = FALSE],
  Xbin     = dat_mix[, "y_bin",  drop = FALSE],
  Xbtrials = m,   # Binomial 시행 횟수 = 1
  k        = 2
)
```

    ## 
    ## Estimating model with k=2, Xnorm=EII ************
    ## Estimating model with k=2, Xnorm=VII **************
    ## Estimating model with k=2, Xnorm=EEI ***************
    ## Estimating model with k=2, Xnorm=VEI ****************
    ## Estimating model with k=2, Xnorm=EVI ************
    ## Estimating model with k=2, Xnorm=VVI ***************
    ## Estimating model with k=2, Xnorm=EEE ********************
    ## Estimating model with k=2, Xnorm=VEE ******************
    ## Estimating model with k=2, Xnorm=EVE **********************
    ## Estimating model with k=2, Xnorm=EEV *****************
    ## Estimating model with k=2, Xnorm=VVE ***
    ## Estimating model with k=2, Xnorm=VEV ******************
    ## Estimating model with k=2, Xnorm=EVV ***********************
    ## Estimating model with k=2, Xnorm=VVV ***********************
    ## 
    ## Best model according to AIC, AICc, AICu, AIC3, BIC, CAIC, ICL is obtained with k = 2 group(s) and parsimonious model VVV 
    ## 
    ## Best model according to AWE is obtained with k = 2 group(s) and parsimonious model VEE

``` r
# 추정 군집 vs 실제 군집
table(estimated = getCluster(res), true = dat_mix[, "z"])
```

    ##          true
    ## estimated   1   2
    ##         1   2 706
    ##         2 291   1

**결과 해석 시 주의점**

추정 군집 번호는 실제 군집 번호와 임의로 대응된다(label switching). 예를
들어 추정 1이 실제 2에 대응할 수 있다. 따라서 표에서 대각선이 아니라
반대 대각선에 관측이 몰려 있어도 분류가 잘 된 것이다.
