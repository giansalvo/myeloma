# https://scikit-survival.readthedocs.io/en/stable/user_guide/00-introduction.html
from sksurv.datasets import load_veterans_lung_cancer
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator


def main():
    data_x, data_y = load_veterans_lung_cancer()

    time, survival_prob = kaplan_meier_estimator(data_y["Status"], data_y["Survival_in_days"])
    plt.step(time, survival_prob, where="post")
    plt.ylabel("est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$")
    plt.show()


if __name__ == '__main__':
    main()
