## Material for my Applied ML course at Ben-Gurion University

**This course covers the applied side of algorithmics in machine learning and deep learning, focusing on hands-on coding experience in Python.**

***	

**Contents**

- What is machine learning (ML)?
- Basics of Python programming
- Applying ML: evaluation, dataset splits, cross-validation, performance measures, bias/variance tradeoff, visualization, confusion matrix, choosing estimators, hyperparameter tuning, statistics
- Supervised learning: models, features, objectives, model training, overfitting, regularization, classification, regression, gradient descent, k nearest neighbors, linear regression, logistic regression, decision tree, random forest, adaptive boosting, gradient boosting, support vector machine, naïve Bayes
- Dimensionality reduction: principal component analysis
- Unsupervised learning: hierarchical clustering, k-means, t-SNE ​
- Artificial neural networks: backpropagation, deep neural network, convolutional neural network
- Evolutionary algorithms: genetic algorithm, ​genetic programming

***	

**Lesson plan**

​[​](http://www.evolutionarycomputation.org/slides/)#1 Python, AI+ML Intro​

- [Python Programming](https://pythonbasics.org/) ([Python](https://www.python.org/downloads/windows/), [PyCharm](https://www.jetbrains.com/pycharm/download/#section=windows)), [Pandas](https://pythonbasics.org/what-is-pandas/), [NumPy](https://numpy.org/devdocs/user/absolute_beginners.html) / [NumPy](https://www.w3schools.com/python/numpy/default.asp) ([Numba](https://numba.pydata.org/)) ([np.dot](https://numpy.org/doc/stable/reference/generated/numpy.dot.html) vs [loop example](code/loop-vs-np.py))
- [Computing Machinery and Intelligence](https://www.cs.mcgill.ca/~dprecup/courses/AI/Materials/turing1950.pdf)
- [Machine Learning: history, applications, recent successes](https://data-psl.github.io/lectures2020/slides/01_machine_learning_successes)

#2 ML Intro, Simple Example, KNN, Cross-Validation

- [Introduction to machine learning](https://data-psl.github.io/lectures2020/slides/02_intro_to_machine_learning)
- [simple weather example](code/weather.py)
- [iris knn (map)](https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py)
- [kfold​](code/kfold.py)

#3 Scikit, Models, Decision Trees

- [Machine learning with scikit-learn](https://data-psl.github.io/lectures2020/slides/04_scikit_learn/#1)
- [Machine learning models](https://data-psl.github.io/lectures2020/slides/03_machine_learning_models/)
- [Boston dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#boston-dataset)
- ​[Decision trees](https://youtu.be/_L39rN6gz7Y)​

#4 Random Forest, Linear Regression Logistic Regression

- [Machine learning models](https://data-psl.github.io/lectures2020/slides/03_machine_learning_models/)
- [Random Forests](https://youtu.be/J4Wdy0Wc_xQ)
- [Linear Regression](https://youtu.be/PaFPbb66DxQ)
- [Linear Regression Example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html), [LinearReg.py](code/LinearReg.py)
- [Logistic Regression](https://youtu.be/yIYKR4sgzI8), [Logistic Regression](https://www.analyticsvidhya.com/blog/2021/07/an-introduction-to-logistic-regression/), [Cross-Entropy Loss](https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e)
- [Optimization of linear models](https://data-psl.github.io/lectures2020/slides/05_optimization_linear_models/)
- ​[Ridge vs. Lasso](https://www.statology.org/when-to-use-ridge-lasso-regression/)​

#5 AdaBoost, Gradient Boosting

- Summary: [Linear Regression](https://medium.com/analytics-vidhya/a-quick-summary-of-linear-regression-42d1dab85e3e), [Logistic Regression](https://www.analyticsvidhya.com/blog/2021/07/an-introduction-to-logistic-regression/), [LinVsLog](code/LinVsLog.py), [PolynomialFeatures](code/PolynomialFeatures.py)
- [Adaptive Boosting](https://youtu.be/LsK-xG1cLYA)​​
- ​[Gradient Boosting](https://youtu.be/3CC4N4z3GJc)
- [AddGBoost](https://www.sciencedirect.com/science/article/pii/S2666827021001225)​

#6 XGBoost, Comparing ML algos, Gradient Descent

- Reminder: [Adaboost](https://www.cs.bgu.ac.il/~sipper/adaboost.jpg), [Gradient boost](https://www.cs.bgu.ac.il/~sipper/gradboost.jpg)
- ​[XGBoost](https://youtu.be/OtD8wVaFm6E)
- [Comparing supervised learning algorithms](https://www.dataschool.io/comparing-supervised-learning-algorithms/)
- ​[How to find the best performing Machine Learning algorithm](https://medium.com/analytics-vidhya/how-to-find-the-best-performing-machine-learning-algorithm-dc4eb4ff34b6) ([load\_iris ](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)vs. [load\_wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html), plt.show(), [Lazy Predict](https://lazypredict.readthedocs.io/en/latest/readme.html))
- [Gradient Descent](https://youtu.be/sDv4f4s2SB8) ([Least Squares](https://www.mathsisfun.com/data/least-squares-regression.html), [Least Squares](https://textbooks.math.gatech.edu/ila/least-squares.html))
- [Stochastic Gradient Descent](https://youtu.be/vMh0zPT0tLI)

#7 Choosing Model, SVM, Bayes, Metrics

- [Choosing the right estimator](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html) ([one model doesn&#39;t fit all](https://media-exp1.licdn.com/dms/image/C5622AQHK0ySAmVHlvQ/feedshare-shrink_800/0/1652450701365?e=1655942400&amp;v=beta&amp;t=3KKtUu5AD30HD1n75tXUizL6UdgeDX8sBjH2UH20XfE)...)
- [Support Vector Machine](https://youtu.be/efR1C6CvhmE) [(](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)[sklearn.svm.SVC)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), [Plot different SVM classifiers in the iris dataset](https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html#sphx-glr-auto-examples-svm-plot-iris-svc-py)
- [Multinomial Naive Bayes](https://youtu.be/O2L2Uv9pdDA)
- Metrics: [sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html?highlight=metrics#module-sklearn.metrics), [Confusion Matrix](https://youtu.be/Kdsp6soqA7o), [Sensitivity and Specificity,](https://youtu.be/vP06aMoz4v8) [ROC and AUC](https://youtu.be/4jRBRDbJemM), ​[balanced accuracy](https://www.statology.org/balanced-accuracy/), [balanced accuracy](https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score), [various metrics from CM](https://en.wikipedia.org/wiki/Confusion_matrix)

#8 Metrics, PCA, t-SNE, Clustering

- [What companies think AI looks like...](https://media-exp1.licdn.com/dms/image/C4D22AQGHDRqNV6uPPg/feedshare-shrink_1280/0/1653029435934?e=1655942400&amp;v=beta&amp;t=PQytCPFFbbKE-AbHBLo3zsNbRcQ8nVqTqb6JbTc4kUs), [ML in practice](https://info.deeplearning.ai/the-batch-one-model-for-hundreds-of-tasks-recognizing-workplace-hazards-when-data-means-danger-vision-transformer-upgrade-1)
- [balanced accuracy](https://www.statology.org/balanced-accuracy/), [balanced accuracy](https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score), [various metrics from CM](https://en.wikipedia.org/wiki/Confusion_matrix)
- [11 Different Uses of Dimensionality Reduction](https://towardsdatascience.com/11-different-uses-of-dimensionality-reduction-4325d62b4fa6)
- ​[PCA](https://youtu.be/FgakZw6K1QQ), [PCA](https://www.sartorius.com/en/knowledge/science-snippets/what-is-principal-component-analysis-pca-and-how-it-is-used-507186), [PCA vs LR](https://starship-knowledge.com/pca-vs-linear-regression#PCA_vs_Linear_Regression_-_How_do_they_Differ), [pca.py](code/pca.py)
- ​[t-SNE](https://youtu.be/NEaUSP4YerM), [tsne.py](code/tsne.py)
- [Hierarchical clustering](https://youtu.be/7xHsRkOdVwo)
- [K-means clustering](https://youtu.be/4b5d3muPQmA), [kmeans.py](code/kmeans.py)

#9 Hyperparameter Tuning, p-vals, t-test, Permutation Test

- [Hyperparameter tuning](https://towardsdatascience.com/hyperparameter-tuning-explained-d0ebb2ba1d35), [Hyperparameter tuning](https://towardsdatascience.com/hyperparameter-tuning-a-practical-guide-and-template-b3bf0504f095), [Optuna](https://optuna.org/), [optuna.py](code/optuna.py)
- [p-values](https://youtu.be/vemZtEM63GY), [how to calculate p-values](https://youtu.be/JQc3yx0-Q9E), [p-hacking](https://youtu.be/HDCOUXE3HMM)
- [Probability is not Likelihood](https://www.youtube.com/watch?v=pYxNSUDSFH4)
- [17 Statistical Hypothesis Tests in Python](https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/)
- [t-test](https://youtu.be/0Pd3dc1GcHc), ​[t-test](https://youtu.be/pTmLQvMM-1M) ([t-test vs p-value)](https://askanydifference.com/difference-between-t-test-and-p-value/), [scpipy ttest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)
- ([chi-square](https://youtu.be/ZjdBM7NO7bY))
- [permutation test](https://youtu.be/GmvpsJHGCxQ) ([AddGBoost ](https://www.sciencedirect.com/science/article/pii/S2666827021001225)+ [code](https://github.com/moshesipper/AddGBoost)), [permutation test](https://www.linkedin.com/feed/update/urn:li:activity:6934781784937107456/)

#10, #11 Neural Networks

- ​[Neural networks](https://youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1)
- [Neural Networks with À La Carte Selection of Activation Functions](https://drive.google.com/file/d/10wy_gFPoNLwQXHkHKAojYBxbKkB5W_da/view?usp=sharing)
- [PyTorch](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html), [PyTorch](https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e)
- [Convolution...](https://www.linkedin.com/posts/pascalbornet_artificialintelligence-ugcPost-6925288775740776448-0S-K/)
- [Growth of AI computing](https://twitter.com/pmddomingos/status/1535112033137401857), [AI move from Academia](https://twitter.com/GaryMarcus/status/1536150812795121664), [Artificial General Intelligence Is Not as Imminent as You Might Think](https://www.scientificamerican.com/article/artificial-general-intelligence-is-not-as-imminent-as-you-might-think1/)


#12, #13 Evolutionary Algorithms

- [Evolutionary Computation](http://www.evolutionarycomputation.org/slides/)
- ​[tiny_ga](https://github.com/moshesipper/tiny_ga)
- ​[NSGA II](https://engineering.purdue.edu/~sudhoff/ee630/Lecture09.pdf)​
- [Schema theorem](https://engineering.purdue.edu/~sudhoff/ee630/Lecture03.pdf)
- GP:[Koza](http://www.genetic-programming.com/c2003lecture1modified.ppt),[ Koza Tutorial,](http://www.genetic-programming.com/gecco2003tutorial.pdf) [Koza &amp; Poli](http://www.cs.bham.ac.uk/~wbl/biblio/cache/bin/cache.php?koza:2003:gpt,http___www.genetic-programming.com_jkpdf_burke2003tutorial.pdf,http://www.genetic-programming.com/jkpdf/burke2003tutorial.pdf), [Yoo, ](https://coinse.kaist.ac.kr/assets/files/teaching/cs454/cs454-slide09.pdf)[Herrmann](https://www.inf.ed.ac.uk/teaching/courses/nat/slides/nat09h.pdf)
- [tiny_gp](https://github.com/moshesipper/tiny_gp)
- [Linear GP](http://www.am.chalmers.se/~wolff/AI2/Lect05LGP.pdf) / [Cartesian GP](http://cs.ijs.si/ppsn2014/files/slides/ppsn2014-tutorial3-miller.pdf) / [Grammatical Evolution](https://web.archive.org/web/20110721124315/http:/www.grammaticalevolution.org/tutorial.pdf)
- ​[Koza's vids](https://www.youtube.com/channel/UC9MEHhji3ODbE_e66EgFkew)
- [Humies](https://www.human-competitive.org/)
- [SAFE, OMNIREP](https://drive.google.com/file/d/1fKymYCJPyd9rBmpEDgRPRe0GE7Yr3SuI/view?usp=sharing)
- [Novelty search](https://www.cs.ucf.edu/~gitars/cap6671-2010/Presentations/lehman_alife08.pdf)​​


***	

**Some Pros and Cons**

- KN Neighbors \
   ✔ Simple, No training, No assumption about data, Easy to implement, New data can be added seamlessly, Only one hyperparameter \
   ✖ Doesn't work well in high dimensions, Sensitive to noisy data, missing values and outliers, Doesn't work well with large data sets —  cost of calculating distance is high, Needs feature scaling, Doesn't work well on imbalanced data, Doesn't deal well with missing values

- Decision Tree \
   ✔ Doesn't require standardization or normalization, Easy to implement, Can handle missing values, Automatic feature selection \
   ✖ High variance, Higher training time, Can become complex, Can easily overfit

- Random Forest \
   ✔ Left-out data can be used for testing, High accuracy, Provides feature importance estimates, Can handle missing values, Doesn't require feature scaling, Good performance on imbalanced datasets, Can handle large dataset, Outliers have little impact, Less overfitting \
   ✖ Less interpretable, More computational resources, Prediction time high

- Linear Regression \
   ✔ Simple, Interpretable, Easy to Implement \
   ✖ Assumes linear relationship between features, Sensitive to outliers

- Logistic Regression \
   ✔ Doesn’t assume linear relationship between independent and dependent variables, Output can be interpreted as probability, Robust to noise \
   ✖ Requires more data, Effective when linearly separable

- Lasso Regression (L1) \
   ✔ Prevents overfitting, Selects features by shrinking coefficients to zero \
   ✖ Selected features will be biased, Prediction can be worse than Ridge

- Ridge Regression (L2) \
   ✔ Prevents overfitting \
   ✖ Increases bias, Less interpretability 

- AdaBoost \
   ✔ Fast, Reduced bias, Little need to tune \
   ✖ Vulnerable to noise, Can overfit

- Gradient Boosting \
   ✔ Good performance \
   ✖ Harder to tune hyperparameters

- XGBoost \
   ✔ Less feature engineering required, Outliers have little impact, Can output feature importance, Handles large datasets, Good model performance, Less prone to overfitting \​
   ✖ Difficult to interpret, Harder to tune as there are numerous hyperparameters

- SVM \
   ✔ Performs well in higher dimensions, Excellent when classes are separable, Outliers have less impact \
   ✖ Slow, Poor performance with overlapping classes, Selecting appropriate kernel functions can be tricky

- Naïve Bayes \
   ✔ Fast, Simple, Requires less training data, Scalable, Insensitive to irrelevant features, Good performance with high-dimensional data \
   ✖ Assumes independence of features

- Deep Learning \
  ✔ Superb performance with unstructured data (images, video, audio, text) \
  ✖ (Very) long training time, Many hyperparameters, Prone to overfitting
