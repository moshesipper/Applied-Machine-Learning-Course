# Course: Applied Machine Learning 

<img align="right" src="https://www.moshesipper.com/uploads/3/0/8/3/30831095/published/pixar-animation-studios-disney-pixar-wall-e-rubik-s-cube-wallpaper-preview2.jpg" width="200" />

**This course covers the applied/coding side of algorithmics in machine learning and deep learning, with a smidgen of evolutionary algorithms.** 

[Course Contents](#contents)	

[Lesson Plan](#lessons)

[Cheat Sheets](#cheat)

[Algorithm Pros and Cons](#pros)

[Resources: Evolutionary Algorithms, Machine Learning, Deep Learning](#resources)


***

<a name="contents">**Course Contents**</a>

- What is machine learning (ML)?
- Basics of Python programming
- Applying ML: evaluation, dataset splits, cross-validation, performance measures, bias/variance tradeoff, visualization, confusion matrix, choosing estimators, hyperparameter tuning, statistics
- Supervised learning: models, features, objectives, model training, overfitting, regularization, classification, regression, gradient descent, k nearest neighbors, linear regression, logistic regression, decision tree, random forest, adaptive boosting, gradient boosting, support vector machine, naïve Bayes
- Dimensionality reduction: principal component analysis
- Unsupervised learning: hierarchical clustering, k-means, t-SNE
- Artificial neural networks: backpropagation, deep neural network, convolutional neural network
- Evolutionary algorithms: genetic algorithm,genetic programming

***	

<a name="lessons">**Lesson plan**</a>

(![](colab.png) marks my colab notebooks)


1: Python, AI+ML Intro

- [Python Programming](https://pythonbasics.org/) ([Python](https://www.python.org/downloads/windows/), [PyCharm](https://www.jetbrains.com/pycharm/download/#section=windows)), [Pandas](https://pythonbasics.org/what-is-pandas/), [NumPy](https://numpy.org/devdocs/user/absolute_beginners.html) / [NumPy](https://www.w3schools.com/python/numpy/default.asp) ([Numba](https://numba.pydata.org/)) ([np.dot](https://numpy.org/doc/stable/reference/generated/numpy.dot.html) vs [loop example](https://colab.research.google.com/drive/1wAfDDyqYkj1izQvn7bDF9tJA4xYlDWzp?usp=sharing) ![](colab.png))
- [Computing Machinery and Intelligence](https://www.cs.mcgill.ca/~dprecup/courses/AI/Materials/turing1950.pdf)
- [Machine Learning: history, applications, recent successes](https://data-psl.github.io/lectures2020/slides/01_machine_learning_successes)
- [Data Science Infographic](https://github.com/dataprofessor/infographic) (Building an ML Model, Data Science Landscape)
- [How to avoid machine learning pitfalls](https://arxiv.org/abs/2108.02497)
- [Top 10 machine learning algorithms with their use-cases](https://medium.com/@avikumart_/top-10-machine-learning-algorithms-with-their-use-cases-f1ea2c1dfd6b)
- [21 Most Important (and Must-know) Mathematical Equations in Data Science](https://www.blog.dailydoseofds.com/p/21-most-important-and-must-know-mathematical)

2: ML Intro, Simple Example, KNN, Cross-Validation

- [Introduction to machine learning](https://data-psl.github.io/lectures2020/slides/02_intro_to_machine_learning),
  [(train/val/test)](https://glassboxmedicine.com/2019/09/15/best-use-of-train-val-test-splits-with-tips-for-medical-data/)
- [simple weather example](https://colab.research.google.com/drive/1XShD6G7sPGLXKtto4GBZPLWJoPcJEJBk?usp=sharing) ![](colab.png)
- [iris knn (map)](https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py)
- [kfold](https://colab.research.google.com/drive/1Hj17jfBbl0tYBVn6ze0YQ7xxTS5Dr1-D?usp=sharing) ![](colab.png)

3: Scikit-learn, Models, Decision Trees

- [Machine learning with scikit-learn](https://data-psl.github.io/lectures2020/slides/04_scikit_learn/#1)
- [A Minimal Example of Machine Learning (with scikit-learn)](https://medium.com/@sipper/a-minimal-example-of-machine-learning-with-scikit-learn-4e98d5dcc6e7)
- [19 Most Elegant Sklearn Tricks I Found After 3 Years of Use](https://pub.towardsai.net/19-most-elegant-sklearn-tricks-i-found-after-3-years-of-use-5bda439fa506)
- [Machine learning models](https://data-psl.github.io/lectures2020/slides/03_machine_learning_models/)
- [Toy datasets (sklearn)](https://scikit-learn.org/stable/datasets/toy_dataset.html)
- [Decision trees](https://youtu.be/_L39rN6gz7Y)
- [Decision trees](https://colab.research.google.com/drive/1wyD94nW0HFvdhCkYLLmkxdVulhZTDD-x?usp=sharing) ![](colab.png)


4: Random Forest, Linear Regression Logistic Regression

- [Machine learning models](https://data-psl.github.io/lectures2020/slides/03_machine_learning_models/)
- [Random Forests](https://youtu.be/J4Wdy0Wc_xQ)
- [Linear Regression](https://youtu.be/PaFPbb66DxQ)
- [LinearReg](https://colab.research.google.com/drive/1fCQjAiEce6hU0osLzCWo73hC9iYva3QK?usp=sharing) ![](colab.png)
- [Logistic Regression](https://youtu.be/yIYKR4sgzI8), [Logistic Regression](https://www.analyticsvidhya.com/blog/2021/07/an-introduction-to-logistic-regression/), [Cross-Entropy Loss](https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e)
- [Optimization of linear models](https://data-psl.github.io/lectures2020/slides/05_optimization_linear_models/)
- [Ridge vs. Lasso](https://www.statology.org/when-to-use-ridge-lasso-regression/), [notes on regularization in ML](https://www.linkedin.com/feed/update/urn:li:activity:7053809365169971201/)

5: AdaBoost, Gradient Boosting

- Summary: [Linear Regression](https://medium.com/analytics-vidhya/a-quick-summary-of-linear-regression-42d1dab85e3e), [Logistic Regression](https://www.analyticsvidhya.com/blog/2021/07/an-introduction-to-logistic-regression/), [LinVsLog](https://colab.research.google.com/drive/1kfMFdrVpL9NczZdKDfA_zGT0NMr3PMYS?usp=sharing) ![](colab.png), [PolynomialFeatures](https://colab.research.google.com/drive/1zjuhudzOZRCbovLwWSxYLxsJ67V7A5Dt?usp=sharing) ![](colab.png)
- [Adaptive Boosting](https://youtu.be/LsK-xG1cLYA)
- [Gradient Boosting](https://youtu.be/3CC4N4z3GJc)
- [AddGBoost](https://www.sciencedirect.com/science/article/pii/S2666827021001225), [Strong(er) Gradient Boosting](https://medium.com/@sipper/strong-er-gradient-boosting-6eb617566328)

6: XGBoost, Comparing ML algos, Gradient Descent

- Reminder: [Adaboost](https://www.cs.bgu.ac.il/~sipper/adaboost.jpg), [Gradient boost](https://www.cs.bgu.ac.il/~sipper/gradboost.jpg)
- [XGBoost](https://youtu.be/OtD8wVaFm6E)
- [Comparing supervised learning algorithms](https://www.dataschool.io/comparing-supervised-learning-algorithms/)
- [How to find the best performing Machine Learning algorithm](https://medium.com/analytics-vidhya/how-to-find-the-best-performing-machine-learning-algorithm-dc4eb4ff34b6), Boston dataset (`from sklearn.datasets import load_boston` -> [racist data destruction?](https://medium.com/@docintangible/racist-data-destruction-113e3eff54a8)) 
- [Gradient Descent](https://youtu.be/sDv4f4s2SB8) ([Least Squares](https://www.mathsisfun.com/data/least-squares-regression.html), [Least Squares](https://textbooks.math.gatech.edu/ila/least-squares.html))
- [Stochastic Gradient Descent](https://youtu.be/vMh0zPT0tLI)
- [Stochastic Gradient Descent Algorithm With Python and NumPy](https://realpython.com/gradient-descent-algorithm-python/)

7: Choosing Model, SVM, Bayes, Metrics

- [Choosing the right estimator](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
- [Support Vector Machine](https://youtu.be/efR1C6CvhmE) [(](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)[sklearn.svm.SVC)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), [Plot different SVM classifiers in the iris dataset](https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html#sphx-glr-auto-examples-svm-plot-iris-svc-py)
- [Multinomial Naive Bayes](https://youtu.be/O2L2Uv9pdDA)
- Metrics: [sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html?highlight=metrics#module-sklearn.metrics), [Confusion Matrix](https://youtu.be/Kdsp6soqA7o), 
[Sensitivity and Specificity](https://youtu.be/vP06aMoz4v8),
[Sensitivity and specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity),
 [ROC and AUC](https://youtu.be/4jRBRDbJemM) ([Online ROC Curve Calculator](https://www.rad.jhmi.edu/jeng/javarad/roc/JROCFITi.html)),[balanced accuracy](https://www.statology.org/balanced-accuracy/), [balanced accuracy](https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score), [various metrics from CM](https://en.wikipedia.org/wiki/Confusion_matrix)
- [Could machine learning fuel a reproducibility crisis in science?](https://www.nature.com/articles/d41586-022-02035-w)

8: Metrics, PCA, t-SNE, Clustering

- [ML in practice](https://info.deeplearning.ai/the-batch-one-model-for-hundreds-of-tasks-recognizing-workplace-hazards-when-data-means-danger-vision-transformer-upgrade-1)
- [balanced accuracy](https://www.statology.org/balanced-accuracy/), [balanced accuracy](https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score), [various metrics from CM](https://en.wikipedia.org/wiki/Confusion_matrix)
- [11 Different Uses of Dimensionality Reduction](https://towardsdatascience.com/11-different-uses-of-dimensionality-reduction-4325d62b4fa6)
- [PCA](https://youtu.be/FgakZw6K1QQ), [PCA](https://www.sartorius.com/en/knowledge/science-snippets/what-is-principal-component-analysis-pca-and-how-it-is-used-507186), [PCA vs LR](https://starship-knowledge.com/pca-vs-linear-regression#PCA_vs_Linear_Regression_-_How_do_they_Differ), [pca](https://colab.research.google.com/drive/1h6xLxKyEltPwsck-mJ5nQPFkMGYI8VOs?usp=sharing) ![](colab.png)
- [t-SNE](https://youtu.be/NEaUSP4YerM), [tsne](https://colab.research.google.com/drive/1vnA5iwWrjDY4AhHL_E86VLq59FwJG2s9?usp=sharing) ![](colab.png)
- [Clustering](https://www.geeksforgeeks.org/clustering-in-machine-learning/)
- [Hierarchical clustering](https://youtu.be/7xHsRkOdVwo)
- [K-means clustering](https://youtu.be/4b5d3muPQmA), [kmeans](https://colab.research.google.com/drive/1aoiM8cnS_DdNOP2njEsjWcyf6-zHMrJ1?usp=sharing) ![](colab.png)
- [From Data to Clusters: When is Your Clustering Good Enough?](https://towardsdatascience.com/from-data-to-clusters-when-is-your-clustering-good-enough-5895440a978a)

9: Hyperparameter Tuning, p-vals, t-test, Permutation Test

- [Model Parameter vs. Hyperparameter](https://www.youtube.com/watch?v=Qcgav8NmPxY&t=1224s)
- [Hyperparameter tuning](https://towardsdatascience.com/hyperparameter-tuning-explained-d0ebb2ba1d35), [Hyperparameter tuning](https://towardsdatascience.com/hyperparameter-tuning-a-practical-guide-and-template-b3bf0504f095), [Optuna](https://optuna.org/), [optuna](https://colab.research.google.com/drive/1FbG9yaUNn8EqL1NgLBBoRIx9E5EPBuIQ?usp=sharing) ![](colab.png)
- [Evaluating Hyperparameters in Machine Learning](https://medium.com/@sipper/evaluating-hyperparameters-in-machine-learning-25b7fa09362d)
- [p-values](https://youtu.be/vemZtEM63GY), [how to calculate p-values](https://youtu.be/JQc3yx0-Q9E), [p-hacking](https://youtu.be/HDCOUXE3HMM)
- [Probability is not Likelihood](https://www.youtube.com/watch?v=pYxNSUDSFH4)
- [17 Statistical Hypothesis Tests in Python](https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/)
- [t-test](https://youtu.be/0Pd3dc1GcHc),[t-test](https://youtu.be/pTmLQvMM-1M) ([t-test vs p-value)](https://askanydifference.com/difference-between-t-test-and-p-value/), [scpipy ttest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)
- ([chi-square](https://youtu.be/ZjdBM7NO7bY))
- [permutation test](https://youtu.be/GmvpsJHGCxQ) ([AddGBoost ](https://www.sciencedirect.com/science/article/pii/S2666827021001225)+ [code](https://github.com/moshesipper/AddGBoost))
- [The Permutation Test: A Statistical Test that has (More than) Survived the Test of Time](https://medium.com/@sipper/the-permutation-test-a-statistical-test-that-has-more-than-survived-the-test-of-time-c909ebe8eb92)

10+11: Neural Networks

- [Neural networks](https://youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1)
- [Neural Networks with À La Carte Selection of Activation Functions](https://drive.google.com/file/d/10wy_gFPoNLwQXHkHKAojYBxbKkB5W_da/view?usp=sharing)
- [PyTorch](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html), [PyTorch](https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e)
- [Convolution...](https://www.linkedin.com/posts/pascalbornet_artificialintelligence-ugcPost-6925288775740776448-0S-K/)
- [Implementing Image Processing Kernels from scratch using Convolution in Python
](https://medium.com/@sabribarac/implementing-image-processing-kernels-from-scratch-using-convolution-in-python-4e966e9aafaf)
- [Growth of AI computing](https://twitter.com/pmddomingos/status/1535112033137401857), [AI move from Academia](https://twitter.com/GaryMarcus/status/1536150812795121664), [Artificial General Intelligence Is Not as Imminent as You Might Think](https://www.scientificamerican.com/article/artificial-general-intelligence-is-not-as-imminent-as-you-might-think1/)
- [Tinker With a Neural Network in Your Browser](https://playground.tensorflow.org/)
- [Introduction to large language models](https://www.youtube.com/watch?v=zizonToFXDs), [Scikit-LLM](https://github.com/iryna-kondr/scikit-llm)
- [A Tiny Large Language Model (LLM), Coded, and Hallucinating](https://medium.com/@sipper/a-tiny-large-language-model-llm-coded-and-hallucinating-9a427b04eb1a)
- [Introduction to image generation (diffusion)](https://www.youtube.com/watch?v=kzxz8CO_oG4)

12+13: Evolutionary Algorithms

- [Evolutionary Computation](http://www.evolutionarycomputation.org/slides/)
- [Evolutionary Algorithms, Genetic Programming, and Learning
](https://medium.com/@sipper/evolutionary-algorithms-genetic-programming-and-learning-dfde441ad0b9)
- [Tiny GA](https://github.com/moshesipper/tiny_ga), [Tiny GP](https://github.com/moshesipper/tiny_gp), [EC-KitY](https://www.eckity.org/)
- Genetic Programming (GP): [Koza](http://www.genetic-programming.com/c2003lecture1modified.ppt), [Koza Tutorial](http://www.genetic-programming.com/gecco2003tutorial.pdf), [Koza &amp; Poli](http://www.cs.bham.ac.uk/~wbl/biblio/cache/bin/cache.php?koza:2003:gpt,http___www.genetic-programming.com_jkpdf_burke2003tutorial.pdf,http://www.genetic-programming.com/jkpdf/burke2003tutorial.pdf), [Yoo](https://coinse.kaist.ac.kr/assets/files/teaching/cs454/cs454-slide09.pdf), [Herrmann](https://www.inf.ed.ac.uk/teaching/courses/nat/slides/nat09h.pdf), [Koza's vids](https://www.youtube.com/channel/UC9MEHhji3ODbE_e66EgFkew)
- [Multi-Objective Optimization](https://engineering.purdue.edu/~sudhoff/ee630/Lecture09.pdf)
- [Schema theorem](https://engineering.purdue.edu/~sudhoff/ee630/Lecture03.pdf)
- [Linear GP](http://www.am.chalmers.se/~wolff/AI2/Lect05LGP.pdf)
- [Cartesian GP](http://cs.ijs.si/ppsn2014/files/slides/ppsn2014-tutorial3-miller.pdf)
- [Grammatical Evolution](https://web.archive.org/web/20110721124315/http:/www.grammaticalevolution.org/tutorial.pdf)
- [New Pathways in Coevolutionary Computation](https://drive.google.com/file/d/1fKymYCJPyd9rBmpEDgRPRe0GE7Yr3SuI/view?usp=sharing)
- [Novelty search](https://www.cs.ucf.edu/~gitars/cap6671-2010/Presentations/lehman_alife08.pdf)
- [Humies](https://www.human-competitive.org/)
- [Building Activation Functions for Deep Networks](https://medium.com/@sipper/building-activation-functions-for-deep-networks-82c2a9c9cc1f)
- [Evolutionary Adversarial Attacks on Deep Networks
](https://medium.com/@sipper/evolutionary-adversarial-attacks-on-deep-networks-ff622b8e15e5)

***	

<a name="cheat">**Cheat Sheets**</a>
*   [Machine Learning Glossary](https://ml-cheatsheet.readthedocs.io/en/latest/index.html)
*	[Cheat Sheets for Machine Learning and Data Science](https://sites.google.com/view/datascience-cheat-sheets)
*	[The Illustrated Machine Learning Website](https://illustrated-machine-learning.github.io/)


***	

<a name="pros">**Algorithm Pros and Cons**</a>

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
   ✔ Less feature engineering required, Outliers have little impact, Can output feature importance, Handles large datasets, Good model performance, Less prone to overfitting \
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


***	

<a name="resources">**Resources: Evolutionary Algorithms, Machine Learning, Deep Learning**</a>

Reads / Vids

*   [Genetic and Evolutionary Algorithms and Programming](https://drive.google.com/file/d/0B6G3tbmMcpR4WVBTeDhKa3NtQjg/view?usp=sharing)
*   [גיא כתבי - אלגוריתמים אבולוציוניים](https://www.youtube.com/watch?v=XPx-a6MVne8&ab_channel=guykatabi) (YouTube) \[גיא בוגר הקורס שלי: _אלגוריתמים אבולוציוניים וחיים מלאכותיים_\]
*	[Choosing Representation, Mutation, and Crossover in Genetic Algorithms
](https://ieeexplore.ieee.org/document/9942691/interactive)
*   [Introduction to Evolutionary Computing](http://www.evolutionarycomputation.org/) (course/book slides)
*   [John Koza Genetic Programming](https://www.youtube.com/channel/UC9MEHhji3ODbE_e66EgFkew) (YouTube)
*   [Some reports in the popular press](https://www.moshesipper.com/publications.html)
*   [Why video games are essential for inventing artificial intelligence](https://togelius.blogspot.co.il/2016/01/why-video-games-are-essential-for.html)
*   [Biologic or “By Ole Logic”](http://www.moshesipper.com/blog/biologic-or-by-ole-logic)
*   [26 Top Machine Learning Interview Questions and Answers: Theory Edition](https://www.blog.confetti.ai/post/26-top-machine-learning-interview-questions-and-answers-theory)
*   [10 Popular Machine Learning Algorithms In A Nutshell](https://www.theinsaneapp.com/2021/11/machine-learning-algorithms-for-beginners.html)
*   [StatQuest with Josh Starmer](https://www.youtube.com/user/joshstarmer)
*   [Machine learning preparatory week @PSL](https://data-psl.github.io/lectures2020/)
*   [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome) (coursera)
*   [ROC-AUC](https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/)
*   [Tinker With a Neural Network in Your Browser](https://playground.tensorflow.org/)
*   [Common Machine Learning Algorithms for Beginners](https://www.dezyre.com/article/common-machine-learning-algorithms-for-beginners/202)
*  [ML YouTube Courses](https://github.com/dair-ai/ML-YouTube-Courses)
*	[Machine Learning Essentials for Biomedical Data Science: Introduction and ML Basics](https://www.youtube.com/watch?v=Qcgav8NmPxY&list=PLafPhSv1OSDfEqFsBnurxzJbcwZSJA8X4)
*	[GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/)
*	[Some Techniques To Make Your PyTorch Models Train (Much) Faster](https://sebastianraschka.com/blog/2023/pytorch-faster.html)
  

Books (🡇 means free to download)

*   M. Sipper, _[Evolved to Win](https://www.moshesipper.com/evolved-to-win.html)_, Lulu, 2011 🡇
*   M. Sipper, _[Machine Nature: The Coming Age of Bio-Inspired Computing](https://www.moshesipper.com/machine-nature-the-coming-age-of-bio-inspired-computing.html)_, McGraw-Hill, New York, 2002
*   A.E. Eiben and J.E. Smith, [_Introduction to Evolutionary Computing_](http://www.cs.vu.nl/~gusz/ecbook/ecbook.html), Springer, 1st edition, 2003, Corr. 2nd printing, 2007
*   R. Poli, B. Langdon, & N. McPhee, [_A Field Guide to Genetic Programming_](http://www.gp-field-guide.org.uk/), 2008 🡇
*   J. Koza, [_Genetic Programming: On the Programming of Computers by Means of Natural Selection_](http://www.genetic-programming.org/gpbook1toc.html), MIT Press, Cambridge, MA, 1992.
*   S. Luke, [_Essentials of Metaheuristics_](http://cs.gmu.edu/~sean/book/metaheuristics/), 2013 🡇
*   A. Geron, [Hands On Machine Learning with Scikit Learn and TensorFlow](https://github.com/yanshengjia/ml-road/blob/master/resources/Hands%20On%20Machine%20Learning%20with%20Scikit%20Learn%20and%20TensorFlow.pdf), 2017 🡇
*   G. James, D. Witten, T. Hastie, R. Tibshirani, [An Introduction to Statistical Learning](https://www.statlearning.com/), 2nd edition, 2021 🡇
*   J. VanderPlas, [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
*   K. Reitz, [The Hitchhiker’s Guide to Python](https://docs.python-guide.org/)
*   M. Nielsen, [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)
*   Z. Michalewicz & D.B. Fogel, [_How to Solve It: Modern Heuristics_](https://www.springer.com/computer/theoretical+computer+science/foundations+of+computations/book/978-3-540-22494-5), 2nd ed. Revised and Extended, 2004
*   Z. Michalewicz. [_Genetic Algorithms + Data Structures = Evolution Programs_](http://www.springeronline.com/sgw/cda/frontpage/0,10735,5-40109-22-1430991-0,00.html). Springer-Verlag, Berlin, 3rd edition, 1996
*   D. Floreano & C. Mattiussi, [_Bio-Inspired Artificial Intelligence: Theories, Methods, and Technologies_](http://baibook.epfl.ch/), MIT Press, 2008
*   A. Tettamanzi & M. Tomassini, [_Soft Computing: Integrating Evolutionary, Neural, and Fuzzy Systems_](https://www.springer.com/computer/theoretical+computer+science/book/978-3-540-42204-4), Springer-Verlag, Heidelberg, 2001
*   M. Mohri, A. Rostamizadeh, and A. Talwalka, [Foundations of Machine Learning](https://www.dropbox.com/s/4fij1xrclwjdu5y/Foundations%20of%20Machine%20Learning%2C%20Mohri%202012.pdf?dl=0), MIT Press, 2012 🡇
*	Simon J.D. Prince, [Understanding Deep Learning](https://udlbook.github.io/udlbook/), MIT Press, 2023 🡇
    

Software

*	[EC-KitY: Evolutionary Computation Tool Kit in Python with Seamless Machine Learning Integration](https://www.eckity.org/)
*   [gplearn: Genetic Programming in Python, with a scikit-learn inspired and compatible API](https://gplearn.readthedocs.io/en/stable/#)
*   [LEAP: Library for Evolutionary Algorithms in Python](https://github.com/AureumChaos/LEAP)
*   [DEAP: Distributed Evolutionary Algorithms in Python](https://deap.readthedocs.io/en/master/)
*   [Swarm Intelligence in Python (Genetic Algorithm, Particle Swarm Optimization, Simulated Annealing, Ant Colony Algorithm, Immune Algorithm, Artificial Fish Swarm Algorithm in Python)](https://github.com/guofei9987/scikit-opt)
*   [Scikit-learn: Machine Learning in Python](https://scikit-learn.org/stable/index.html)
*   [Mlxtend (machine learning extensions)](https://rasbt.github.io/mlxtend/) 
*   [PyTorch (deep networks)](https://pytorch.org/)
*   [Best-of Machine Learning with Python](https://github.com/ml-tooling/best-of-ml-python)
*   [Fundamental concepts of PyTorch through self-contained examples](https://github.com/jcjohnson/pytorch-examples)
*   [Faster Python calculations with Numba](https://pythonspeed.com/articles/numba-faster-python)  
    

Datasets

*   [Tabular & cleaned (PMLB)](https://github.com/EpistasisLab/pmlb)
*   [By domain](https://www.datasetlist.com/)
*  [By application](https://github.com/awesomedata/awesome-public-datasets)
*   [Search engine](https://datasetsearch.research.google.com/)
*   [Kaggle competitions](https://www.kaggle.com/datasets)
*   [OpenML](https://www.openml.org/)
*   [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php)
*  [Image Databases](https://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm)
*   [AWS Open Data Registry](https://registry.opendata.aws/)
*  [Wikipedia ML Datasets](https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research)
*   [The Big Bad NLP Database](https://datasets.quantumstat.com/)
*  [Datasets for Machine Learning and Deep Learning](https://sebastianraschka.com/blog/2021/ml-dl-datasets.html)
*   [Browse State-of-the-Art](https://paperswithcode.com/sota)  
    