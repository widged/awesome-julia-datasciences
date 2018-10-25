# awesome-julia-datasciences
Resources about Julia for DataSciences / Machine Learning

I really find it easier to maintain bookmarks in a datasheet format:

https://airtable.com/invite/l?inviteId=invxOmYmPMGSkckQn&inviteToken=0b986a9651139333f3715b0f4959e15899b10d4fb9dadaa03870dcea4b70784f

Overview below, but the list will be kept up to date on airtable and converted to an awesome format from time to tim

## Table of Contents

<!-- MarkdownTOC depth=4 -->

- [General-Purpose Machine Learning](#julia-general-purpose)
- [Natural Language Processing](#julia-nlp)
- [Data Analysis / Data Visualization](#julia-data-analysis)
- [Misc Stuff / Presentations](#julia-misc)

<!-- /MarkdownTOC -->

<a name="apl"></a>
## APL

 * [Julia](http://julialang.org) – high-level, high-performance dynamic programming language for technical computing
 * [IJulia](https://github.com/JuliaLang/IJulia.jl) – a Julia-language backend combined with the Jupyter interactive environment

<a name="julia-general-purpose"></a>
### General-Purpose Machine Learning

* [MachineLearning](https://github.com/benhamner/MachineLearning.jl) - Julia Machine Learning library.
* [MLBase](https://github.com/JuliaStats/MLBase.jl) - A set of functions to support the development of machine learning algorithms.
* [PGM](https://github.com/JuliaStats/PGM.jl) - A Julia framework for probabilistic graphical models.
* [DA](https://github.com/trthatcher/DiscriminantAnalysis.jl) - Julia package for Regularized Discriminant Analysis.
* [Regression](https://github.com/lindahua/Regression.jl) - Algorithms for regression analysis (e.g. linear regression and logistic regression).
* [Local Regression](https://github.com/JuliaStats/Loess.jl) - Local regression, so smooooth!.
* [Naive Bayes](https://github.com/nutsiepully/NaiveBayes.jl) - Simple Naive Bayes implementation in Julia.
* [Mixed Models](https://github.com/dmbates/MixedModels.jl) - A Julia package for fitting (statistical) mixed-effects models.
* [Simple MCMC](https://github.com/fredo-dedup/SimpleMCMC.jl) - basic mcmc sampler implemented in Julia.
* [Distance](https://github.com/JuliaStats/Distance.jl) - Julia module for Distance evaluation.
* [Decision Tree](https://github.com/bensadeghi/DecisionTree.jl) - Decision Tree Classifier and Regressor.
* [Neural](https://github.com/compressed/BackpropNeuralNet.jl) - A neural network in Julia.
* [MCMC](https://github.com/doobwa/MCMC.jl) - MCMC tools for Julia.
* [Mamba](https://github.com/brian-j-smith/Mamba.jl) - Markov chain Monte Carlo (MCMC) for Bayesian analysis in Julia.
* [GLM](https://github.com/JuliaStats/GLM.jl) - Generalized linear models in Julia.
* [Gaussian Processes](https://github.com/STOR-i/GaussianProcesses.jl) - Julia package for Gaussian processes.
* [Online Learning](https://github.com/lendle/OnlineLearning.jl)
* [GLMNet](https://github.com/simonster/GLMNet.jl) - Julia wrapper for fitting Lasso/ElasticNet GLM models using glmnet.
* [Clustering](https://github.com/JuliaStats/Clustering.jl) - Basic functions for clustering data: k-means, dp-means, etc.
* [SVM](https://github.com/JuliaStats/SVM.jl) - SVM's for Julia.
* [Kernel Density](https://github.com/JuliaStats/KernelDensity.jl) - Kernel density estimators for julia.
* [Dimensionality Reduction](https://github.com/JuliaStats/DimensionalityReduction.jl) - Methods for dimensionality reduction.
* [NMF](https://github.com/JuliaStats/NMF.jl) - A Julia package for non-negative matrix factorization.
* [ANN](https://github.com/EricChiang/ANN.jl) - Julia artificial neural networks.
* [Mocha](https://github.com/pluskid/Mocha.jl) - Deep Learning framework for Julia inspired by Caffe.
* [XGBoost](https://github.com/dmlc/XGBoost.jl) - eXtreme Gradient Boosting Package in Julia.
* [ManifoldLearning](https://github.com/wildart/ManifoldLearning.jl) - A Julia package for manifold learning and nonlinear dimensionality reduction.
* [MXNet](https://github.com/dmlc/mxnet) - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Go, Javascript and more.
* [Merlin](https://github.com/hshindo/Merlin.jl) - Flexible Deep Learning Framework in Julia.
* [ROCAnalysis](https://github.com/davidavdav/ROCAnalysis.jl) - Receiver Operating Characteristics and functions for evaluation probabilistic binary classifiers.
* [GaussianMixtures](https://github.com/davidavdav/GaussianMixtures.jl) - Large scale Gaussian Mixture Models.
* [ScikitLearn](https://github.com/cstjean/ScikitLearn.jl) - Julia implementation of the scikit-learn API.
* [Knet](https://github.com/denizyuret/Knet.jl) - Koç University Deep Learning Framework.

<a name="julia-nlp"></a>
#### Natural Language Processing

* [Topic Models](https://github.com/slycoder/TopicModels.jl) - TopicModels for Julia.
* [Text Analysis](https://github.com/johnmyleswhite/TextAnalysis.jl) - Julia package for text analysis.


<a name="julia-data-analysis"></a>
### Data Analysis / Data Visualization

* [Graph Layout](https://github.com/IainNZ/GraphLayout.jl) - Graph layout algorithms in pure Julia.
* [LightGraphs](https://github.com/JuliaGraphs/LightGraphs.jl) - Graph modeling and analysis.
* [Data Frames Meta](https://github.com/JuliaStats/DataFramesMeta.jl) - Metaprogramming tools for DataFrames.
* [Julia Data](https://github.com/nfoti/JuliaData) - library for working with tabular data in Julia.
* [Data Read](https://github.com/WizardMac/ReadStat.jl) - Read files from Stata, SAS, and SPSS.
* [Hypothesis Tests](https://github.com/JuliaStats/HypothesisTests.jl) - Hypothesis tests for Julia.
* [Gadfly](https://github.com/GiovineItalia/Gadfly.jl) - Crafty statistical graphics for Julia.
* [Stats](https://github.com/JuliaStats/Stats.jl) - Statistical tests for Julia.
* [RDataSets](https://github.com/johnmyleswhite/RDatasets.jl) - Julia package for loading many of the data sets available in R.
* [DataFrames](https://github.com/JuliaStats/DataFrames.jl) - library for working with tabular data in Julia.
* [Distributions](https://github.com/JuliaStats/Distributions.jl) - A Julia package for probability distributions and associated functions.
* [Data Arrays](https://github.com/JuliaStats/DataArrays.jl) - Data structures that allow missing values.
* [Time Series](https://github.com/JuliaStats/TimeSeries.jl) - Time series toolkit for Julia.
* [Sampling](https://github.com/lindahua/Sampling.jl) - Basic sampling algorithms for Julia.

<a name="julia-misc"></a>
### Misc Stuff / Presentations

* [DSP](https://github.com/JuliaDSP/DSP.jl) - Digital Signal Processing (filtering, periodograms, spectrograms, window functions).
* [JuliaCon Presentations](https://github.com/JuliaCon/presentations) - Presentations for JuliaCon.
* [SignalProcessing](https://github.com/davidavdav/SignalProcessing.jl) - Signal Processing tools for Julia.
* [Images](https://github.com/JuliaImages/Images.jl) - An image library for Julia.
    
