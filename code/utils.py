import os
import numpy as np
# import scipy.interpolate as interp
import matplotlib.pyplot as plt
# import pandas as pd
import seaborn as sns
import json
from sklearn.model_selection import KFold

# File with auxiliary functions used by all notebooks
def lorentzian( x, x0, a, gam ):
    return a * gam**2 / ( gam**2 + ( x - x0 )**2)
    #define a lorentzian describing the whole spectrum

def spectrum(x,ufreqs,uints,g,xstep):
    total_spec=np.zeros(x.size)
    for f in ufreqs:
        f_index=list(ufreqs).index(f)
        total_spec+=lorentzian(x,f,uints[f_index],g)
    total_spec=total_spec/np.max(total_spec) #Normalization such that max feature set to 1
    return total_spec

def natom_spectrum(x,ufreqs,uints,g,xstep,natoms):
    total_spec=np.zeros(x.size)
    for f in ufreqs:
        f_index=list(ufreqs).index(f)
        total_spec+=lorentzian(x,f,uints[f_index],g)
    total_spec=total_spec/natoms
    return total_spec

def def_angle_labels(input_angles,ncats):
    angle_labels=[]
    if ncats==3:
        for angle in input_angles: 
            #3 categories: 
            if angle <= 10.:
                angletag='LE10'
            elif angle > 10. and angle <= 20.:  
                angletag='G10LE20'
            elif angle > 20.:  
                angletag='G20'      
            angle_labels.append(angletag)
    elif ncats==6:
        for angle in input_angles:
            if angle <= 5.:
                angletag='LE5'
            elif angle > 5. and angle <= 10.:  
                angletag='G5LE10'
            elif angle > 10. and angle <= 15.:  
                angletag='G10LE15'
            elif angle > 15. and angle <= 20.:  
                angletag='G15LE20'
            elif angle > 20. and angle <= 25.:  
                angletag='G20LE25'
            elif angle > 25.:  
                angletag='G25'  
            angle_labels.append(angletag)
    elif ncats==10:
        for angle in input_angles:
            if angle <= 3.:
                angletag='LE3'
            elif angle > 3. and angle <= 6.:  
                angletag='G3LE6'
            elif angle > 6. and angle <= 9.:  
                angletag='G6LE9'
            elif angle > 9. and angle <= 12.:  
                angletag='G9LE12'
            elif angle > 12. and angle <= 15.:  
                angletag='G12LE15'
            elif angle > 15. and angle <= 18.:  
                angletag='G15LE18'
            elif angle > 18. and angle <= 21.:  
                angletag='G18LE21'
            elif angle > 21. and angle <= 24.:  
                angletag='G21LE24'
            elif angle > 24. and angle <= 27.:  
                angletag='G24LE27'
            elif angle > 27.:  
                angletag='G27'  
            angle_labels.append(angletag)
            
    elif ncats==30:
        testangle=np.arange(1,30,1)
        for angle in input_angles:
            if angle <= 1.:
                angletag='LE1'
            elif angle > 29.:  
                angletag='G29'
            
            for k in range(testangle.shape[0]-1):
                if angle>testangle[k] and angle<=testangle[k+1]:
                    angletag='G'+str(testangle[k])+'LE'+str(testangle[k+1])
            
            angle_labels.append(angletag)
        
    else:
        print('ERROR: Number of categories not supported')
    return angle_labels

#Function for plotting the confusion matrix
def plotConfusion(model, features, labels, labelNames):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    sns.set(font_scale=1.3)
    mat = confusion_matrix(labels, model.predict(features))
    sns.heatmap(mat.T,
        square=True, annot=True, cmap='Blues',
        xticklabels=labelNames, yticklabels=labelNames)
    plt.xlabel('True labels',fontsize=14)
    plt.ylabel('Predicted labels',fontsize=14)
    plt.tick_params(axis='both', which='both',labelsize=14)
    plt.tight_layout()
    return plt

def plotConfusionCV(model, features, labels,labelNames):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    sns.set(font_scale=1.3)
    #Sum confusion matrices over splits:
    mat = []
    kf = KFold(3, random_state=0, shuffle=True)
    for selTrain,selTest in kf.split(features):
        model.fit(features[selTrain], np.array(labels)[selTrain])
        mat.append(
            confusion_matrix(np.array(labels)[selTest], 
                model.predict(features[selTest]), labels=labelNames ) )
    mat = np.sum(np.array(mat), axis=0)
    sns.heatmap(mat.T, #note transpose, similar to imshow()
        square=True, annot=True, cmap='BuPu',
        xticklabels=labelNames, yticklabels=labelNames)
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.tight_layout()
    return plt

#Predict and plot input vs predictions:
def plot_prediction(features,target,model):
    Fit = model.predict(features)
    
    fig = plt.figure(figsize=plt.figaspect(1.), dpi=100, facecolor='w', edgecolor='k')
    plt.scatter(target, Fit, marker='x',s=20.)
    plt.xlabel(r'Input angle [$^{\circ}$]',fontsize=13)
    plt.ylabel(r'Predicted angle [$^{\circ}$]',fontsize=13)
    plt.tick_params(axis='both', which='both',labelsize=13)
    
    #Add perfect fit line for reference:
    dataRange = [0,30]
    plt.plot(dataRange, dataRange, 'k')
    plt.axis('square')
    plt.xlim(dataRange)
    plt.ylim(dataRange)
    s=r'R$^2$: {:.2f}'.format(model.score(features,target))
    plt.text(3.5, 25., s, fontsize=13)
    s=r'MAE: {:.2f}'.format(np.mean(np.abs(target-Fit)))
    plt.text(3.5, 22., s, fontsize=13)
    print('Model score on test data:',model.score(features,target))
    #Report accuracy (Mean Absolute Error):
    print('Accuracy (MAE):', 
          np.mean(np.abs(target-Fit)))
    print('Accuracy (RMSE):', 
          np.sqrt(np.mean((target-Fit)**2)))
    plt.tight_layout()
    return plt



from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, ylim=None, xlim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5),set_xticks=None,set_yticks=None):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure(figsize=plt.figaspect(1.),dpi=100)
    
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    train_scores_max = np.max(train_scores, axis=1)
    train_scores_min = np.min(train_scores, axis=1)
    
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    test_scores_max = np.max(test_scores, axis=1)
    test_scores_min = np.min(test_scores, axis=1)
    
    if set_xticks is not None:
        plt.xticks(set_xticks)
    if set_yticks is not None:
        plt.yticks(set_yticks)

    plt.grid()

    plt.fill_between(train_sizes,train_scores_min,
                     train_scores_max, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_min,
                     test_scores_max, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 's-', color="r",
             label="Training")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="CV")

    plt.legend(loc="lower right")
    
    plt.tight_layout()
    return plt


def get_regression_metrics(model, features_train, target_train, kfolds):
    from sklearn.model_selection import cross_validate

    test_scores=cross_validate(model, features_train, target_train, cv=kfolds,
                           scoring=['r2','neg_mean_absolute_error','neg_root_mean_squared_error'])

    print('Average R2 +- std')
    print(np.round(np.mean(test_scores['test_r2']),2),'$\pm$',np.round(np.std(test_scores['test_r2']),2))
    print('MAE +- std')
    print(np.round(np.mean(test_scores['test_neg_mean_absolute_error']),2),
                 '$\pm$',np.round(np.std(test_scores['test_neg_mean_absolute_error']),2))
    print('RMSE +- std')
    print(np.round(np.mean(test_scores['test_neg_root_mean_squared_error']),2),
          '$\pm$',np.round(np.std(test_scores['test_neg_root_mean_squared_error']),2))
                       
    return test_scores