from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state  
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

RANDOMSTATE = 28
TESTSIZE = 0.2

class RunState:
    PASS = 0
    SHOW = 1
    SAVE = 2

class Classifier:
    DECISIONTREE = 0
    RANDOMFOREST = 1
    BOOSTEDDECISIONTREE = 2

def main():
    matrix = np.loadtxt("spambase_augmented.csv", delimiter=",")
    x = matrix[:, :-1]
    y = matrix[:, -1]

    # Keep shuffle the same for all tests
    x, y = shuffle(x, y, random_state=RANDOMSTATE) # pyright: ignore (Fake type issue)

    decision_tree(x, y)
    random_forest(x, y)
    adaboost_decision_tree(x, y)
    kfcv(x, y)


# k-fold cross validation
def kfcv(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TESTSIZE, random_state=RANDOMSTATE, shuffle=False)

    tuned_rf_estimator = tune_random_forest(x_train, y_train)
    tuned_adt_estimator = tune_adaboost(x_train, y_train)

    rf_clf = RandomForestClassifier(n_estimators=tuned_rf_estimator, max_features='sqrt', criterion='gini', random_state=RANDOMSTATE)
    rf_clf.fit(x_train, y_train)
    rf_score = rf_clf.score(x_test, y_test)

    base_clf = tree.DecisionTreeClassifier(random_state=RANDOMSTATE)
    adt_clf = AdaBoostClassifier(base_clf, n_estimators=tuned_adt_estimator, random_state=RANDOMSTATE)
    adt_clf.fit(x_train, y_train)
    adt_score = adt_clf.score(x_test, y_test)

    print(f"Tuned Random Forest Score: {rf_score}")
    print(f"AdaBoost Decision Tree Score: {adt_score}")


def tune_random_forest(x, y):
    n_estimators = [i*50 for i in range(11)]

    best_n_estimator = 0
    best_score = 0
    for n_estimator in n_estimators:
        clf = RandomForestClassifier(n_estimators=n_estimator, max_features='sqrt', criterion='gini', random_state=RANDOMSTATE)
        score = kf_validate(clf, x, y)
        if score > best_score:
            best_score = score
            best_n_estimator = n_estimator

    return best_n_estimator

def tune_adaboost(x, y):
    n_estimators = [i*50 for i in range(11)]

    best_n_estimator = 0 
    best_score = 0
    for n_estimator in n_estimators:
        base_clf = tree.DecisionTreeClassifier(random_state=RANDOMSTATE)
        clf = AdaBoostClassifier(base_clf, n_estimators=n_estimator, random_state=RANDOMSTATE)
        score = kf_validate(clf, x, y)
        if score > best_score:
            best_score = score
            best_n_estimator = n_estimator

    return best_n_estimator


def kf_fold(data, k):
    folds = []
    flen = len(data) // k
    indices = np.arange(len(data))
    for i in range(k):
        test_indices = data[i*flen:(i+1)*flen]
        train_i = np.delete(indices, np.arange(i * flen, (i + 1) * flen))
        folds.append((train_i, test_indices))

    return folds

def kf_validate(clf, x, y):
    kf_indices = kf_fold(x, 5)
    scores = []
    for train_i, test_i in kf_indices:
        x_train, x_test = x[train_i], x[test_i]
        y_train, y_test = y[train_i], y[test_i]
        clf.fit(x_train, y_train)
        train_accuracy = clf.score(x_train, y_train)
        scores.append(train_accuracy)

    return np.mean(scores)


def decision_tree(x, y):
    dt_plot_errors_vs_training_set_size(x, y, RunState.PASS)
    dt_plot_errors_vs_criterion(x, y, RunState.PASS)
    dt_plot_errors_vs_max_depth(x, y, RunState.PASS)

def dt_plot_errors_vs_training_set_size(x, y, rstate):
    if rstate == RunState.PASS:
       return 

    train_errors = []
    test_errors = []
    for i in range(1, 9 + 1):
        training_set_percent = i/10

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=training_set_percent, random_state=RANDOMSTATE, shuffle=False)

        train_error, test_error = calc_min_error_with_prune(x_train, y_train, x_test, y_test)

        print(f"training set size percent: {training_set_percent * 100}")
        print_error_data(train_error, test_error)
        train_errors.append(train_error)
        test_errors.append(test_error)

    x_data = [i*10 for i in range(1, 9 + 1)]
    xlabel = "Training Set Size %"
    title = "Training and Test Error vs. Training Set %"
    plot_name = "Decision-Tree-Training-Test-Error-vs-Training-Set-Size.png"
    plot(x_data, xlabel, title, train_errors, test_errors, plot_name, rstate)

def dt_plot_errors_vs_criterion(x, y, rstate):
    if rstate == RunState.PASS:
       return 

    train_errors = []
    test_errors = []
    criterion = ["gini", "entropy", "log_loss"] 
    for c in criterion:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TESTSIZE, random_state=RANDOMSTATE, shuffle=False)

        train_error, test_error = calc_min_error_with_prune(x_train, y_train, x_test, y_test, criterion=c)

        print(f"criterion: {criterion}")
        print_error_data(train_error, test_error)
        train_errors.append(train_error)
        test_errors.append(test_error)

    x_data = criterion
    xlabel = "Criterion"
    title = "Training and Test Error vs. Criterion"
    plot_name = "Decision-Tree-Training-Test-Error-vs-Criterion.png" 
    plot(x_data, xlabel, title, train_errors, test_errors, plot_name, rstate)

def dt_plot_errors_vs_max_depth(x, y, rstate):
    if rstate == RunState.PASS:
       return 

    train_errors = []
    test_errors = []
    depths = [i for i in range(1,31)] 
    for depth in depths:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TESTSIZE, random_state=RANDOMSTATE, shuffle=False)

        train_error, test_error = calc_min_error_with_prune(x_train, y_train, x_test, y_test, max_depth=depth)

        print(f"Max Depth: {depth}")
        print_error_data(train_error, test_error)
        train_errors.append(train_error)
        test_errors.append(test_error)

    x_data = depths
    xlabel = "Depth"
    title = "Training and Test Error vs. Max Depth"
    plot_name = "Decision-Tree-Training-Test-Error-vs-Max-Depth.png" 
    plot(x_data, xlabel, title, train_errors, test_errors, plot_name, rstate)

def calc_min_error_with_prune(x_train, y_train, x_test, y_test, **classifier_params):
    clf = tree.DecisionTreeClassifier(random_state=RANDOMSTATE)

    path = clf.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas = path.ccp_alphas

    best_train_alpha = None
    best_test_alpha = None
    min_train_error = float("inf")
    min_test_error = float("inf")
    for alpha in ccp_alphas:
        pruned_clf = tree.DecisionTreeClassifier(random_state=RANDOMSTATE, ccp_alpha=alpha, **classifier_params)
        pruned_clf.fit(x_train, y_train)
        train_error, test_error = calculate_errors(x_train, x_test, y_train, y_test, pruned_clf)

        print(f"Alpha: {alpha}")
        print_error_data(train_error, test_error)
        
        if train_error < min_train_error :
            min_train_error = train_error 
            best_train_alpha = alpha

        if test_error < min_test_error :
            min_test_error = test_error 
            best_test_alpha = alpha

    print(f"Best train pruning alpha: {best_train_alpha}")
    print(f"Best test pruning alpha: {best_test_alpha}")
    print()
    return (min_train_error, min_test_error)

def random_forest(x, y):
    rf_plot_errors_vs_training_set_size(x, y, RunState.PASS)
    rf_plot_errors_vs_forest_size(x, y, RunState.PASS)
    rf_plot_errors_vs_min_samples_split(x, y, RunState.PASS)
    pass

def rf_plot_errors_vs_training_set_size(x, y, rstate):
    if rstate == RunState.PASS:
       return 

    train_errors = []
    test_errors = []
    for i in range(1, 9 + 1):
        training_set_percent = i/10
        clf = RandomForestClassifier(random_state=RANDOMSTATE)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=training_set_percent, random_state=RANDOMSTATE, shuffle=False)

        clf.fit(x_train, y_train)

        train_error, test_error = calculate_errors(x_train, x_test, y_train, y_test, clf)

        print(f"training set size percent: {training_set_percent * 100}")
        print_error_data(train_error, test_error)

        train_errors.append(train_error)
        test_errors.append(test_error)

    x_data = [i*10 for i in range(1, 9 + 1)]
    xlabel = "Training Set Size %"
    title = "Training and Test Error vs. Training Set %"
    plot_name = "Random-Forest-Training-Test-Error-vs-Training-Set-Size.png"
    plot(x_data, xlabel, title, train_errors, test_errors, plot_name, rstate)

def rf_plot_errors_vs_forest_size(x, y, rstate):
    if rstate == RunState.PASS:
       return 

    train_errors = []
    test_errors = []
    forest_sizes = [i for i in range(1,300)]
    for fsize in forest_sizes:
        clf = RandomForestClassifier(random_state=RANDOMSTATE, n_estimators=fsize)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TESTSIZE, random_state=RANDOMSTATE, shuffle=False)

        clf.fit(x_train, y_train)

        train_error, test_error = calculate_errors(x_train, x_test, y_train, y_test, clf)

        print(f"forest size: {fsize}")
        print_error_data(train_error, test_error)

        train_errors.append(train_error)
        test_errors.append(test_error)

    x_data = forest_sizes
    xlabel = "Forest Size"
    title = "Training and Test Error vs. Forest Size"
    plot_name = "Random-Forest-Training-Test-Error-vs-Forest-Size.png"
    plot(x_data, xlabel, title, train_errors, test_errors, plot_name, rstate)

def rf_plot_errors_vs_min_samples_split(x, y, rstate):
    if rstate == RunState.PASS:
       return 

    train_errors = []
    test_errors = []
    min_samples = [i for i in range(2,150)]
    for min_sample in min_samples:
        clf = RandomForestClassifier(random_state=RANDOMSTATE, min_samples_split=min_sample)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TESTSIZE, random_state=RANDOMSTATE, shuffle=False)

        clf.fit(x_train, y_train)

        train_error, test_error = calculate_errors(x_train, x_test, y_train, y_test, clf)

        print(f"min sample: {min_sample}")
        print_error_data(train_error, test_error)

        train_errors.append(train_error)
        test_errors.append(test_error)

    x_data = min_samples 
    xlabel = "Min Samples"
    title = "Training and Test Error vs. Min Samples Split"
    plot_name = "Random-Forest-Training-Test-Error-vs-Min-Samples-Split.png"
    plot(x_data, xlabel, title, train_errors, test_errors, plot_name, rstate)

def adaboost_decision_tree(x, y):
    # training/test error curves vs training set size
    adt_plot_errors_vs_training_set_size(x, y, RunState.PASS)
    adt_plot_errors_vs_number_of_iterations(x, y, RunState.PASS)
    adt_plot_errors_vs_inner_max_depth(x, y, RunState.PASS)
    pass

def adt_plot_errors_vs_training_set_size(x, y, rstate):
    if rstate == RunState.PASS:
       return 

    train_errors = []
    test_errors = []
    for i in range(1, 9 + 1):
        training_set_percent = i/10

        dt = tree.DecisionTreeClassifier(max_depth=1)
        clf = AdaBoostClassifier(dt, random_state=RANDOMSTATE)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=training_set_percent, random_state=RANDOMSTATE, shuffle=False)

        clf.fit(x_train, y_train)

        train_error, test_error = calculate_errors(x_train, x_test, y_train, y_test, clf)

        print(f"Training Set Size: {training_set_percent * 100}")
        print_error_data(train_error, test_error)

        train_errors.append(train_error)
        test_errors.append(test_error)

    x_data = [i*10 for i in range(1, 9 + 1)]
    xlabel = "Training Set Size %"
    title = "Training and Test Error vs. Training Set %"
    plot_name = "AdaBoost-Training-Test-Error-vs-Training-Set-Size.png"
    plot(x_data, xlabel, title, train_errors, test_errors, plot_name, rstate)

def adt_plot_errors_vs_number_of_iterations(x, y, rstate):
    if rstate == RunState.PASS:
       return 

    train_errors = []
    test_errors = []

    iterations = [i for i in range(1, 100)]
    for num_iterations in iterations:
        dt = tree.DecisionTreeClassifier(max_depth=1)
        clf = AdaBoostClassifier(dt, random_state=RANDOMSTATE, n_estimators=num_iterations)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TESTSIZE, random_state=RANDOMSTATE, shuffle=False)

        clf.fit(x_train, y_train)

        train_error, test_error = calculate_errors(x_train, x_test, y_train, y_test, clf)

        print(f"Number of iterations: {num_iterations}")
        print_error_data(train_error, test_error)

        train_errors.append(train_error)
        test_errors.append(test_error)

    x_data = iterations
    xlabel = "Number of iterations"
    title = "Training and Test Error vs. Number of iterations"
    plot_name = "AdaBoost-Training-Test-Error-vs-Number-Of-Iterations.png"
    plot(x_data, xlabel, title, train_errors, test_errors, plot_name, rstate)

def adt_plot_errors_vs_inner_max_depth(x, y, rstate):
    if rstate == RunState.PASS:
       return 

    train_errors = []
    test_errors = []

    depths = [i for i in range(1,11)] 
    for depth in depths:
        dt = tree.DecisionTreeClassifier(max_depth=depth)
        clf = AdaBoostClassifier(dt, random_state=RANDOMSTATE)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TESTSIZE, random_state=RANDOMSTATE, shuffle=False)

        clf.fit(x_train, y_train)

        train_error, test_error = calculate_errors(x_train, x_test, y_train, y_test, clf)

        print(f"Max Depth: {depth}")
        print_error_data(train_error, test_error)

        train_errors.append(train_error)
        test_errors.append(test_error)

    x_data = depths 
    xlabel = "Max Depth"
    title = "Training and Test Error vs. Inner Max Depth"
    plot_name = "AdaBoost-Training-Test-Error-vs-Inner-Max-Depth.png"
    plot(x_data, xlabel, title, train_errors, test_errors, plot_name, rstate)


def plot(x_data, xlabel, title, train_errors, test_errors, plot_name, rstate):
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, train_errors, label="Training Error %", color='blue', marker='o')
    plt.plot(x_data, test_errors, label="Test Error %", color='red', marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Error %")
    plt.legend()
    plt.grid(True)
    if rstate == RunState.SAVE:
        plt.savefig(plot_name)
    else:
        plt.show()

def calculate_errors(x_train, x_test, y_train, y_test, clf):
    train_accuracy = clf.score(x_train, y_train)
    test_accuracy = clf.score(x_test, y_test)
    train_error = (1-train_accuracy) * 100
    test_error = (1-test_accuracy) * 100
    return (train_error, test_error)

def print_error_data(train_error, test_error):
    print(f"train error {train_error}")
    print(f"test error {test_error}")
    print()

if __name__ == "__main__":
    main()
