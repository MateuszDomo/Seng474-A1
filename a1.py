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

def decision_tree(x, y):
    # training/test error curves vs training set size
    dt_plot_errors_vs_training_set_size(x, y, RunState.PASS)
    # training/test error curves vs criterion
    dt_plot_errors_vs_criterion(x, y, RunState.PASS)
    # training/test error curves vs max depth
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

    xlabel = "Training Set Size %"
    title = "Training and Test Error vs. Training Set %"
    plot_name = "Decision-Tree-Training-Test-Error-vs-Training-Set-Size.png"
    plot(xlabel, title, train_errors, test_errors, plot_name, rstate)

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

    xlabel = "Criterion"
    title = "Training and Test Error vs. Criterion"
    plot_name = "Decision-Tree-Training-Test-Error-vs-Criterion.png" 
    plot(xlabel, title, train_errors, test_errors, plot_name, rstate)

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

    xlabel = "Depth"
    title = "Training and Test Error vs. Max Depth"
    plot_name = "Decision-Tree-Training-Test-Error-vs-Max-Depth.png" 
    plot(xlabel, title, train_errors, test_errors, plot_name, rstate)

def calc_min_error_with_prune(x_train, y_train, x_test, y_test, **classifier_params):
    clf = tree.DecisionTreeClassifier(random_state=RANDOMSTATE)
    path = clf.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas = path.ccp_alphas

    best_train_alpha = None
    best_test_alpha = None
    min_train_error = 0
    min_test_error = 0
    for alpha in ccp_alphas:
        pruned_clf = tree.DecisionTreeClassifier(random_state=RANDOMSTATE, ccp_alpha=alpha, **classifier_params)
        pruned_clf.fit(x_train, y_train)
        train_error, test_error = calculate_errors(x_train, x_test, y_train, y_test, clf)
        
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
    # training/test error curves vs training set size
    rf_plot_errors_vs_training_set_size(x, y, RunState.PASS)
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

    xlabel = "Training Set Size %"
    title = "Training and Test Error vs. Training Set %"
    plot_name = "Random-Forest-Training-Test-Error-vs-Training-Set-Size.png"
    plot(xlabel, title, train_errors, test_errors, plot_name, rstate)


def adaboost_decision_tree(x, y):
    # training/test error curves vs training set size
    adt_plot_errors_vs_training_set_size(x, y, RunState.PASS)
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

        print(f"Training Set Size: {training_set_percent*10}")
        print_error_data(train_error, test_error)

        train_errors.append(train_error)
        test_errors.append(test_error)

    xlabel = "Training Set Size %"
    title = "Training and Test Error vs. Training Set %"
    plot_name = "AdaBoost-Training-Test-Error-vs-Training-Set-Size.png"
    plot(xlabel, title, train_errors, test_errors, plot_name, rstate)


def plot(xlabel, title, train_errors, test_errors, plot_name, rstate):
    plt.figure(figsize=(10, 6))
    plt.plot([i*10 for i in range(1, 9 + 1)], train_errors, label="Training Error %", color='blue', marker='o')
    plt.plot([i*10 for i in range(1, 9 + 1)], test_errors, label="Test Error %", color='red', marker='o')
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

def something():
    ## This is some dummy data just so you have a complete working example
    X = [[0, 0], [1, 1], [0, 0], [1, 1],[0, 0], [1, 1], [0, 0], [1, 1]]
    Y = [0, 1, 1, 0, 0, 1, 1, 0]
    M = 10 # number of trees in random forest
    rf = RandomForestClassifier(n_estimators = M, random_state = 0)
    rf = rf.fit(X, Y)
    n_samples = len(X)
    n_samples_bootstrap = n_samples



    ## THE ACTUAL STARTER CODE YOU SHOULD GRAB BEGINS BELOW

    ## Assumptions
    #    - n_samples is the number of examples
    #    - n_samples_bootstrap is the number of samples in each bootstrap sample
    #      (this should be equal to n_samples)
    #    - rf is a random forest, obtained via a call to
    #      RandomForestClassifier(...) in scikit-learn

    unsampled_indices_for_all_trees= []
    for estimator in rf.estimators_:
        random_instance = check_random_state(estimator.random_state)
        sample_indices = random_instance.randint(0, n_samples, n_samples_bootstrap)
        sample_counts = np.bincount(sample_indices, minlength = n_samples)
        unsampled_mask = sample_counts == 0
        indices_range = np.arange(n_samples)
        unsampled_indices = indices_range[unsampled_mask]
        unsampled_indices_for_all_trees += [unsampled_indices]

    ## Result:
    #    unsampled_indices_for_all_trees is a list with one element for each tree
    #    in the forest. In more detail, the j'th element is an array of the example
    #    indices that were \emph{not} used in the training of j'th tree in the
    #    forest. For examle, if the 1st tree in the forest was trained on a
    #    bootstrap sample that was missing only the first and seventh training
    #    examples (corresponding to indices 0 and 6), and if the last tree in the
    #    forest was trained on a boostrap sample that was missing the second,
    #    third, and sixth training examples (indices 1, 2, and 5), then
    #    unsampled_indices_for_all_trees would begin like:  
    #        [array([0, 6]),
    #         ...
    #         array([1, 2, 5])]

    print(unsampled_indices_for_all_trees)
