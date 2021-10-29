##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: 
# Complete the following methods of Evaluator:
# - confusion_matrix()
# - accuracy()
# - precision()
# - recall()
# - f1_score()
##############################################################################

import numpy as np


class Evaluator(object):
    """ Class to perform evaluation
    """

    def confusion_matrix(self, prediction, annotation, class_labels=None):
        """ Computes the confusion matrix.
        
        Parameters
        ----------
        prediction : np.array
            an N dimensional numpy array containing the predicted
            class labels
        annotation : np.array
            an N dimensional numpy array containing the ground truth
            class labels
        class_labels : np.array
            a C dimensional numpy array containing the ordered set of class
            labels. If not provided, defaults to all unique values in
            annotation.
        
        Returns
        -------
        np.array
            a C by C matrix, where C is the number of classes.
            Classes should be ordered by class_labels.
            Rows are ground truth per class, columns are predictions.
        """

        if not class_labels:
            class_labels = np.unique(annotation)


        #######################################################################
        #                 ** TASK 3.1: COMPLETE THIS METHOD **
        #######################################################################

        # if class labels given - confusion matrix should order output according to labels
        # if not given - generate sorted set of class labels from ground truth and use to order matrix
        # need to match up ground truths and predictions
        # define 2d array
        confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)
        # iter over predictions
        for label in list(class_labels):
            indices = [i for i, x in enumerate(list(annotation)) if x == label]
            for index in indices:
                p = prediction[index]
                row = [i for i, x in enumerate(list(class_labels)) if x == label][0]
                col = [i for i, x in enumerate(list(class_labels)) if x == p][0]
                confusion[row][col] = confusion[row][col] + 1

        # check what corresponding ground truth should be
        # update array accordingly with +1 for  whichever of TP,FP,...
        # return result

        return confusion

    def accuracy(self, confusion):
        """ Computes the accuracy given a confusion matrix.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions
        
        Returns
        -------
        float
            The accuracy (between 0.0 to 1.0 inclusive)
        """
        total = 0
        correct_classifications = 0
        for i in range(len(list(confusion))):
            for j in range(len(list(confusion[0]))):
                if i == j:
                    correct_classifications += confusion[i][j]
                total += confusion[i][j]

        accuracy = correct_classifications / total

        #######################################################################
        #                 ** TASK 3.2: COMPLETE THIS METHOD **
        #######################################################################

        return accuracy

    def precision(self, confusion):
        """ Computes the precision score per class given a confusion matrix.
        
        Also returns the macro-averaged precision across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the precision score for each
            class in the same order as given in the confusion matrix.
        float
            The macro-averaged precision score across C classes.   
        """

        # Initialise array to store precision for C classes
        p = np.zeros((len(confusion),))

        #######################################################################
        #                 ** TASK 3.3: COMPLETE THIS METHOD **
        #######################################################################

        # diagonal element divided by sum of columns
        for j in range(len(list(confusion[0]))):
            diagonal = confusion[j][j]
            total = 0
            for i in range(len(list(confusion))):
                total += confusion[i][j]
            precision = diagonal / total
            p[j] = precision

        macro_p = sum(list(p)) / len(list(p))

        return p, macro_p

    def recall(self, confusion):
        """ Computes the recall score per class given a confusion matrix.
        
        Also returns the macro-averaged recall across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the recall score for each
            class in the same order as given in the confusion matrix.
        
        float
            The macro-averaged recall score across C classes.   
        """

        # Initialise array to store recall for C classes
        r = np.zeros((len(confusion),))

        #######################################################################
        #                 ** TASK 3.4: COMPLETE THIS METHOD **
        #######################################################################

        # diagonal element divided by sum of rows
        for i in range(len(list(confusion))):
            diagonal = confusion[i][i]
            total = 0
            for j in range(len(list(confusion[0]))):
                total += confusion[i][j]
            recall = diagonal / total
            r[i] = recall

        macro_r = sum(list(r)) / len(list(r))

        return r, macro_r

    def f1_score(self, confusion):
        """ Computes the f1 score per class given a confusion matrix.
        
        Also returns the macro-averaged f1-score across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the f1 score for each
            class in the same order as given in the confusion matrix.
        
        float
            The macro-averaged f1 score across C classes.   
        """

        # Initialise array to store recall for C classes
        f = np.zeros((len(confusion),))

        #######################################################################
        #                 ** YOUR TASK: COMPLETE THIS METHOD **
        #######################################################################

        precisions = list(self.precision(confusion)[0])
        recalls = list(self.recall(confusion)[0])
        for i in range(len(precisions)):
            p = precisions[i]
            r = recalls[i]
            f_1 = 2 * ((p * r) / (p + r))
            f[i] = f_1

        macro_f = sum(list(f)) / len(list(f))

        return f, macro_f
