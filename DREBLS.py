import numpy as np
from construct_L import construct_L
import matplotlib.pyplot as plt
from scipy.linalg import solve_sylvester


def DREBLS(A, Y, A_test=None, Y_test=None, lambda_para1=1e-3, lambda_para2=1e-3, epochs=500, epsilon=1e-1, t=1e-5):
    """
    Discriminative relaxed-graph embedding broad learning system (DREBLS)
    """
    n, d = A.shape
    _, c = Y.shape
    B = (Y - 1) * 2 + 1  # Transform labels to -1/+1
    gnd = np.argmax(Y, axis=1)


    # Construct Laplacian matrices (D and V)
    D, V = construct_L(A, gnd)

    # Precompute terms
    AA = np.dot(A.T, A)
    ADA = np.dot(A.T, np.dot(D, A))
    AVA = np.dot(A.T, np.dot(V, A))
    AVA2 = np.dot(A.T, np.dot(V.T, A))

    # Initialize W0 and P
    W0 = np.linalg.solve(AA + 0.01 * np.eye(d), np.dot(A.T, Y))
    P = W0.copy()

    losses = []
    train_accuracies = []
    test_accuracies = []

    for i in range(epochs):
        # Optimize R
        E = np.dot(A, W0) - Y
        M = np.maximum(B * E, np.zeros((n, c)))
        R = Y + (B * M)

        # Optimize T
        T = np.linalg.solve(np.dot(P.T, P) + t * np.eye(c), np.dot(P.T, W0))

        # Optimize P via Sylvester equation
        item1 = lambda_para1 * ADA
        item2 = lambda_para2 * np.dot(T, T.T)
        item3 = lambda_para1 * np.dot(AVA2, W0) + lambda_para2 * np.dot(W0, T.T)
        P = solve_sylvester(item1, item2, item3)

        # Optimize W
        W = np.linalg.solve(
            AA + lambda_para1 * ADA + lambda_para2 * np.eye(d),
            np.dot(A.T, R) + lambda_para1 * np.dot(AVA, P) + lambda_para2 * np.dot(P, T)
        )

        # Calculate loss
        t1 = np.dot(A, W) - R
        t2 = np.trace(np.dot(t1.T, t1))
        t3 = np.trace(np.dot(W.T, np.dot(ADA, W))) + \
             np.trace(np.dot(P.T, np.dot(ADA, P))) - \
             2 * np.trace(np.dot(W.T, np.dot(AVA, P)))
        t4 = W - np.dot(P, T)
        t5 = np.trace(np.dot(t4.T, t4))
        current_loss = t2 + lambda_para1 * t3 + lambda_para2 * t5
        losses.append(current_loss)

        # Calculate training accuracy
        train_accuracy = np.mean(result(np.dot(A, W)) == np.argmax(Y, axis=1))
        train_accuracies.append(train_accuracy)

        # Calculate test accuracy (if test set is provided)
        if A_test is not None and Y_test is not None:
            test_accuracy = np.mean(result(np.dot(A_test, W)) == np.argmax(Y_test, axis=1))
            test_accuracies.append(test_accuracy)
        else:
            test_accuracy = None

        # Print loss and accuracies (every 10 epochs)
        if i % 10 == 0:
            print(f"Epoch {i}, Loss: {current_loss}, Training Accuracy: {train_accuracy * 100:.2f}%",
                  f"Test Accuracy: {test_accuracy * 100:.2f}%" if test_accuracy is not None else "")

        # Check convergence
        if np.trace(np.dot((W - W0).T, (W - W0))) < epsilon:
            print(f"Converged at epoch {i}, Loss: {current_loss}")
            break

        # Check if loss increased (optional)
        if i > 0 and current_loss > losses[-2]:  # Compare with previous loss
            W = W0  # Revert to previous W if loss increased
            print(f"Loss increased at epoch {i}, reverting to previous W")
            break

        W0 = W

    # Plot loss curve
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('DREBLS Training Loss')
    plt.show()

    # Plot training accuracy curve
    plt.plot(train_accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.title('DREBLS Training Accuracy')
    plt.show()

    # Plot test accuracy curve (if test set is provided)
    if A_test is not None and Y_test is not None:
        plt.plot(test_accuracies)
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.title('DREBLS Test Accuracy')
        plt.show()

    return W

def result(x):
    """Helper function to get predicted class labels"""
    return np.argmax(x, axis=1)