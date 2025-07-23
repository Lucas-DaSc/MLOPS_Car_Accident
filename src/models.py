from sklearn.linear_model import SGDClassifier

#SGDClassifier
def get_model_sgdc():
    return SGDClassifier(loss='log_loss',random_state=42)
