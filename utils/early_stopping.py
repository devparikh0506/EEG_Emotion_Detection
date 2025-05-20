# reference : https://stackoverflow.com/a/71999355
class EarlyStopping:
    def __init__(self, patience=5, mode='min', delta=0.0):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.mode = mode
        self.delta = delta
        self.best_model_wts  = None
        self._best_score : int | None = None

        if mode == 'min':
            self.best_score = float('inf')
        elif mode == 'max':
            self.best_score = float('-inf')
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def __call__(self, current_score):
        if self.mode == 'min':
            improved = current_score < self.best_score - self.delta
        else:  # mode == 'max'
            improved = current_score > self.best_score + self.delta

        if improved:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
