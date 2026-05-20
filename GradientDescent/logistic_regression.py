import torch

class LogisticRegression:
    
    def __init__(self, dimension):
        self.W = torch.rand(dimension, device="cuda", dtype=torch.float32, requires_grad=True)
        self.b = torch.rand(1, device="cuda", dtype=torch.float32, requires_grad=True)
    
    def logistic_regression(self, X):
        y_pred = torch.matmul(X, self.W) + self.b # wx+b
        y_pred = torch.sigmoid(y_pred) # 1/(1+e**(-y_pred))
        return y_pred
    
    def binary_cross_entropy(self, y, y_pred):
            loss = (y * torch.log_(y_pred)) + ((1-y) * log(1-y_pred))
            return torch.mean(-1 * loss)
        
    def fit(self, x, y, batch_size=12, lr=0.01):
        batch_count = (x.shape[0]+ batch_size-1)//batch_size
        
        loss = 0
        for i in range(batch_count):
            start = i*batch_size
            end = start + batch_size
            #convert to tensor
            X = torch.tensor(x[start:end], dtype=torch.float32, device="cuda")
            Y = torch.tensor(y[start:end], dtype=torch.float32, device="cuda")
            # main training process
            y_pred = self.logistic_regression(X)
            
            curr_loss = self.binary_cross_entropy(Y, y_pred)
            curr_loss.backward()
            with torch.no_grad():
                self.W -= lr * self.W.grad
                self.b -= lr * self.b.grad
            
            self.W.grad.zero_()
            self.b.grad.zero_()
            
            loss += curr_loss.item()
        
        return loss/batch_count
    
    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            y = self.logistic_regression(x)
            return y