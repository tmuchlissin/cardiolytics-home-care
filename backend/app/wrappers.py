# import torch
# import numpy as np

# from torch.utils.data import Dataset, DataLoader
# from sklearn.base import BaseEstimator, ClassifierMixin

# class TabularDataset(Dataset):
#     def __init__(self, X, cat_idx, cont_idx, targets):
#         self.X = X
#         self.cat_idx = cat_idx
#         self.cont_idx = cont_idx
#         self.targets = targets
        
#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         row = self.X[idx]
#         x_cat = row[self.cat_idx].astype(np.int64)
#         x_cont = row[self.cont_idx].astype(np.float32)
#         target = np.array(self.targets[idx]).astype(np.float32)
#         return x_cat, x_cont, target

# class PyTorchClassifier(BaseEstimator, ClassifierMixin):
#     def __init__(self, model, epochs=10, lr=1e-3, batch_size=32,
#                  cat_idx=None, cont_idx=None, device=None):
#         self.model = model
#         self.epochs = epochs
#         self.lr = lr
#         self.batch_size = batch_size
#         self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # Set default indices if not provided
#         self.cat_idx = cat_idx if cat_idx is not None else [8, 9, 10, 11, 12, 13]
#         self.cont_idx = cont_idx if cont_idx is not None else [0, 1, 2, 3, 4, 5, 6, 7]

#     def fit(self, X, y):
#         dataset = TabularDataset(X, self.cat_idx, self.cont_idx, y)
#         loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
#         self.model.to(self.device)
#         self.model.train()
#         loss_fn = torch.nn.BCEWithLogitsLoss()
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

#         for epoch in range(self.epochs):
#             for x_cat, x_cont, targets in loader:
#                 x_cat, x_cont, targets = x_cat.to(self.device), x_cont.to(self.device), targets.to(self.device)
#                 optimizer.zero_grad()
#                 preds = self.model(x_cat, x_cont)
#                 loss = loss_fn(preds, targets.unsqueeze(1))
#                 loss.backward()
#                 optimizer.step()
#         return self

#     def predict(self, X):
#         self.model.eval()
#         # Ensure X is a writable NumPy array
#         X = np.copy(X)
#         # Ensure cat_idx and cont_idx are set
#         if not hasattr(self, 'cat_idx') or self.cat_idx is None:
#             self.cat_idx = [8, 9, 10, 11, 12, 13]
#         if not hasattr(self, 'cont_idx') or self.cont_idx is None:
#             self.cont_idx = [0, 1, 2, 3, 4, 5, 6, 7]
#         dummy_targets = np.zeros(len(X))
#         dataset = TabularDataset(X, self.cat_idx, self.cont_idx, dummy_targets)
#         loader = DataLoader(dataset, batch_size=self.batch_size)
#         all_preds = []
#         with torch.no_grad():
#             for x_cat, x_cont, _ in loader:
#                 x_cat, x_cont = x_cat.to(self.device), x_cont.to(self.device)
#                 preds = torch.sigmoid(self.model(x_cat, x_cont)).cpu().numpy()
#                 all_preds.append((preds >= 0.5).astype(int))
#         return np.concatenate(all_preds).squeeze()

#     def predict_proba(self, X):
#         self.model.eval()
#         # Ensure X is a writable NumPy array
#         X = np.copy(X)
#         # Ensure cat_idx and cont_idx are set
#         if not hasattr(self, 'cat_idx') or self.cat_idx is None:
#             self.cat_idx = [8, 9, 10, 11, 12, 13]
#         if not hasattr(self, 'cont_idx') or self.cont_idx is None:
#             self.cont_idx = [0, 1, 2, 3, 4, 5, 6, 7]
#         dummy_targets = np.zeros(len(X))
#         dataset = TabularDataset(X, self.cat_idx, self.cont_idx, dummy_targets)
#         loader = DataLoader(dataset, batch_size=self.batch_size)
#         all_probs = []
#         with torch.no_grad():
#             for x_cat, x_cont, _ in loader:
#                 x_cat, x_cont = x_cat.to(self.device), x_cont.to(self.device)
#                 probs = torch.sigmoid(self.model(x_cat, x_cont)).cpu().numpy()
#                 all_probs.append(probs)
#         probs = np.concatenate(all_probs).squeeze()
#         return np.vstack([1 - probs, probs]).T