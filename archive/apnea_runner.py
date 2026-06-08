import torch, apnea_trainer
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = apnea_trainer.ParallelCNNTransformer().to(DEVICE)
model.load_state_dict(torch.load("apnea_parallel_cnn_transformer.pth"))
model.eval()

apnea_trainer.tqdm = lambda x, **kwargs: x
X, y = apnea_trainer.load_data()
_, X_test, _, y_test = apnea_trainer.train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

test_loader = torch.utils.data.DataLoader(
    apnea_trainer.ApneaECGDataset(X_test, y_test), batch_size=64, shuffle=False
)

test_preds, test_trues = [], []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X.to(DEVICE))
        preds = outputs.argmax(dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_trues.extend(batch_y.numpy())

acc, prec, rec, spec, f1 = apnea_trainer.evaluate_metrics(
    np.array(test_trues), np.array(test_preds)
)
print(f"Paper Reproduction Metrics:")
print(f"  Accuracy:    {acc:.4f}")
print(f"  Precision:   {prec:.4f}")
print(f"  Recall:      {rec:.4f}")
print(f"  Specificity: {spec:.4f}")
print(f"  F1-Score:    {f1:.4f}")
