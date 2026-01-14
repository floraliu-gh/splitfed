import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 導入你的模型定義與類別
from client_model import ClientModel
from server_model import ServerModel
from client1 import Client     
from server1 import MainServer   
from fed_server import fedserver

# --- 1. 設定 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

K = 5           # Client 數量
rounds = 20      # 總通訊輪數 (Aggregation 次數)
local_epochs = 1 # 每一輪，Client 在本地要跑幾個 Epoch
batch_size = 128
lr = 0.001

# --- 2. 資料準備 ---
print("\n準備 EuroSAT 資料集...")
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = datasets.EuroSAT('./data', download=True, transform=transform)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset_full, test_dataset = random_split(full_dataset, [train_size, test_size])

# 切分 Clients 資料
nk_list = [len(train_dataset_full) // K] * K
nk_list[-1] += len(train_dataset_full) - sum(nk_list)
client_subsets = random_split(train_dataset_full, nk_list)

# 建立 Loaders
dataloaders = [DataLoader(sub, batch_size=batch_size, shuffle=True) for sub in client_subsets]
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- 3. 初始化物件 (真正呼叫類別) ---

# 初始化 Server 物件 (注意：Server 只需一個模型)
global_server_model = ServerModel().to(device)
main_server = MainServer(global_server_model, device, lr=lr)

# 初始化 K 個 Client 物件
clients = []
for i in range(K):
    # 每個 Client 起始模型都一樣
    c_model = ClientModel().to(device)
    # 建立 Client 實例
    c_instance = Client(model=c_model, data_loader=dataloaders[i], device=device, lr=lr)
    clients.append(c_instance)

# --- 4. 輔助: 評估函數 ---
def evaluate(c_model, s_model, loader):
    c_model.eval()
    s_model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # Split Forward
            activations = c_model(x)
            outputs = s_model(activations)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return correct / total, all_labels, all_preds

# --- 5. 訓練迴圈 (SplitFed 核心邏輯) ---
print(f"\n開始訓練: Rounds={rounds}, Local Epochs={local_epochs}")
train_losses, test_accuracies = [], []

for r in range(rounds):
    print(f'--- Round {r+1}/{rounds} ---')
    round_loss = 0.0
    total_steps = 0
    
    # 在每一輪中，每個 Client 進行本地訓練
    for i in range(K):
        # 根據 local_epochs 決定每個 client 跑幾次
        # 因為你的 Client 類別內建了 data_iter，每次呼叫 ClientUpdate 就是抓一個 batch
        steps_per_client = len(dataloaders[i]) * local_epochs
        
        for _ in range(steps_per_client):
            # [步驟 1] Client 前向傳播 (取得 smashed data A_k 和 label y)
            A_k_detach, y = clients[i].ClientUpdate()
            
            # [步驟 2] Server 運算 (Forward + Backward 並取得 A_k 的梯度)
            dA_k, loss_val = main_server.ServerUpdate(A_k_detach, y)
            
            # [步驟 3] Server 更新權重
            main_server.step()
            
            # [步驟 4] Client 反向傳播 (傳入梯度 dA_k)
            clients[i].ClientBackprop(dA_k)
            
            round_loss += loss_val
            total_steps += 1
            
    avg_loss = round_loss / total_steps
    train_losses.append(avg_loss)
    
    # --- 6. Federated Aggregation Phase (FL 聚合) ---
    print("Aggregating Client Models...")
    # 取得所有 Client 的模型
    client_models_list = [c.model for c in clients]
    # 聚合 (假設 fedserver 傳回一個聚合後的 model)
    aggregated_model = fedserver(client_models_list, nk_list, sum(nk_list))
    
    # 將聚合後的權重載入回每一個 Client 物件的模型中
    global_weights = aggregated_model.state_dict()
    for c in clients:
        c.model.load_state_dict(global_weights)
        
    # --- 7. 評估本輪結果 ---
    # 使用第一個 Client 的模型(已聚合)搭配目前的 Server 模型
    test_acc, y_true, y_pred = evaluate(clients[0].model, main_server.model, test_loader)
    test_accuracies.append(test_acc)
    print(f"Loss: {avg_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")

# --- 8. 結果視覺化 ---
print(f'\nFinal Test Accuracy: {test_accuracies[-1] * 100:.2f}%')

# 混淆矩陣
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Final Confusion Matrix')
plt.show()

# 曲線圖
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Train Loss')
plt.subplot(1, 2, 2)
plt.plot(test_accuracies)
plt.title('Test Accuracy')
plt.show()