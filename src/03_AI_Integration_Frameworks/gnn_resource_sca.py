import sys
import os
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# Thiết lập đường dẫn import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.plot_utils import save_plot_as_pdf

# --- Thiết lập Hệ thống & Tham số ---
K = 5               # Số lượng nodes/users
NOISE_POWER = 1e-9
P_MAX = 0.5; P_TOTAL = 1.5; MIN_POWER = 1e-6
LOG_2 = np.log(2)
SCA_MAX_ITERS = 50
TRUST_REGION_SQ = 0.01

# Tham số GNN
N_SAMPLES = 1000     # Số lượng mẫu dữ liệu để huấn luyện
GNN_EPOCHS = 20      # Số lượng epochs huấn luyện
LR = 0.01

# --- HÀM TÍNH TOÁN CORE (Tái sử dụng) ---

# (Tái sử dụng các hàm calculate_rate, calculate_h_k_gradient và sca_optimize_power từ Task 7)
# Để giữ code ngắn gọn, chúng ta sẽ định nghĩa lại hàm SCA chính trong file này.

# Hàm Hỗ trợ: Tính Rate và Gradient (giữ nguyên logic Task 1)
# ... [define calculate_rate, calculate_h_k_gradient] ...
def calculate_rate(p, H, sigma2):
    rates = np.zeros(K)
    epsilon = 1e-12
    for k in range(K):
        signal = H[k, k] * p[k]
        interference = np.sum(H[k, np.arange(K) != k] * p[np.arange(K) != k])
        sinr = signal / (interference + sigma2 + epsilon)
        rates[k] = np.log(1 + sinr) / LOG_2
    return np.sum(rates)

def calculate_h_k_gradient(p_t, H, sigma2, k):
    I_k_t = np.sum(H[k, np.arange(K) != k] * p_t[np.arange(K) != k]) + sigma2 + 1e-12
    grad = np.zeros(K)
    for i in range(K):
        if i != k:
            grad[i] = H[k, i] / I_k_t
    return grad

def sca_optimize_power(H, sigma2, K, P_MAX, P_TOTAL, p_init):
    # Đây là logic SCA từ Task 7, giải bài toán Sum-Rate Max
    p_t = p_init.copy()

    # ... (Logic SCA loop, sử dụng cp.SCS) ...
    for t in range(SCA_MAX_ITERS):
        p = cp.Variable(K)
        objective_terms = []

        for k in range(K):
            A_k_term = cp.sum(H[k, :] @ p) + sigma2
            g_k = cp.log(A_k_term)

            grad_h_k = calculate_h_k_gradient(p_t, H, sigma2, k)
            I_k_t = np.sum(H[k, np.arange(K) != k] * p_t[np.arange(K) != k]) + sigma2 + 1e-12
            h_k_t = np.log(I_k_t)
            h_k_approx = h_k_t + cp.sum(grad_h_k @ (p - p_t))

            objective_terms.append(g_k - h_k_approx)

        objective = cp.Maximize(cp.sum(objective_terms) / LOG_2)
        constraints = [p >= MIN_POWER, p <= P_MAX, cp.sum(p) <= P_TOTAL, cp.sum_squares(p - p_t) <= TRUST_REGION_SQ]

        try:
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.SCS, verbose=False)

            if problem.status not in ["optimal", "optimal_inaccurate"]: break

            p_new = p.value
            if p_new is None or np.any(np.isnan(p_new)): break

            if np.linalg.norm(p_new - p_t) < 1e-4 and t > 5: break

            p_t = p_new
        except Exception:
            break

    return p_t

# --- 3. GNN MODEL (Graph Convolutional Network) ---

class GCNPredictor(torch.nn.Module):
    """
    GNN đơn giản để học mối quan hệ kênh và dự đoán ma trận kênh.
    Nodes: K users. Features: K (CSI từ tất cả các nodes khác).
    Output: K*K (Flattened H matrix).
    """
    def __init__(self, num_features, num_nodes):
        super(GCNPredictor, self).__init__()
        # 3 lớp GCN
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 1) # Output 1 feature/node
        self.linear = torch.nn.Linear(num_nodes, num_nodes * num_nodes) # Map node features to H matrix

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Lớp 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # Lớp 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Lớp 3 (Output features for each node)
        x = self.conv3(x, edge_index)

        # Flatten and reshape to K*K matrix
        x = x.view(-1) # Flatten vector
        return self.linear(x).view(K, K) # Reshape to KxK

# --- 4. DATA GENERATION & TRAINING LOOP ---

def create_graph_data(H_matrix):
    """Tạo đối tượng PyG Data từ H matrix."""
    # Node Features (X): Mỗi user/node có K features (CSI của nó đến mọi người dùng khác)
    x = torch.tensor(H_matrix.T, dtype=torch.float).view(K, K)

    # Edge Index: Cấu trúc mạng (Graph đầy đủ)
    edges = []
    for i in range(K):
        for j in range(K):
            if i != j:
                edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Target (Y): H matrix
    y = torch.tensor(H_matrix, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)

def generate_channel(K):
    """Tạo ma trận độ lợi kênh."""
    H = np.random.exponential(scale=1.0, size=(K, K))
    return H


def train_gnn_and_optimize_sca(n_samples=N_SAMPLES, epochs=GNN_EPOCHS):

    # Tạo tập dữ liệu (Mô phỏng H thay đổi theo thời gian/vị trí)
    data_list = []
    for _ in range(n_samples):
        H_sample = generate_channel(K)
        data_list.append(create_graph_data(H_sample))

    # --- Huấn luyện GNN ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNPredictor(num_features=K, num_nodes=K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss() # Dùng MSE để dự đoán H matrix

    print(f"Bắt đầu huấn luyện GNN trên {device}...")

    # Huấn luyện
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_total = 0
        for data in data_list:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            loss_total += loss

        loss_total.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss_total.item() / n_samples:.6f}')

    # --- Tích hợp GNN và SCA (Test Phase) ---
    model.eval()

    # Tạo một ma trận kênh mới để dự đoán
    H_test_true = generate_channel(K)
    data_test = create_graph_data(H_test_true).to(device)

    with torch.no_grad():
        H_predicted_tensor = model(data_test)

    H_predicted = H_predicted_tensor.cpu().numpy()

    # 5. SCA Sử dụng Output của GNN

    p_init = np.ones(K) * P_TOTAL / K

    # 5a. Chạy SCA trên Kênh TRUE (Baseline)
    p_true = sca_optimize_power(H_test_true, NOISE_POWER, K, P_MAX, P_TOTAL, p_init)
    rate_true = calculate_rate(p_true, H_test_true, NOISE_POWER)

    # 5b. Chạy SCA trên Kênh PREDICTED (Hybrid Model)
    p_predicted = sca_optimize_power(H_predicted, NOISE_POWER, K, P_MAX, P_TOTAL, p_init)

    # Tính Rate thực tế của nghiệm Predicted trên kênh TRUE
    rate_predicted_on_true_H = calculate_rate(p_predicted, H_test_true, NOISE_POWER)

    # Đánh giá độ chính xác của GNN và so sánh với Baseline SCA

    return rate_true, rate_predicted_on_true_H, np.sum(p_true), np.sum(p_predicted)

# --- CHẠY CHÍNH ---
if __name__ == "__main__":

    if not os.path.exists('output'):
        os.makedirs('output')

    np.random.seed(42)
    torch.manual_seed(42)

    rate_true, rate_hybrid, power_true, power_hybrid = train_gnn_and_optimize_sca()

    print("\n" + "="*40)
    print("Kết quả cuối cùng Task 8 (GNN-SCA Hybrid)")
    print("="*40)
    print(f"Baseline SCA Rate (Perfect CSI): {rate_true:.4f} bits/s/Hz")
    print(f"Hybrid GNN-SCA Rate (Predicted CSI): {rate_hybrid:.4f} bits/s/Hz")
    print(f"Độ mất mát hiệu suất (Rate Loss): {(rate_true - rate_hybrid) / rate_true * 100:.2f} %")
    print(f"Công suất sử dụng (Hybrid): {power_hybrid:.4f} W")

    # Vì đây là một lần chạy đơn, chúng ta sẽ không vẽ đồ thị hội tụ, mà chỉ kết quả so sánh.
    print("Đồ thị so sánh không được tạo ra cho Task 8 (Single Run Analysis).")
