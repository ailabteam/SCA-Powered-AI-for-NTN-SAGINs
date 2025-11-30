import sys
import os
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# --- Sửa lỗi Import Path ---
# Thêm thư mục gốc của project vào Python Path để import utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.plot_utils import save_plot_as_pdf

# --- Thiết lập Hệ thống & Tham số ---
K = 5               # Số lượng người dùng (User)
P_MAX = 0.5         # Công suất tối đa cho mỗi người dùng (W)
P_TOTAL = 1.5       # Công suất tổng tối đa của hệ thống (W)
NOISE_POWER = 1e-9  # Công suất nhiễu (sigma^2)
DELTA_SQUARED = 0.1 # Bán kính vùng tin cậy (Trust Region) ban đầu (delta^2)
MIN_POWER = 1e-6    # Công suất tối thiểu để tránh log(0)
LOG_2 = np.log(2)   # Hằng số chuyển đổi log tự nhiên sang log2

# --- 1. HÀM MÔ PHỎNG KÊNH ---
def generate_channel(K):
    """Tạo ma trận độ lợi kênh (h_kj)."""
    # Tạo độ lợi kênh ngẫu nhiên
    H = np.random.exponential(scale=1.0, size=(K, K))
    return H

# --- 2. HÀM TÍNH TOÁN CORE ---

def calculate_rate(p, H, sigma2):
    """Tính toán Sum-Rate phi lồi thực tế (để kiểm tra)."""
    rates = np.zeros(K)
    epsilon = 1e-12 # Epsilon cho ổn định số học
    for k in range(K):
        signal = H[k, k] * p[k]
        
        # Nhiễu + Nhiễu (Ngoại trừ người dùng k)
        interference = np.sum(H[k, np.arange(K) != k] * p[np.arange(K) != k])
        
        sinr = signal / (interference + sigma2 + epsilon)
        # Sử dụng công thức chuyển đổi log tự nhiên sang log2
        rates[k] = np.log(1 + sinr) / LOG_2 
    return np.sum(rates)

def calculate_h_k_gradient(p_t, H, sigma2, k):
    """Tính Gradient của thành phần phi lồi h_k = log(I_k + sigma^2) tại p_t."""
    # Mẫu số I_k_t: Tổng nhiễu + nhiễu tại điểm p_t
    I_k_t = np.sum(H[k, np.arange(K) != k] * p_t[np.arange(K) != k]) + sigma2 + 1e-12
    
    grad = np.zeros(K)
    
    # Chỉ p_i (i != k) ảnh hưởng đến I_k. Đạo hàm d/dp_i [log(I_k)] = h_{ki} / I_k_t
    for i in range(K):
        if i != k:
            grad[i] = H[k, i] / I_k_t
            
    return grad

# --- 3. THUẬT TOÁN SCA CHÍNH ---

def sca_sum_rate_maximization(H, sigma2, K, P_MAX, P_TOTAL, 
                              max_iters=100, tolerance=1e-4, delta_sq=DELTA_SQUARED):
    
    # Khởi tạo công suất ban đầu (chia đều)
    p_t = np.ones(K) * P_TOTAL / K
    
    sum_rates = [calculate_rate(p_t, H, sigma2)]
    
    print(f"Bước lặp 0: Rate = {sum_rates[0]:.4f}")

    current_delta_sq = delta_sq # Biến vùng tin cậy có thể thay đổi

    for t in range(max_iters):
        
        # Bước 1: Khai báo biến tối ưu (p)
        p = cp.Variable(K)
        
        # Bước 2: Xây dựng hàm mục tiêu lồi
        objective_terms = []
        
        for k in range(K):
            # 2a. Thành phần lõm g_k(p) = log(I_k + h_kk p_k + sigma^2)
            A_k_term = cp.sum(H[k, :] @ p) + sigma2
            g_k = cp.log(A_k_term) 
            
            # 2b. Tính xấp xỉ tuyến tính cho h_k(p) tại p_t
            grad_h_k = calculate_h_k_gradient(p_t, H, sigma2, k)
            
            # Lấy giá trị tại p_t (đảm bảo tính hợp lệ)
            I_k_t = np.sum(H[k, np.arange(K) != k] * p_t[np.arange(K) != k]) + sigma2 + 1e-12
            h_k_t = np.log(I_k_t)
            
            # Xấp xỉ tuyến tính: h_k_t + grad_h_k^T * (p - p_t)
            h_k_approx = h_k_t + cp.sum(grad_h_k @ (p - p_t))
            
            objective_terms.append(g_k - h_k_approx)

        # Bài toán con lồi (Maximize Concave)
        objective = cp.Maximize(cp.sum(objective_terms) / LOG_2) 
        
        # Bước 3: Xây dựng ràng buộc
        constraints = [
            p >= MIN_POWER,           # Ràng buộc công suất min
            p <= P_MAX,
            cp.sum(p) <= P_TOTAL,
            # Ràng buộc Vùng Tin Cậy
            cp.sum_squares(p - p_t) <= current_delta_sq 
        ]
        
        # Bước 4: Giải bài toán con lồi
        try:
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                # Thử solver khác
                problem.solve(solver=cp.CVXOPT, verbose=False)
                if problem.status not in ["optimal", "optimal_inaccurate"]:
                    print(f"Lỗi: Bài toán con không thể giải ở bước {t+1}.")
                    break
                    
            p_new = p.value
            
            # Đảm bảo p_new là hợp lệ và không null
            if p_new is None or np.any(np.isnan(p_new)):
                print(f"Lỗi: Công suất mới không hợp lệ (None/NaN) tại bước {t+1}.")
                break
            
            rate_new = calculate_rate(p_new, H, sigma2)
            sum_rates.append(rate_new)
            
            print(f"Bước lặp {t+1}: Rate = {rate_new:.4f}")
            
            # Kiểm tra điều kiện dừng
            if t > 0 and (sum_rates[-1] - sum_rates[-2]) / sum_rates[-2] < tolerance and rate_new > sum_rates[-2]:
                print(f"Hội tụ thành công tại bước {t+1}.")
                break
                
            # CẬP NHẬT p_t cho bước lặp tiếp theo
            p_t = p_new
            
            # Giảm dần vùng tin cậy để hội tụ chính xác hơn
            # current_delta_sq = max(current_delta_sq * 0.95, 1e-6)

        except Exception as e:
            print(f"Solver Exception tại bước {t+1}: {e}")
            break

    return p_t, sum_rates

# --- CHẠY VÀ VẼ ĐỒ THỊ ---
if __name__ == "__main__":
    
    # Tạo thư mục output nếu chưa có
    if not os.path.exists('output'):
        os.makedirs('output')

    # Mô phỏng kênh truyền
    H_matrix = generate_channel(K)
    
    # Chạy thuật toán SCA
    optimal_power, convergence_rates = sca_sum_rate_maximization(
        H_matrix, NOISE_POWER, K, P_MAX, P_TOTAL
    )
    
    print("\nKết quả cuối cùng:")
    print(f"Công suất tối ưu (W): {optimal_power}")
    print(f"Tổng công suất sử dụng: {np.sum(optimal_power):.4f} W")
    print(f"Tốc độ dữ liệu tối đa: {convergence_rates[-1]:.4f} bits/s/Hz")
    
    # Vẽ đồ thị hội tụ
    plt.figure()
    plt.plot(convergence_rates, marker='o', linestyle='-', color='b')
    plt.title('SCA Convergence for Sum-Rate Maximization (Task 1)')
    plt.xlabel('Iteration')
    plt.ylabel('Sum-Rate (bits/s/Hz)')
    plt.grid(True)
    
    # Lưu plot
    save_plot_as_pdf(plt, 'output/T1_SCA_Convergence.pdf')
    
    print("Đồ thị hội tụ đã được lưu tại output/T1_SCA_Convergence.pdf")
