import sys
import os
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Thiết lập đường dẫn import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.plot_utils import save_plot_as_pdf

# --- Thiết lập Hệ thống & Tham số Cố định ---
P_MAX = 0.5         
P_TOTAL = 1.5       
NOISE_POWER = 1e-9  
MIN_POWER = 1e-6    
LOG_2 = np.log(2)
SCA_MAX_ITERS = 50
SCA_TOLERANCE = 1e-4
TRUST_REGION_SQ = 0.01

# --- HÀM TÍNH TOÁN CORE (Tái sử dụng) ---

def generate_channel(K, seed=None):
    if seed is not None:
        np.random.seed(seed)
    H = np.random.exponential(scale=1.0, size=(K, K))
    return H

def calculate_rate(p, H, sigma2, K):
    """Tính toán Sum-Rate phi lồi cho K users."""
    rates = np.zeros(K)
    epsilon = 1e-12 
    for k in range(K):
        signal = H[k, k] * p[k]
        
        # Cần đảm bảo index slicing đúng cho kích thước K hiện tại
        interference = np.sum(H[k, np.arange(K) != k] * p[np.arange(K) != k])
        
        sinr = signal / (interference + sigma2 + epsilon)
        rates[k] = np.log(1 + sinr) / LOG_2 
    return np.sum(rates)

def calculate_h_k_gradient(p_t, H, sigma2, K, k):
    """Tính Gradient (phi lồi) cho K users."""
    I_k_t = np.sum(H[k, np.arange(K) != k] * p_t[np.arange(K) != k]) + sigma2 + 1e-12
    grad = np.zeros(K)
    for i in range(K):
        if i != k:
            grad[i] = H[k, i] / I_k_t
    return grad

def sca_optimize_power(H, sigma2, K, p_init, max_iters=SCA_MAX_ITERS):
    
    p_t = p_init.copy()
    convergence_rates = []
    
    for t in range(max_iters):
        p = cp.Variable(K)
        objective_terms = []
        
        for k in range(K):
            # Xấp xỉ Rate phi lồi
            A_k_term = cp.sum(H[k, :] @ p) + sigma2
            g_k = cp.log(A_k_term)
            
            grad_h_k = calculate_h_k_gradient(p_t, H, sigma2, K, k)
            I_k_t = np.sum(H[k, np.arange(K) != k] * p_t[np.arange(K) != k]) + sigma2 + 1e-12
            h_k_t = np.log(I_k_t)
            h_k_approx = h_k_t + cp.sum(grad_h_k @ (p - p_t))
            
            objective_terms.append(g_k - h_k_approx)

        objective = cp.Maximize(cp.sum(objective_terms) / LOG_2)
        
        constraints = [
            p >= MIN_POWER,
            p <= P_MAX,
            cp.sum(p) <= P_TOTAL,
            cp.sum_squares(p - p_t) <= TRUST_REGION_SQ
        ]
        
        try:
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.SCS, verbose=False) # Ưu tiên SCS
            
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                problem.solve(solver=cp.ECOS, verbose=False)
                if problem.status not in ["optimal", "optimal_inaccurate"]: break
                    
            p_new = p.value
            if p_new is None or np.any(np.isnan(p_new)): break
            
            rate_new = calculate_rate(p_new, H, sigma2, K)
            convergence_rates.append(rate_new)
            
            # Kiểm tra hội tụ SCA nội bộ
            if len(convergence_rates) > 1 and np.abs(rate_new - convergence_rates[-2]) < SCA_TOLERANCE and t > 5: break
                
            p_t = p_new

        except Exception:
            break
            
    return p_t, convergence_rates


# --- MÔ PHỎNG TRANSFER LEARNING ---

def tl_sca_comparison():
    
    K_source = 3  # Mạng mật độ thấp
    K_target = 8  # Mạng mật độ cao
    
    # --- GIAI ĐOẠN 1: SOURCE DOMAIN (K=3) ---
    # Sử dụng kênh cố định (seed 42)
    H_source = generate_channel(K_source, seed=10)
    p_init_source = np.ones(K_source) * P_TOTAL / K_source
    
    p_star_source, _ = sca_optimize_power(H_source, NOISE_POWER, K_source, p_init_source, max_iters=20)
    
    # --- GIAI ĐOẠN 2: TARGET DOMAIN (K=8) ---
    H_target = generate_channel(K_target, seed=20) # Kênh mới
    
    # ----------------------------------------------------
    # SCENARIO A: BASELINE (Random Initialization / Equal Power)
    # ----------------------------------------------------
    p_init_baseline = np.ones(K_target) * P_TOTAL / K_target
    _, rates_baseline = sca_optimize_power(H_target, NOISE_POWER, K_target, p_init_baseline)
    
    # ----------------------------------------------------
    # SCENARIO B: TRANSFER LEARNING (Warm-Start)
    # ----------------------------------------------------
    
    # 1. Mapping: Chuyển p* từ K=3 sang K=8
    p_init_tl = np.zeros(K_target)
    
    # Giữ công suất cũ
    p_init_tl[:K_source] = p_star_source 
    
    # Công suất còn lại được chia đều cho các nodes mới
    power_remaining = P_TOTAL - np.sum(p_star_source)
    if power_remaining < 0:
        # Nếu công suất cũ đã vượt quá P_TOTAL (không xảy ra ở đây, nhưng là logic quan trọng)
        p_init_tl = p_init_tl / np.sum(p_init_tl) * P_TOTAL
        power_remaining = 0
        
    num_new_users = K_target - K_source
    if num_new_users > 0:
        p_init_tl[K_source:] = power_remaining / num_new_users
        
    # Đảm bảo p_init_tl hợp lệ
    p_init_tl = np.clip(p_init_tl, MIN_POWER, P_MAX)

    # 2. Chạy SCA với Warm-Start TL
    _, rates_tl = sca_optimize_power(H_target, NOISE_POWER, K_target, p_init_tl)

    return rates_baseline, rates_tl

# --- CHẠY VÀ VẼ ĐỒ THỊ ---
if __name__ == "__main__":
    
    if not os.path.exists('output'):
        os.makedirs('output')

    rates_baseline, rates_tl = tl_sca_comparison()
    
    # Vẽ đồ thị so sánh hội tụ
    plt.figure()
    plt.plot(rates_baseline, marker='.', linestyle='--', label='Baseline (Equal Init)')
    plt.plot(rates_tl, marker='o', linestyle='-', label='TL Warm-Start')
    
    plt.title('Transfer Learning for SCA Warm-Start (Task 9)')
    plt.xlabel('SCA Iteration')
    plt.ylabel('Sum-Rate (bits/s/Hz)')
    plt.legend()
    plt.grid(True)
    
    save_plot_as_pdf(plt, 'output/T9_TL_SCA_Convergence.pdf')
    
    print("\n" + "="*40)
    print("Kết quả cuối cùng Task 9 (TL Warm-Start)")
    print("="*40)
    print(f"Iterations (Baseline): {len(rates_baseline)}")
    print(f"Iterations (TL Warm-Start): {len(rates_tl)}")
    print(f"Final Rate (Baseline): {rates_baseline[-1]:.4f}")
    print(f"Final Rate (TL Warm-Start): {rates_tl[-1]:.4f}")
    
    if len(rates_tl) < len(rates_baseline):
        print("\n=> TL Warm-Start ĐÃ TĂNG TỐC hội tụ.")
    else:
        print("\n=> TL Warm-Start KHÔNG LÀM TĂNG TỐC hội tụ.")
        
    print("Đồ thị hội tụ đã được lưu tại output/T9_TL_SCA_Convergence.pdf")
