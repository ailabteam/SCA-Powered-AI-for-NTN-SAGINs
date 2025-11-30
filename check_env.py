import numpy as np
import cvxpy as cp
import torch
import sys

def check_cuda_and_torch():
    """Kiểm tra PyTorch và trạng thái CUDA."""
    print("--- 1. Kiểm tra PyTorch và CUDA ---")
    print(f"Phiên bản Python: {sys.version}")
    print(f"Phiên bản PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA khả dụng: CÓ")
        print(f"Số lượng GPU: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        # Thử tạo một Tensor trên GPU
        try:
            x = torch.randn(2, 2).cuda()
            print(f"  Tensor đã được tạo trên GPU (device: {x.device})")
        except Exception as e:
            print(f"  Lỗi khi tạo tensor trên GPU: {e}")
    else:
        print("CUDA khả dụng: KHÔNG (Kiểm tra lại cài đặt)")
        
    print("-" * 30)

def check_cvxpy():
    """Kiểm tra CVXPY và khả năng giải bài toán lồi đơn giản."""
    print("--- 2. Kiểm tra CVXPY ---")
    try:
        # Bài toán lồi đơn giản: min x^2 + y^2 s.t. x + y = 1
        x = cp.Variable()
        y = cp.Variable()
        objective = cp.Minimize(cp.square(x) + cp.square(y))
        constraints = [x + y == 1]
        problem = cp.Problem(objective, constraints)
        
        # Giải bài toán
        problem.solve()
        
        print(f"Phiên bản CVXPY: {cp.__version__}")
        print(f"Trạng thái giải: {problem.status}")
        print(f"Giá trị tối ưu: {problem.value:.4f}")
        print(f"Nghiệm (x, y): ({x.value:.4f}, {y.value:.4f})")
        
        if problem.status in ["optimal", "optimal_inaccurate"]:
            print("CVXPY hoạt động tốt.")
        else:
            print("CVXPY gặp vấn đề khi giải.")
            
    except Exception as e:
        print(f"Lỗi CVXPY: {e}")
        
    print("-" * 30)

def check_numpy_and_plotting():
    """Kiểm tra Numpy và Matplotlib (không show, chỉ save)."""
    print("--- 3. Kiểm tra Numpy và Matplotlib ---")
    try:
        import matplotlib.pyplot as plt
        
        X = np.linspace(0, 2 * np.pi, 100)
        Y = np.sin(X)
        
        plt.figure()
        plt.plot(X, Y)
        plt.title("Numpy and Matplotlib Check")
        plt.xlabel("X Axis")
        
        output_file = "test_plot.pdf"
        plt.savefig(output_file)
        plt.close()
        
        print(f"Numpy và Matplotlib hoạt động. Đồ thị mẫu đã lưu tại: {output_file}")
    except Exception as e:
        print(f"Lỗi Numpy/Matplotlib: {e}")
        
    print("-" * 30)


if __name__ == "__main__":
    check_cuda_and_torch()
    check_cvxpy()
    check_numpy_and_plotting()
    print("Kiểm tra môi trường hoàn tất.")
