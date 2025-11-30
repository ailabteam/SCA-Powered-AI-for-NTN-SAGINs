import matplotlib.pyplot as plt
import os

def save_plot_as_pdf(plt_instance, filename, dpi=300):
    """
    Lưu đối tượng plot Matplotlib thành file PDF.
    plt_instance là đối tượng matplotlib.pyplot
    """
    # Lấy đường dẫn thư mục và đảm bảo nó tồn tại
    dirname = os.path.dirname(filename)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
        
    try:
        # Lấy Figure hiện tại từ đối tượng plt_instance (matplotlib.pyplot)
        fig = plt_instance.gcf() 
        fig.savefig(filename, format='pdf', dpi=dpi)
        plt_instance.close(fig) # Đóng Figure sau khi lưu
    except Exception as e:
        print(f"Error saving plot to PDF: {e}")
