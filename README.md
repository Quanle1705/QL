1.
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

# Sửa lại backend của matplotlib để nó hiển thị cửa sổ (nếu bạn chạy trong môi trường không có GUI)
# Nếu bạn đang dùng Jupyter Notebook, có thể không cần dòng này
try:
    matplotlib.use('TkAgg') 
except ImportError:
    print("TkAgg backend không có sẵn, dùng backend mặc định.")

# ======================================================================
# BƯỚC 1: Tải MỘT tín hiệu mẫu (ví dụ: tín hiệu thứ 10 trong file)
# ======================================================================
JSON_file_path = r"Post-Processing_DataSet.json" # Đường dẫn file JSON của bạn

try:
    with open(JSON_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Lấy một tín hiệu mẫu. Bạn có thể thay đổi số 10 thành số khác.
    # Thử với một người trẻ (ví dụ data[5]) và một người già (ví dụ data[100])
    sample_index = 10
    data_sig_goc = np.array(data[sample_index]["signal_Normalized"])
    sample_age = data[sample_index]["age"]
    print(f"Đang phân tích tín hiệu mẫu từ người {sample_age} tuổi.")

except Exception as e:
    print(f"Lỗi khi tải dữ liệu: {e}")
    # Tạo dữ liệu giả nếu không tải được file
    data_sig_goc = np.sin(np.linspace(0, 8*np.pi, 200)) + np.random.rand(200) * 0.2
    sample_age = "Giả định (Tải lỗi)"

# ======================================================================
# BƯỚC 2: Tính toán các đạo hàm (dùng Savitzky-Golay)
# ======================================================================

# Đây là các tham số cho bộ lọc, bạn có thể thử thay đổi
window = 11      # Chiều dài cửa sổ (phải là số lẻ)
polyorder = 3    # Bậc của đa thức (phải nhỏ hơn window)

# Tín hiệu đã làm mịn (giống .denoised)
# deriv=0 nghĩa là chỉ làm mịn, không tính đạo hàm
sig_smooth = savgol_filter(data_sig_goc, window, polyorder, deriv=0)

# Tín hiệu đạo hàm bậc 2 (Gia tốc - APG)
# deriv=2 là tính đạo hàm bậc 2
sig_deriv2 = savgol_filter(data_sig_goc, window, polyorder, deriv=2)

# Tín hiệu đạo hàm bậc 4 (Như trong bài báo Nature)
# deriv=4 là tính đạo hàm bậc 4
sig_deriv4 = savgol_filter(data_sig_goc, window, polyorder=5, deriv=4)

# ======================================================================
# BƯỚC 3: Vẽ biểu đồ để xem kết quả
# ======================================================================
print("Đang vẽ biểu đồ...")
x_axis = np.arange(len(data_sig_goc))

# Tạo 3 biểu đồ con xếp chồng lên nhau
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# --- Biểu đồ 1: Tín hiệu Gốc vs. Đã làm mịn ---
ax1.plot(x_axis, data_sig_goc, 'k-', label='Gốc', alpha=0.5) # Tín hiệu gốc (màu đen, mờ)
ax1.plot(x_axis, sig_smooth, 'r-', label='Đã làm mịn (Smoothed)') # Tín hiệu đã làm mịn (màu đỏ)
ax1.set_title(f"Tín hiệu Gốc vs. Đã làm mịn (Mẫu: {sample_age} tuổi)")
ax1.legend()
ax1.grid(True)

# --- Biểu đồ 2: Đạo hàm bậc 2 (Gia tốc - APG) ---
ax2.plot(x_axis, sig_deriv2, 'g-')
ax2.set_title('Đạo hàm bậc 2 (Gia tốc - APG)')
ax2.axhline(0, color='black', linestyle='--', linewidth=0.5) # Vẽ đường 0
ax2.grid(True)

# --- Biểu đồ 3: Đạo hàm bậc 4 (Theo bài báo Nature) ---
ax3.plot(x_axis, sig_deriv4, 'b-')
ax3.set_title('Đạo hàm bậc 4 (Theo bài báo Nature)')
ax3.axhline(0, color='black', linestyle='--', linewidth=0.5) # Vẽ đường 0
ax3.grid(True)

# Hiển thị biểu đồ
plt.xlabel('Thời gian (điểm dữ liệu)')
plt.tight_layout()
plt.show()
# ======================================================================
# BƯỚC 4: Tự động tìm các đỉnh/đáy trên Đạo hàm bậc 2 (APG)
# ======================================================================
print("\n--- Phân tích Đạo hàm bậc 2 (APG - Sóng Xanh lá) ---")

# sig_deriv2 là mảng chứa tín hiệu sóng màu xanh lá
# distance=10: Yêu cầu các đỉnh phải cách nhau ít nhất 10 điểm dữ liệu
# height=0: Chỉ tìm các đỉnh có độ cao > 0

# 1. Tìm các đỉnh (sóng a, b, c...)
peaks_idx, peaks_props = find_peaks(sig_deriv2, distance=10, height=0)
peaks_heights = peaks_props['peak_heights']

print(f"Tìm thấy {len(peaks_idx)} đỉnh (peaks) tại vị trí: {peaks_idx}")
print(f"Với độ cao tương ứng: {[round(h, 5) for h in peaks_heights]}")

# 2. Tìm các đáy (valleys)
# (Chúng ta tìm đỉnh trên tín hiệu bị lật ngược)
valleys_idx, valleys_props = find_peaks(-sig_deriv2, distance=10, height=0)
valleys_heights = -valleys_props['peak_heights'] # Trả lại dấu âm

print(f"\nTìm thấy {len(valleys_idx)} đáy (valleys) tại vị trí: {valleys_idx}")
print(f"Với độ sâu tương ứng: {[round(h, 5) for h in valleys_heights]}")

# ======================================================================
# BƯỚC 5: Tự động tìm các đỉnh/đáy trên Đạo hàm bậc 4
# ======================================================================
print("\n--- Phân tích Đạo hàm bậc 4 (Sóng Xanh dương) ---")

# sig_deriv4 là mảng chứa tín hiệu sóng màu xanh dương

# 1. Tìm các đỉnh
peaks_idx_d4, peaks_props_d4 = find_peaks(sig_deriv4, distance=10, height=0)
peaks_heights_d4 = peaks_props_d4['peak_heights']

print(f"Tìm thấy {len(peaks_idx_d4)} đỉnh (peaks) tại vị trí: {peaks_idx_d4}")
print(f"Với độ cao tương ứng: {[round(h, 6) for h in peaks_heights_d4]}")

# 2. Tìm các đáy
valleys_idx_d4, valleys_props_d4 = find_peaks(-sig_deriv4, distance=10, height=0)
valleys_heights_d4 = -valleys_props_d4['peak_heights']

print(f"\nTìm thấy {len(valleys_idx_d4)} đáy (valleys) tại vị trí: {valleys_idx_d4}")
print(f"Với độ sâu tương ứng: {[round(h, 6) for h in valleys_heights_d4]}")
print("Đã hiển thị biểu đồ. Hãy quan sát kết quả.")
 
2.
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import seaborn as sns  # Dam bao ban da import
from scipy.signal import savgol_filter, find_peaks  # Dam bao ban da import
from scipy.integrate import simpson
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Lasso

# Sua loi backend Matplotlib
try:
    matplotlib.use('TkAgg')
except ImportError:
    print("Khong tim thay backend TkAgg, su dung backend mac dinh.")

# ======================================================================
# HAM TAI DU LIEU
# ======================================================================
def load_json_all_data(json_file_path):
    #Read the JSON file and return two groups by gender
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Data classification: group by gender
    male_all_data = [item for item in data if item['Sex'] == 'Male']
    female_all_data = [item for item in data if item['Sex'] == 'Female']

    return male_all_data, female_all_data

# ======================================================================
# HAM getfeature (HAM MOI, TU DAO HAM)
# ======================================================================
def getfeature(data):
    # 13 Cot: 10 dac trung moi + DBP + SBP + Age
    num_features = 13
    feature_maxt = np.zeros((len(data), num_features))
    
    # Tham so cho bo loc
    window = 11
    poly_d2 = 3 # Bac poly cho deriv 2
    poly_d4 = 5 # Bac poly cho deriv 4

    for i in range(len(data)):
        try:
            data_sin = data[i]
            data_sig = np.array(data_sin["signal_Normalized"])

            # --- 1. TINH TOAN CAC DAO HAM ---
            sig_deriv2 = savgol_filter(data_sig, window, poly_d2, deriv=2)
            sig_deriv4 = savgol_filter(data_sig, window, poly_d4, deriv=4)

            # --- 2. TRICH XUAT TU DAO HAM BAC 2 (APG) ---
            peaks_d2_idx, peaks_d2_props = find_peaks(sig_deriv2, distance=10, height=0)
            peaks_d2_h = peaks_d2_props['peak_heights']
            
            valleys_d2_idx, valleys_d2_props = find_peaks(-sig_deriv2, distance=10, height=0)
            valleys_d2_h = -valleys_d2_props['peak_heights']

            # Gan cac song mot cach an toan
            a_wave = peaks_d2_h[0] if len(peaks_d2_h) > 0 else 0
            b_wave = peaks_d2_h[1] if len(peaks_d2_h) > 1 else 0
            c_wave = peaks_d2_h[2] if len(peaks_d2_h) > 2 else 0
            e_wave = valleys_d2_h[0] if len(valleys_d2_h) > 0 else 0
            d_wave = valleys_d2_h[1] if len(valleys_d2_h) > 1 else 0

            # Gan vao mang feature (cot 0-4)
            feature_maxt[i, 0] = b_wave / (a_wave + 1e-9) # b/a ratio
            feature_maxt[i, 1] = c_wave / (a_wave + 1e-9) # c/a ratio
            feature_maxt[i, 2] = d_wave / (a_wave + 1e-9) # d/a ratio
            feature_maxt[i, 3] = e_wave / (a_wave + 1e-9) # e/a ratio
            feature_maxt[i, 4] = a_wave # a_wave_height

            # --- 3. TRICH XUAT TU DAO HAM BAC 4 ---
            peaks_d4_idx, peaks_d4_props = find_peaks(sig_deriv4, distance=10, height=0)
            peaks_d4_h = peaks_d4_props['peak_heights']
            
            valleys_d4_idx, valleys_d4_props = find_peaks(-sig_deriv4, distance=10, height=0)
            valleys_d4_h = -valleys_d4_props['peak_heights']

            # Gan cac dac trung mot cach an toan
            feature_maxt[i, 5] = np.max(peaks_d4_h) if len(peaks_d4_h) > 0 else 0
            feature_maxt[i, 6] = np.min(valleys_d4_h) if len(valleys_d4_h) > 0 else 0
            feature_maxt[i, 7] = peaks_d4_h[0] if len(peaks_d4_h) > 0 else 0
            feature_maxt[i, 8] = valleys_d4_h[0] if len(valleys_d4_h) > 0 else 0
            feature_maxt[i, 9] = peaks_d4_h[1] if len(peaks_d4_h) > 1 else 0

            # --- 4. THEM CAC DAC TRUNG CU TOT ---
            feature_maxt[i, 10] = data_sin["DBP"]   # (index -3)
            feature_maxt[i, 11] = data_sin["SBP"]   # (index -2)
            feature_maxt[i, 12] = data_sin["age"]   # (index -1)

        except (KeyError, IndexError, Exception) as e:
            continue 

    return feature_maxt

# ======================================================================
# CAC HAM TIEN ICH (VE BIEU DO)
# ======================================================================

# Thu muc luu ket qua
RESULTS_DIR = r'D:\HuNan\SD\VA_Plot'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Ham ve bieu do Ket qua (Regression Plot)
def plot_results(true_age, pred_age, model_name, color='#1f77b4'):
    plt.rcParams.update({'font.size': 12, 'savefig.dpi': 300})
    fig, ax = plt.subplots(figsize=(8, 8))
    try:
        mae = mean_absolute_error(true_age, pred_age)
        rmse = np.sqrt(mean_squared_error(true_age, pred_age))
        r2 = r2_score(true_age, pred_age)
        r = np.corrcoef(true_age, pred_age)[0, 1]

        fit_model = LinearRegression()
        fit_model.fit(true_age.reshape(-1, 1), pred_age)
        fit_line = fit_model.predict(np.array([20, 80]).reshape(-1, 1))
        errors = pred_age - fit_model.predict(true_age.reshape(-1, 1))
        std_error = np.std(errors)
        x = np.array([20, 80])
        y_fit = fit_model.predict(x.reshape(-1, 1))

        ax.fill_between(x, y_fit - 1.96 * std_error, y_fit + 1.96 * std_error,
                        color='lightgray', alpha=0.3, label='95% confidence band')
        ax.scatter(true_age, pred_age, s=30, alpha=0.7, c=color)
        ax.plot([20, 80], fit_line, 'r-', linewidth=1.5,
                label=f'Fit line (Slope={fit_model.coef_[0]:.2f})')
        ax.plot([20, 80], [20, 80], 'k--', linewidth=1, label='Ideal Prediction')
        ax.set_xlim(20, 80)
        ax.set_ylim(20, 80)
        ax.set_xlabel('Chronological Age (Y)')
        ax.set_ylabel(f'Vascular Age ({model_name})')
        ax.set_title(f'{model_name} Model\n'
                     f'MAE={mae:.2f} years, RMSE={rmse:.2f} years, R²={r2:.2f}, R={r:.2f}')
        ax.grid(color='lightgray', linestyle='--', linewidth=0.5)
        ax.legend(loc='upper left')
        filename = f"{RESULTS_DIR}/{model_name.replace(' ', '_')}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    except Exception as e:
        print(f"Loi khi ve plot_results: {e}")
    finally:
        plt.close(fig)

# Ham ve bieu do Phan phoi Loi
def plot_error_distribution(true_age, pred_age, model_name, color='#1f77b4'):
    plt.rcParams.update({'font.size': 12, 'savefig.dpi': 300})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # <<< LOI BAT DAU O DAY: ban da dan ham moi vao ben trong ham nay >>>
    
    try:
        errors = pred_age - true_age
        ax1.scatter(true_age, errors, s=30, alpha=0.7, c=color)
        ax1.axhline(y=0, color='k', linestyle='--', linewidth=1)
        ax1.set_xlabel('Chronological Age (Y)')
        ax1.set_ylabel('Vascular Age (Y)')
        ax1.set_title(f'{model_name} - Error vs Age')
        ax1.grid(color='lightgray', linestyle='--', linewidth=0.5)

        z = np.polyfit(true_age, errors, 1)
        p = np.poly1d(z)
        ax1.plot(true_age, p(true_age), "r--", label=f'Trend (Slope={z[0]:.3f})')
        ax1.legend()

        ax2.hist(errors, bins=15, color=color, alpha=0.7, edgecolor='black', density=True)
        ax2.axvline(x=0, color='k', linestyle='--', linewidth=1)
        ax2.set_xlabel('Prediction Error (Y)')
        ax2.set_ylabel('Density')
        ax2.set_title(f'{model_name} - Error Distribution')
        ax2.grid(color='lightgray', linestyle='--', linewidth=0.5)

        mu, std = np.mean(errors), np.std(errors)
        xmin, xmax = ax2.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = (1 / (std * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * std ** 2)))
        ax2.plot(x, p, 'k', linewidth=2, label=f'Normal fit ($\mu$={mu:.2f}, $\sigma$={std:.2f})')
        ax2.legend()
        plt.tight_layout()
        filename = f"{RESULTS_DIR}/{model_name.replace(' ', '_')}_error_dist.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    except Exception as e:
        print(f"Loi khi ve plot_error_distribution: {e}")
    finally:
        plt.close(fig)

# ======================================================================
# HAM MOI: Ve Bieu do Phan phoi VA Gap (>>> VI TRI DUNG LA O DAY <<<)
# ======================================================================
def plot_va_gap_histogram(df_male, df_female):
    print("\n========== Ve Bieu do Phan phoi VA Gap (Nhom Nguy Co) ==========")
    try:
        plt.figure(figsize=(14, 7))
        
        # --- Ve cho Nam ---
        plt.subplot(1, 2, 1)
        sns.histplot(df_male['VA_Gap'], bins=20, kde=True, color='blue')
        plt.axvline(x=0, color='black', linestyle='--')
        plt.axvline(x=10, color='red', linestyle='--', label='Nguy co cao (> +10)')
        plt.axvline(x=3, color='orange', linestyle='--', label='Canh bao (> +3)')
        plt.title(f'Phan phoi VA Gap - NAM (N={len(df_male)})')
        plt.xlabel('Vascular Age Gap (Predicted - True)')
        plt.ylabel('So luong (Count)')
        plt.legend()

        # --- Ve cho Nu ---
        plt.subplot(1, 2, 2)
        sns.histplot(df_female['VA_Gap'], bins=20, kde=True, color='red')
        plt.axvline(x=0, color='black', linestyle='--')
        plt.axvline(x=10, color='red', linestyle='--', label='Nguy co cao (> +10)')
        plt.axvline(x=3, color='orange', linestyle='--', label='Canh bao (> +3)')
        plt.title(f'Phan phoi VA Gap - NU (N={len(df_female)})')
        plt.xlabel('Vascular Age Gap (Predicted - True)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/VA_Gap_Distribution.png", dpi=300)
        plt.close()
        print(f"Da luu bieu do VA Gap tai: {RESULTS_DIR}/VA_Gap_Distribution.png")
    except Exception as e:
        print(f"Loi khi ve Histogram VA Gap: {e}")
# ======================================================================
# HAM MOI: Ve Bieu do Hop (Box Plot) chi tiet VA Gap theo Nhom Tuoi
# ======================================================================
def plot_va_gap_by_age_boxplot(df_male, df_female):
    print("\n========== Ve Bieu do Hop chi tiet VA Gap theo Nhom Tuoi ==========")
    
    # Dinh nghia cac nhom tuoi (bins)
    age_bins = [0, 30, 40, 50, 60, 100]
    bin_labels = ['<30', '30-40', '40-50', '50-60', '>60']
    
    # Tao cot 'Age_Group' trong ca 2 DataFrame
    # right=False nghia la [0, 30), [30, 40), ...
    df_male['Age_Group'] = pd.cut(df_male['Age'], bins=age_bins, labels=bin_labels, right=False)
    df_female['Age_Group'] = pd.cut(df_female['Age'], bins=age_bins, labels=bin_labels, right=False)

    try:
        plt.figure(figsize=(16, 8))
        
        # --- Ve cho Nam ---
        plt.subplot(1, 2, 1)
        sns.boxplot(x='Age_Group', y='VA_Gap', data=df_male, palette='coolwarm')
        
        # Them duong tham chieu so 0
        plt.axhline(y=0, color='black', linestyle='--')
        plt.axhline(y=10, color='red', linestyle=':', label='Nguy co cao (+10)')
        plt.axhline(y=3, color='orange', linestyle=':', label='Canh bao (+3)')
        
        plt.title(f'Phan phoi VA Gap theo Nhom Tuoi - NAM (N={len(df_male)})')
        plt.xlabel('Nhom Tuoi (Tuoi that)')
        plt.ylabel('Vascular Age Gap (Predicted - True)')
        plt.legend()

        # --- Ve cho Nu ---
        plt.subplot(1, 2, 2)
        sns.boxplot(x='Age_Group', y='VA_Gap', data=df_female, palette='coolwarm')

        # Them duong tham chieu so 0
        plt.axhline(y=0, color='black', linestyle='--')
        plt.axhline(y=10, color='red', linestyle=':', label='Nguy co cao (+10)')
        plt.axhline(y=3, color='orange', linestyle=':', label='Canh bao (+3)')

        plt.title(f'Phan phoi VA Gap theo Nhom Tuoi - NU (N={len(df_female)})')
        plt.xlabel('Nhom Tuoi (Tuoi that)')
        plt.ylabel('Vascular Age Gap (Predicted - True)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/VA_Gap_by_Age_Boxplot.png", dpi=300)
        plt.close()
        print(f"Da luu bieu do Box Plot chi tiet tai: {RESULTS_DIR}/VA_Gap_by_Age_Boxplot.png")
    
    except Exception as e:
        print(f"Loi khi ve Bieu do Hop (Box Plot): {e}")
# ======================================================================
# HAM MAIN CHINH
# ======================================================================
def main():
    # --- 1. TAI VA TRICH XUAT DAC TRUNG MOI ---
    JSON_file_path = r"Post-Processing_DataSet.json"
    male_all_data, female_all_data = load_json_all_data(JSON_file_path)
    
    print("Dang trich xuat dac trung cho Nam...")
    male_all_feature = getfeature(male_all_data)
    print("Dang trich xuat dac trung cho Nu...")
    female_all_feature = getfeature(female_all_data)

    # --- 2. LAM SACH DU LIEU (RAT QUAN TRONG) ---
    
    # Dinh nghia 13 ten cot
    new_col_names = [
        'b/a_ratio', 'c/a_ratio', 'd/a_ratio', 'e/a_ratio', 'a_wave_height',
        'd4_max_peak', 'd4_max_valley', 'd4_p1', 'd4_v1', 'd4_p2',
        'DBP', 'SBP', 'Age'
    ]
    
    # --- Xu ly cho Nam (Male) ---
    df_male = pd.DataFrame(male_all_feature, columns=new_col_names)
    df_male = df_male[df_male['Age'] > 0] # Loc hang loi (Age=0)
    df_male.replace([np.inf, -np.inf], np.nan, inplace=True) # Thay the inf
    df_male.dropna(inplace=True) # Xoa bat ky hang nao co NaN
    
    # Tach X va y tu DataFrame da lam sach
    Y_male_new = df_male['Age'].values
    male_features_new = df_male.drop('Age', axis=1).values
    
    # --- Xu ly cho Nu (Female) ---
    df_female = pd.DataFrame(female_all_feature, columns=new_col_names)
    df_female = df_female[df_female['Age'] > 0] # Loc hang loi (Age=0)
    df_female.replace([np.inf, -np.inf], np.nan, inplace=True) # Thay the inf
    df_female.dropna(inplace=True) # Xoa bat ky hang nao co NaN
    
    # Tach X va y tu DataFrame da lam sach
    Y_female_new = df_female['Age'].values
    female_features_new = df_female.drop('Age', axis=1).values

    print(f"Da xu ly & lam sach. Du lieu Nam: {male_features_new.shape}, Nhan Nam: {Y_male_new.shape}")
    print(f"Da xu ly & lam sach. Du lieu Nu: {female_features_new.shape}, Nhan Nu: {Y_female_new.shape}")

    # --- 3. VE MA TRAN TUONG QUAN MOI ---
    print("\n========== Ve Ma tran Tuong quan (Dac trung moi) ==========")
   
    # Dung DataFrame da duoc lam sach o BUOC 2
    try:
        plt.figure(figsize=(16, 14)) 
        corr_male_new = df_male.corr() # Dung df_male da loc
        sns.heatmap(corr_male_new, annot=True, fmt='.2f', cmap='vlag', center=0, 
                    annot_kws={"size": 9}, linewidths=.5) 
        plt.title('Correlation Matrix - MALE (New Features)', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout() 
        plt.savefig(f"{RESULTS_DIR}/Correlation_Matrix_Male_NEW.png", dpi=300)
        plt.close()
        print(f"Da luu ma tran tuong quan MOI cua Nam tai: {RESULTS_DIR}/Correlation_Matrix_Male_NEW.png")
    except Exception as e:
        print(f"Loi khi ve Ma tran Tuong quan Nam: {e}")

    # Dung DataFrame da duoc lam sach o BUOC 2
    try:
        plt.figure(figsize=(16, 14)) 
        corr_female_new = df_female.corr() # Dung df_female da loc
        sns.heatmap(corr_female_new, annot=True, fmt='.2f', cmap='vlag', center=0, 
                    annot_kws={"size": 9}, linewidths=.5) 
        plt.title('Correlation Matrix - FEMALE (New Features)', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout() 
        plt.savefig(f"{RESULTS_DIR}/Correlation_Matrix_Female_NEW.png", dpi=300)
        plt.close()
        print(f"Da luu ma tran tuong quan MOI cua Nu tai: {RESULTS_DIR}/Correlation_Matrix_Female_NEW.png")
    except Exception as e:
        print(f"Loi khi ve Ma tran Tuong quan Nu: {e}")

    # --- 4. VE BIEU DO PHAN BO TUOI (HISTOGRAM) ---
    print("\n========== Ve Bieu do Phan bo Tuoi (Histogram) ==========")
    
    # Su dung Y_male_new va Y_female_new da duoc loc sach
    plt.figure(figsize=(12, 6))
    
    # Ve cho Nam
    plt.subplot(1, 2, 1) # 1 hang, 2 cot, bieu do thu 1
    sns.histplot(Y_male_new, bins=20, kde=True, color='blue')
    plt.title(f'Phan bo Tuoi - NAM (N={len(Y_male_new)})')
    plt.xlabel('Tuoi that (Chronological Age)')
    plt.ylabel('So luong (Count)')
    
    # Ve cho Nu
    plt.subplot(1, 2, 2) # 1 hang, 2 cot, bieu do thu 2
    sns.histplot(Y_female_new, bins=20, kde=True, color='red')
    plt.title(f'Phan bo Tuoi - NU (N={len(Y_female_new)})')
    plt.xlabel('Tuoi that (Chronological Age)')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/Age_Distribution_Histogram.png", dpi=300)
    plt.close()
    
    print(f"Da luu bieu do Histogram Phan bo Tuoi tai: {RESULTS_DIR}/Age_Distribution_Histogram.png")

    # --- 5. HUAN LUYEN MO HINH XGBOOST ---
    
    print("\n========== XGBoost Model (VOI DAC TRUNG MOI) ==========")

    # --- PHAN TINH CHINH CHO NAM (MALE) ---
    print("\n--- Bat dau Tinh chinh (Tuning) cho mo hinh Nam (Male) ---")
    
    # Dung du lieu da lam sach
    X_train_xgb_m, X_test_xgb_m, y_train_xgb_m, y_test_xgb_m = train_test_split(
        male_features_new, Y_male_new, test_size=0.3, random_state=42)

    model_xgb_base = XGBRegressor(objective='reg:squarederror', random_state=42)

    # Khong gian tham so de tim kiem
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9, 11], 
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_lambda': [1, 1.5, 2] 
    }

    random_search_male = RandomizedSearchCV(
        model_xgb_base, 
        param_distributions=param_dist, 
        n_iter=50, 
        cv=5, 
        scoring='neg_mean_absolute_error',
        n_jobs=1, 
        verbose=2, 
        random_state=42
    )

    print("Dang huan luyen RandomizedSearchCV cho Nam...")
    random_search_male.fit(X_train_xgb_m, y_train_xgb_m)

    best_model_male = random_search_male.best_estimator_
    print("\nCac tham so tot nhat cho Nam (Male) voi dac trung moi: ")
    print(random_search_male.best_params_)

    y_pred_male = best_model_male.predict(X_test_xgb_m)
    rmse_male = np.sqrt(mean_squared_error(y_test_xgb_m, y_pred_male))
    
    print(f'XGBoost Model male RMSE (voi dac trung moi): {rmse_male:.4f}')
    
    plot_results(y_test_xgb_m, y_pred_male, 'XGB-Male', color='#1f77b4')
    error_stats = plot_error_distribution(y_test_xgb_m, y_pred_male, 'XGB-Male', color='#1f77b4')

    # --- PHAN TINH CHINH CHO NU (FEMALE) ---
    print("\n--- Bat dau Tinh chinh (Tuning) cho mo hinh Nu (Female) ---")

    X_train_xgb_f, X_test_xgb_f, y_train_xgb_f, y_test_xgb_f = train_test_split(
        female_features_new, Y_female_new, test_size=0.3, random_state=42)

    model_xgb_base_female = XGBRegressor(objective='reg:squarederror', random_state=42)

    random_search_female = RandomizedSearchCV(
        model_xgb_base_female, 
        param_distributions=param_dist, 
        n_iter=50, 
        cv=5, 
        scoring='neg_mean_absolute_error',
        n_jobs=1, 
        verbose=2, 
        random_state=42
    )

    print("Dang huan luyen RandomizedSearchCV cho Nu...")
    random_search_female.fit(X_train_xgb_f, y_train_xgb_f)

    best_model_female = random_search_female.best_estimator_
    print("\nCac tham so tot nhat cho Nu (Female) voi dac trung moi: ")
    print(random_search_female.best_params_)

    y_pred_female = best_model_female.predict(X_test_xgb_f)
    rmse_female = np.sqrt(mean_squared_error(y_test_xgb_f, y_pred_female))
    
    print(f'XGBoost Model female RMSE (voi dac trung moi): {rmse_female:.4f}')
    
    plot_results(y_test_xgb_f, y_pred_female, 'XGB-Female', color='#1f77b4')
    error_stats = plot_error_distribution(y_test_xgb_f, y_pred_female, 'XGB-Female', color='#1f77b4')

    # ======================================================================
    # BUOC 6: PHAN TANG NGUY CO TIM MACH (SU DUNG MO HINH TOT NHAT)
    # ======================================================================
    print("\n========== BAT DAU PHAN TANG NGUY CO TIM MACH ==========")

    # best_model_male va best_model_female la 2 mo hinh tot nhat da duoc huan luyen o tren
    
    # --- 1. Ap dung mo hinh vao TOAN BO du lieu ---
    
    # Lay TOAN BO dac trung (X) va nhan (Y) da duoc lam sach
    X_male_all = df_male.drop('Age', axis=1).values
    Y_male_all = df_male['Age'].values
    
    X_female_all = df_female.drop('Age', axis=1).values
    Y_female_all = df_female['Age'].values
    
    # Du doan tren TOAN BO du lieu
    print("Dang du doan tren toan bo tap du lieu Nam...")
    male_pred_all = best_model_male.predict(X_male_all)
    df_male['Predicted_Age'] = male_pred_all
    
    print("Dang du doan tren toan bo tap du lieu Nu...")
    female_pred_all = best_model_female.predict(X_female_all)
    df_female['Predicted_Age'] = female_pred_all

    # --- 2. Tinh toan Vascular Age Gap (VA Gap) ---
    df_male['VA_Gap'] = df_male['Predicted_Age'] - df_male['Age']
    df_female['VA_Gap'] = df_female['Predicted_Age'] - df_female['Age']
    
    # --- 3. Dinh nghia cac nhom nguy co ---
    def stratify_risk(va_gap):
        if va_gap > 10:
            return '3. Nguy co Cao'  # Lao hoa nhanh hon > 10 tuoi
        elif va_gap > 3:
            return '2. Canh bao'     # Lao hoa nhanh hon 3-10 tuoi
        else:
            return '1. Binh thuong'  # Lao hoa binh thuong (< 3 tuoi)

    # --- 4. Tao cot Phan tang ---
    df_male['Risk_Group'] = df_male['VA_Gap'].apply(stratify_risk)
    df_female['Risk_Group'] = df_female['VA_Gap'].apply(stratify_risk)

    # --- 5. Hien thi Ket qua ---
    print("\n--- KET QUA PHAN TANG NGUY CO (NAM) ---")
    # In ra 10 hang dau tien de xem
    print(df_male[['Age', 'Predicted_Age', 'VA_Gap', 'Risk_Group']].head(10))
    # In ra bang tong hop so luong
    print("\nTong hop Nhom Nguy co (Nam):")
    print(df_male['Risk_Group'].value_counts().sort_index())

    print("\n--- KET QUA PHAN TANG NGUY CO (NU) ---")
    # In ra 10 hang dau tien de xem
    print(df_female[['Age', 'Predicted_Age', 'VA_Gap', 'Risk_Group']].head(10))
    # In ra bang tong hop so luong
    print("\nTong hop Nhom Nguy co (Nu):")
    print(df_female['Risk_Group'].value_counts().sort_index())

    # --- 6. Ve bieu do VA Gap ---
    plot_va_gap_histogram(df_male, df_female)
    # --- 6. Ve bieu do VA Gap ---
    plot_va_gap_histogram(df_male, df_female)
    
    # *** THEM DONG MOI CUA BAN VAO DAY: ***
    plot_va_gap_by_age_boxplot(df_male, df_female)

# ======================================================================
# PHAN CHAY CODE
# ======================================================================
if __name__ == "__main__":
    main()


