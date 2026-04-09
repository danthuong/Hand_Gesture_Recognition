import pandas as pd

def get_missings_percentage(df, columns):
    """
    Tính phần trăm dữ liệu bị thiếu (missing values) cho từng cột trong DataFrame.
    """
    missing_data = df[columns].isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    
    result = pd.DataFrame({
        'Total Missing': missing_data,
        'Percentage (%)': missing_percentage
    })
    
    # Chỉ hiển thị các cột có dữ liệu bị thiếu
    result = result[result['Total Missing'] > 0].sort_values(by='Percentage (%)', ascending=False)
    
    if result.empty:
        print("Không có dữ liệu bị thiếu trong các cột được chọn.")
    else:
        return result
