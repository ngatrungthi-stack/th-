import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.random_forest import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# CẤU HÌNH TRANG & KHỞI TẠO SESSION STATE
# ==========================================
st.set_page_config(
    page_title="Phát Hiện Giao Dịch Gian Lận",
    page_icon="🛡️",
    layout="wide"
)

# Khởi tạo các biến lưu trữ toàn cục trong session_state
if 'df' not in st.state:
    st.session_state['df'] = None
if 'models' not in st.session_state:
    st.session_state['models'] = {}
if 'metrics' not in st.session_state:
    st.session_state['metrics'] = {}
if 'is_trained' not in st.session_state:
    st.session_state['is_trained'] = False
if 'X_train' not in st.session_state:
    st.session_state['X_train'] = None
if 'X_test' not in st.session_state:
    st.session_state['X_test'] = None
if 'y_train' not in st.session_state:
    st.session_state['y_train'] = None
if 'y_test' not in st.session_state:
    st.session_state['y_test'] = None

# Dữ liệu mẫu khởi tạo (Dựa trên cấu trúc dataset1.csv từ notebook)
def load_sample_data():
    # Tạo dữ liệu ngẫu nhiên mô phỏng cấu trúc của bài toán
    np.random.seed(42)
    n_samples = 1000
    data = {}
    for i in range(1, 15):
        data[f'X_{i}'] = np.random.randn(n_samples) * (i * 0.5) + (i * 0.2)
    
    # Biến mục tiêu default (0: bình thường, 1: rủi ro/gian lận) với tỉ lệ mất cân bằng ~10%
    data['default'] = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    return pd.DataFrame(data)

# Nếu chưa có dữ liệu nào, nạp dữ liệu mặc định mô phỏng ban đầu
if st.session_state['df'] is None:
    st.session_state['df'] = load_sample_data()

# ==========================================
# THÀNH PHẦN 1: SIDEBAR (ĐIỀU HƯỚNG TÀI NGUYÊN)
# ==========================================
with st.sidebar:
    st.title("🛡️ Quản Lý Dữ Liệu")
    st.subheader("Tải Tập Dữ Liệu Lên")
    uploaded_file = st.file_uploader("Chọn file CSV hoặc Excel", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state['df'] = pd.read_csv(uploaded_file)
            else:
                st.session_state['df'] = pd.read_excel(uploaded_file)
            st.success(f"Tải thành công file: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Lỗi khi đọc file: {e}")
            
    st.divider()
    st.markdown("""
    **Thông tin bài toán:**
    - **Mục tiêu:** Dự báo rủi ro gian lận / vỡ nợ tài chính (`default`).
    - **Đặc trưng:** Gồm 14 biến chỉ số từ `X_1` đến `X_14`.
    """)

# ==========================================
# VÙNG CHÍNH: TIÊU ĐỀ & ĐIỀU PHỐI TAB
# ==========================================
st.title("🛡️ Ứng Dụng Phân Tích & Phát Hiện Giao Dịch Gian Lận")
st.write("Giải pháp Machine Learning giúp tự động phát hiện sớm và xếp hạng các giao dịch rủi ro tài chính.")

# Khởi tạo hệ thống Tab lớn theo chuẩn kiến trúc phân vùng
tab_tong_quan, tab_tien_xu_ly, tab_huan_luyen, tab_danh_gia, tab_su_dung = st.tabs([
    "📊 TỔNG QUAN DỮ LIỆU",
    "⚙️ TIỀN XỬ LÝ DỮ LIỆU",
    "🏋️ HUẤN LUYỆN MÔ HÌNH",
    "📈 ĐÁNH GIÁ MÔ HÌNH",
    "🔮 SỬ DỤNG MÔ HÌNH"
])

# ------------------------------------------
# THÀNH PHẦN 2: TAB "TỔNG QUAN DỮ LIỆU"
# ------------------------------------------
with tab_tong_quan:
    st.header("Tổng Quan Tập Dữ Liệu Hiện Tại")
    df = st.session_state['df']
    
    # Khối 1: Metadata Cơ bản
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Số lượng bản ghi (Dòng)", df.shape[0])
    with col2:
        st.metric("Số lượng thuộc tính (Cột)", df.shape[1])
    with col3:
        target_col = 'default' if 'default' in df.columns else df.columns[-1]
        st.metric("Biến mục tiêu xác định", target_col)
        
    # Khối 2: Dataframe preview
    st.subheader("Xem trước dữ liệu")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Khối 3: Thống kê mô tả & Phân phối biến mục tiêu
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader("Thống kê mô tả các biến số (X)")
        st.dataframe(df.describe().T, use_container_width=True)
        
    with col_right:
        st.subheader("Phân phối biến mục tiêu")
        if target_col in df.columns:
            target_counts = df[target_col].value_counts().reset_index()
            target_counts.columns = ['Trạng thái', 'Số lượng']
            target_counts['Trạng thái'] = target_counts['Trạng thái'].map({0: '0: Bình thường', 1: '1: Gian lận/Rủi ro'})
            
            fig = px.pie(target_counts, values='Số lượng', names='Trạng thái', 
                         color='Trạng thái',
                         color_discrete_map={'0: Bình thường': '#1f77b4', '1: Gian lận/Rủi ro': '#d62728'},
                         hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Không tìm thấy cột phân loại mục tiêu mục định.")

# ------------------------------------------
# THÀNH PHẦN 3: TAB "TIỀN XỬ LÝ DỮ LIỆU"
# ------------------------------------------
with tab_tien_xu_ly:
    st.header("Cấu Hình Tiền Xử Lý & Chia Tách Dữ Liệu")
    df = st.session_state['df']
    
    st.markdown("""
    **Quy trình chuẩn hóa:**
    - Tách các biến độc lập $X$ (từ `X_1` đến `X_14`) và biến phụ thuộc $y$ (`default`).
    - Thực hiện chia nhỏ tập dữ liệu cho pha Huấn luyện (Train) và Kiểm thử (Test).
    """)
    
    # Lựa chọn tỷ lệ Split dữ liệu bằng Slider
    test_size = st.slider("Tỉ lệ tập Kiểm thử (Test size)", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    random_state = st.number_input("Random State (Đảm bảo tính nhất quán kết quả)", value=42, step=1)
    
    if st.button("Thực hiện Tiền Xử Lý & Chia Tách", type="primary"):
        # Giả định cột mục tiêu là 'default' theo dữ liệu
        if 'default' in df.columns:
            X = df.drop(columns=['default'])
            y = df['default']
        else:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        # Lưu vào session state để dùng cho các tab sau
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        
        st.success("✅ Đã xử lý phân tách dữ liệu thành công!")
        
        # Hiển thị thông số sau chia tách
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"**Tập Huấn Luyện (Train Set):** {X_train.shape[0]} mẫu dữ liệu dòng.")
        with c2:
            st.info(f"**Tập Kiểm Thử (Test Set):** {X_test.shape[0]} mẫu dữ liệu dòng.")

# ------------------------------------------
# THÀNH PHẦN 4: TAB "HUẤN LUYỆN MÔ HÌNH"
# ------------------------------------------
with tab_huan_luyen:
    st.header("Huấn Luyện Các Thuật Toán Học Máy")
    
    if st.session_state['X_train'] is None:
        st.warning("⚠️ Vui lòng thực hiện bước 'Tiền Xử Lý & Chia Tách Dữ Liệu' ở Tab trước.")
    else:
        st.write("Hệ thống sẽ chạy đồng thời 3 thuật toán phân lớp được cấu hình chuẩn xác từ Notebook thử nghiệm:")
        
        # Lấy dữ liệu từ session state
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        
        if st.button("🚀 Bắt Đầu Huấn Luyện Hệ Thống Mô Hình", type="primary"):
            with st.spinner("Hệ thống đang tính toán huấn luyện mô hình... Vui lòng đợi."):
                
                # 1. Logistic Regression
                model1 = LogisticRegression(max_iter=1000, random_state=42)
                model1.fit(X_train, y_train)
                y_pred1 = model1.predict(X_test)
                
                # 2. Decision Tree
                model2 = DecisionTreeClassifier(random_state=42)
                model2.fit(X_train, y_train)
                y_pred2 = model2.predict(X_test)
                
                # 3. Random Forest
                model3 = RandomForestClassifier(random_state=42)
                model3.fit(X_train, y_train)
                y_pred3 = model3.predict(X_test)
                
                # Lưu mô hình vào session state
                st.session_state['models'] = {
                    'Logistic Regression': model1,
                    'Decision Tree': model2,
                    'Random Forest': model3
                }
                
                # Tính toán và lưu trữ Metrics đánh giá
                for name, preds in zip(['Logistic Regression', 'Decision Tree', 'Random Forest'], [y_pred1, y_pred2, y_pred3]):
                    st.session_state['metrics'][name] = {
                        'accuracy': accuracy_score(y_test, preds),
                        'precision': precision_score(y_test, preds, zero_division=0),
                        'recall': recall_score(y_test, preds, zero_division=0),
                        'f1': f1_score(y_test, preds, zero_division=0),
                        'cm': confusion_matrix(y_test, preds)
                    }
                
                st.session_state['is_trained'] = True
                st.success("🎉 Đã huấn luyện thành công tất cả 3 mô hình học máy!")

# ------------------------------------------
# THÀNH PHẦN 5: TAB "ĐÁNH GIÁ MÔ HÌNH"
# ------------------------------------------
with tab_danh_gia:
    st.header("Biểu Đồ So Sánh & Đánh Giá Chi Tiết Hiệu Năng")
    
    if not st.session_state['is_trained']:
        st.warning("⚠️ Vui lòng hoàn tất bước huấn luyện tại tab 'HUẤN LUYỆN MÔ HÌNH' trước khi xem đánh giá.")
    else:
        metrics_data = st.session_state['metrics']
        
        # Bảng tổng hợp so sánh các chỉ số vô hướng
        records = []
        for model_name, values in metrics_data.items():
            records.append({
                "Mô hình": model_name,
                "Accuracy (Độ chính xác tổng)": round(values['accuracy'], 3),
                "Precision (Độ tin cậy lớp 1)": round(values['precision'], 3),
                "Recall (Độ nhạy bắt gian lận)": round(values['recall'], 3),
                "F1-Score (Trung bình điều hòa)": round(values['f1'], 3)
            })
        df_compare = pd.DataFrame(records)
        
        st.subheader("Bảng thống kê so sánh chỉ số kiểm thử")
        st.dataframe(df_compare, use_container_width=True)
        
        # Trực quan hóa so sánh bằng biểu đồ cột Plotly
        df_melted = df_compare.melt(id_vars="Mô hình", var_name="Chỉ số", value_name="Giá trị")
        fig_metric = px.bar(df_melted, x="Chỉ số", y="Giá trị", color="Mô hình", barmode="group",
                            title="Biểu đồ trực quan so sánh hiệu năng các mô hình",
                            labels={"Giá trị": "Thang điểm (0 - 1)"},
                            color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_metric, use_container_width=True)
        
        st.divider()
        st.subheader("Chi Tiết Ma Trận Nhầm Lẫn (Confusion Matrix)")
        
        # Hiển thị ma trận nhầm lẫn của từng mô hình qua cấu trúc cột gọn gàng
        c1, c2, c3 = st.columns(3)
        cols = [c1, c2, c3]
        
        for idx, (model_name, values) in enumerate(metrics_data.items()):
            with cols[idx]:
                st.markdown(f"**{model_name}**")
                cm = values['cm']
                
                # Tạo biểu đồ Heatmap cho ma trận nhầm lẫn bằng Plotly
                fig_cm = px.imshow(cm,
                                   text_auto=True,
                                   labels=dict(x="Nhãn Dự Đoán", y="Nhãn Thực Tế"),
                                   x=['0 (Bình thường)', '1 (Rủi ro)'],
                                   y=['0 (Bình thường)', '1 (Rủi ro)'],
                                   color_continuous_scale='Blues')
                fig_cm.update_layout(width=300, height=300, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_cm, use_container_width=False)

# ------------------------------------------
# THÀNH PHẦN 6: TAB "SỬ DỤNG MÔ HÌNH"
# ------------------------------------------
with tab_su_dung:
    st.header("Ứng Dụng Mô Hình Dự Báo Dữ Liệu Khách Hàng Mới")
    
    if not st.session_state['is_trained']:
        st.warning("⚠️ Vui lòng huấn luyện mô hình thành công trước khi chạy cấu hình dự báo.")
    else:
        # Lựa chọn mô hình tốt nhất để thực hiện dự đoán
        model_options = list(st.session_state['models'].keys())
        # Tự động gợi ý Random Forest nếu có sẵn (theo notebook là mô hình tốt nhất)
        default_idx = model_options.index('Random Forest') if 'Random Forest' in model_options else 0
        
        selected_model_name = st.selectbox("Chọn mô hình sử dụng để kết xuất dự báo:", model_options, index=default_idx)
        chosen_model = st.session_state['models'][selected_model_name]
        
        # Chọn chế độ nhập liệu đầu vào
        mode = st.radio("Chọn phương thức nạp dữ liệu đầu vào:", 
                        options=["CHẾ ĐỘ 1 — NHẬP TRỰC TIẾP TỪNG KHÁCH HÀNG", "CHẾ ĐỘ 2 — TẢI FILE DANH SÁCH BATCH (EXCEL/CSV)"])
        
        # --- CHẾ ĐỘ 1: NHẬP FORM TRỰC TIẾP ---
        if mode == "CHẾ ĐỘ 1 — NHẬP TRỰC TIẾP TỪNG KHÁCH HÀNG":
            st.subheader("Nhập thông số các chỉ số giao dịch (Từ X_1 đến X_14)")
            
            # Khởi tạo form nhập liệu dựa trên các cột đặc trưng gốc
            with st.form(key="predict_single_form"):
                col_f1, col_f2, col_f3 = st.columns(3)
                
                # Giá trị test mẫu mặc định trích xuất từ kịch bản kịch bản 1 của file notebook gốc
                sample_values = [0.13, 0.03, 0.02, 0.14, 0.78, 7.75, 1.09, 0.47, 4.31, 0.14, 1.43, 2.96, 115.11, 1.02]
                input_features = {}
                
                for idx in range(1, 15):
                    # Phân bổ đều các widget vào 3 cột thiết kế gọn đẹp
                    if idx % 3 == 1:
                        with col_f1:
                            input_features[f'X_{idx}'] = st.number_input(f"Chỉ số X_{idx}", value=sample_values[idx-1], format="%.4f")
                    elif idx % 3 == 2:
                        with col_f2:
                            input_features[f'X_{idx}'] = st.number_input(f"Chỉ số X_{idx}", value=sample_values[idx-1], format="%.4f")
                    else:
                        with col_f3:
                            input_features[f'X_{idx}'] = st.number_input(f"Chỉ số X_{idx}", value=sample_values[idx-1], format="%.4f")
                            
                submit_btn = st.form_submit_button("Thực hiện phân tích rủi ro", type="primary")
                
                if submit_btn:
                    # Chuyển đổi dữ liệu input sang cấu trúc DataFrame khớp định dạng mô hình đã fit
                    input_df = pd.DataFrame([input_features])
                    prediction = chosen_model.predict(input_df)[0]
                    
                    # Tính xác suất nếu mô hình hỗ trợ
                    try:
                        prob = chosen_model.predict_proba(input_df)[0][prediction] * 100
                        prob_str = f" (Độ tin cậy: {prob:.2f}%)"
                    except:
                        prob_str = ""
                    
                    st.markdown("### Kết quả đánh giá hệ thống:")
                    if prediction == 1:
                        st.error(f"🚨 CẢNH BÁO: Giao dịch có dấu hiệu **GIAN LẬN / RỦI RO CAO**{prob_str}.")
                    else:
                        st.success(f"✅ AN TOÀN: Giao dịch được xác thực **BÌNH THƯỜNG / KHÔNG NGUY HIỂM**{prob_str}.")
                        
        # --- CHẾ ĐỘ 2: TẢI TẬP FILE ĐỒNG LOẠT ---
        else:
            st.subheader("Tải lên danh sách thông tin khách hàng mới cần kiểm định")
            st.markdown("⚠️ *Yêu cầu: File tải lên phải chứa đầy đủ 14 cột chỉ số từ X_1 đến X_14 như định dạng dữ liệu huấn luyện ban đầu.*")
            
            batch_file = st.file_uploader("Chọn file dữ liệu batch cần dự báo (.csv/.xlsx)", type=["csv", "xlsx"], key="batch_file")
            
            if batch_file is not None:
                try:
                    if batch_file.name.endswith('.csv'):
                        df_new = pd.read_csv(batch_file)
                    else:
                        df_new = pd.read_excel(batch_file)
                        
                    # Lọc lấy các cột X_1 đến X_14 bất kể người dùng có kèm cột label y hay chưa
                    required_cols = [f'X_{i}' for i in range(1, 15)]
                    
                    if not all(col in df_new.columns for col in required_cols):
                        st.error("Lỗi cấu trúc: Tập dữ liệu mới thiếu một trong các cột từ X_1 đến X_14.")
                    else:
                        X_new_payload = df_new[required_cols]
                        
                        # Thực hiện dự đoán hàng loạt
                        predictions = chosen_model.predict(X_new_payload)
                        
                        # Đính kèm kết quả dự báo trực tiếp vào bảng hiển thị
                        df_new['Dự_Báo_Kết_Quả'] = predictions
                        df_new['Trạng_Thái_Rủi_Ro'] = df_new['Dự_Báo_Kết_Quả'].map({0: 'An toàn', 1: '🚨 Rủi ro/Gian lận'})
                        
                        st.success(f"🎯 Đã quét phân tích tự động xong {len(df_new)} bản ghi dữ liệu khách hàng!")
                        
                        # Khối thống kê tổng quan của đợt Batch prediction
                        total_risk = int((predictions == 1).sum())
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.metric("Tổng số ca phát hiện rủi ro nghi vấn", f"{total_risk} ca")
                        with col_m2:
                            st.metric("Tỉ lệ rủi ro phát hiện trong tệp mới", f"{(total_risk/len(df_new))*100:.2f}%")
                            
                        st.subheader("Bảng chi tiết kết quả xử lý tự động")
                        st.dataframe(df_new, use_container_width=True)
                        
                        # Xuất file kết quả sau xử lý cho người dùng tải xuống phục vụ báo cáo
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            df_new.to_excel(writer, sheet_name='Ket_Qua_Du_Bao', index=False)
                        
                        st.download_button(
                            label="📥 Tải Xuống File Báo Cáo Kết Quả Dự Báo (Excel)",
                            data=buffer.getvalue(),
                            file_name="Ket_qua_du_bao_gian_lan_he_thong.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                except Exception as e:
                    st.error(f"Gặp lỗi trong quá trình phân tích file dữ liệu lớn: {e}")
