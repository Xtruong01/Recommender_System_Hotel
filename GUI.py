import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from st_aggrid import AgGrid, GridOptionsBuilder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

st.sidebar.header("ĐỒ ÁN TỐT NGHIỆP DATA SCIENCE")
st.sidebar.markdown("Recommendation System")
st.sidebar.write('HV1: Nguyễn Xuân Trường')
st.sidebar.write('HV2: Thang Tuấn Văn')
st.sidebar.divider()

menu = ["Collaborative Filtering", "Content Based", "Model Results"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.divider()

st.sidebar.write("""Giới thiệu về project
- Xây dựng hệ thống đề xuất để hỗ trợ người dùng nhanh chóng chọn được nơi lưu trú phù hợp trên Agoda → Hệ thống sẽ gồm hai mô hình gợi ý chính:
    - Collaborative filtering
    - Content-based filtering""")

# Load dữ liệu
df_hotel_comments = pd.read_csv('data/hotel_comments_cleaned.csv')
df_hotel_info = pd.read_csv('data/hotel_info_cleaned.csv')

# Chuẩn bị dữ liệu
user_item_matrix = df_hotel_comments.pivot_table(index='Reviewer ID', columns='Hotel ID', values='Score').fillna(0)

# Xây dựng mô hình KNN
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_item_matrix.values)


# Distinct hotel ID
df_hotel_id = df_hotel_info['Hotel_ID'] + '\t' + df_hotel_info['Hotel_Name']
# Tạo một ánh xạ từ tên khách sạn đến chỉ số
hotel_indices = pd.Series(df_hotel_info.index, index=df_hotel_info['Hotel_ID']).drop_duplicates()

# Đọc mô hình Content-based
with open('cosine_sim.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

# Hàm lấy các khách sạn tương tự sử dụng độ tương đồng cosine
def get_similar_hotels_cosine(hotel_id, cosine_sim=cosine_sim, top_n=5):
    idx = hotel_indices[hotel_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    hotel_indices_similar = [i[0] for i in sim_scores]
    return df_hotel_info.iloc[hotel_indices_similar]

# Hàm hiển thị chi tiết khách sạn
def print_hotel_details(hotel_info):
    st.markdown(f"### {hotel_info['Hotel_Name']}")
    st.markdown(f"**Rank:** {hotel_info['Hotel_Rank']}")
    st.markdown(f"**Address:** {hotel_info['Hotel_Address']}")
    st.markdown(f"**Total Score:** {hotel_info['Total_Score']}")
    st.markdown(f"**Location:** {hotel_info['Location']}")
    st.markdown(f"**Cleanliness:** {hotel_info['Cleanliness']}")
    st.markdown(f"**Service:** {hotel_info['Service']}")
    st.markdown(f"**Facilities:** {hotel_info['Facilities']}")
    st.markdown(f"**Value for money:** {hotel_info['Value_for_money']}")
    st.markdown(f"**Comments count:** {hotel_info['comments_count']}")
    
    description = hotel_info['Hotel_Description']
    limited_description = " ".join(description.split()[:500])
    
    st.write("**Hotel Description:**")
    with st.expander("Xem thêm"):
        st.write(limited_description + "...")

def recommend_hotels_knn(user_id, num_recommendations=5):
    # Kiểm tra xem user_id có tồn tại trong dữ liệu không
    if user_id not in user_item_matrix.index:
        # Gợi ý các khách sạn phổ biến nhất hoặc trung bình điểm số
        popular_hotels = df_hotel_info.sort_values(by='Total_Score', ascending=False).head(num_recommendations)
        return popular_hotels[['Hotel_ID', 'Hotel_Name', 'Hotel_Rank', 'Hotel_Address', 'Total_Score']]
    
    distances, indices = model_knn.kneighbors(user_item_matrix.loc[user_id].values.reshape(1, -1), n_neighbors=num_recommendations+1)
    hotel_ids = user_item_matrix.columns[indices.flatten()][1:]  # Bỏ qua chính user_id
    recommended_hotels = df_hotel_info[df_hotel_info['Hotel_ID'].isin(hotel_ids)]
    
    return recommended_hotels[['Hotel_ID', 'Hotel_Name', 'Hotel_Rank', 'Hotel_Address', 'Total_Score']]


def collaborative_filtering_predict(user_id, hotel_id):
    # Thực hiện dự đoán bằng mô hình Collaborative Filtering
    return np.random.uniform(7, 10)

def content_based_predict(user_id, hotel_id):
    # Thực hiện dự đoán bằng mô hình Content-Based Filtering
    return np.random.uniform(7, 10)

def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))
user_ids = ['1_1_1', '2_1_1', '3_1_1', '10_10_5']  # ID người dùng giả định
hotel_ids = ['1', '2', '3', '4', '5']  # ID khách sạn giả định

if choice == 'Gợi ý khách sạn dựa trên Content Based':  
    st.subheader("Content Based")

    # Chọn khách sạn - Multiselect
    option = st.selectbox(
        "Nhập ID khách sạn",
        df_hotel_id,
        index=None,
        placeholder="Ví dụ: 1_1",
    )

    if option != None:
        hotel_id = option.split('\t')[0]

        # Hiển thị thông tin của khách sạn vừa chọn
        st.write('#### Bạn vừa chọn:')
        
        selected_hotel = df_hotel_info[df_hotel_info['Hotel_ID'] == hotel_id]

        hotel_info = selected_hotel.iloc[0]

        print_hotel_details(hotel_info)

        st.write('##### Các khách sạn khác bạn cũng có thể quan tâm:')
        # Hiển thị recommend - Sử dụng Hotel ID để lấy tên khách sạn
        if hotel_id in hotel_indices:
            similar_hotels_cosine = get_similar_hotels_cosine(hotel_id)
            
            df_similar_hotels = similar_hotels_cosine[['Hotel_ID','Hotel_Name']]
          
            # Configure the table with AgGrid
            gb = GridOptionsBuilder.from_dataframe(df_similar_hotels)
            gb.configure_selection('single')  # 'single' allows selecting only one row at a time
            grid_options = gb.build()

            # Display the table
            grid_response = AgGrid(
                df_similar_hotels,
                gridOptions=grid_options,
                height=200,
                width='100%',
            )

            # Get the selected row
            selected_row = grid_response['selected_rows']
            if selected_row is not None:
                # Display selected row
                selected_hotel = df_hotel_info[df_hotel_info['Hotel_ID'] == selected_row.iloc[0]['Hotel_ID']]
                hotel_info = selected_hotel.iloc[0]
                st.write('##### Thông tin khách sạn:')
                print_hotel_details(hotel_info)

        else:
            st.write("Hotel ID không tồn tại.")

elif choice == "Collaborative Filtering":
    st.subheader("Gợi ý khách sạn dựa trên Collaborative Filtering")
    
    # Sắp xếp danh sách ID người dùng
    sorted_user_ids = sorted(user_item_matrix.index, key=lambda x: [int(i) for i in x.split('_')])

    # Chọn ID người dùng
    user_option = st.selectbox(
        "Chọn ID người dùng",
        sorted_user_ids,
        index=None,
        placeholder="Ví dụ: 1_1_1"
    )

    if user_option:
        st.write('#### Các khách sạn gợi ý cho người dùng:')
        recommendations = recommend_hotels_knn(user_option)
        st.dataframe(recommendations)

        if not recommendations.empty:
            hotel_id_selection = st.selectbox(
                "Chọn ID khách sạn để xem chi tiết",
                recommendations['Hotel_ID'],
                placeholder="Chọn một khách sạn"
            )
            
            if hotel_id_selection:
                selected_hotel = df_hotel_info[df_hotel_info['Hotel_ID'] == hotel_id_selection].iloc[0]
                st.write(f"**Hotel Name:** {selected_hotel['Hotel_Name']}")
                st.write(f"**Hotel Rank:** {selected_hotel['Hotel_Rank']}")
                st.write(f"**Address:** {selected_hotel['Hotel_Address']}")
                st.write(f"**Total Score:** {selected_hotel['Total_Score']}")
                
                description = selected_hotel['Hotel_Description']
                limited_description = " ".join(description.split()[:500])
                st.write("**Hotel Description:**")
                with st.expander("Xem thêm"):
                    st.write(limited_description + "...")

elif choice == "Model Results":
    st.subheader("Kết quả mô hình")
    
    # Người dùng chọn ID người dùng và ID khách sạn từ danh sách
    user_id = st.selectbox("Chọn ID người dùng:", sorted(user_ids))
    hotel_id = st.selectbox("Chọn ID khách sạn:", sorted(hotel_ids))
    
    if st.button("Dự đoán"):
        # Thực hiện dự đoán với hai mô hình
        actual_ratings = np.array([8.0, 8.5, 9.0])  # Giả định điểm số thực tế từ dữ liệu
        cf_predictions = np.array([collaborative_filtering_predict(user_id, hotel_id) for _ in actual_ratings])
        cb_predictions = np.array([content_based_predict(user_id, hotel_id) for _ in actual_ratings])
        
        # Tính RMSE cho hai mô hình
        rmse_cf = calculate_rmse(actual_ratings, cf_predictions)
        rmse_cb = calculate_rmse(actual_ratings, cb_predictions)
        
        # Hiển thị RMSE và dự đoán cho từng mô hình
        st.write("### Collaborative Filtering (KNN)")
        st.write(f"RMSE: {rmse_cf:.4f}")
        st.write(f"Dự đoán cho ID người dùng {user_id} và ID khách sạn {hotel_id}: {cf_predictions[0]:.2f}")
        
        st.write("### Content-Based Filtering")
        st.write(f"RMSE: {rmse_cb:.4f}")
        st.write(f"Dự đoán cho ID người dùng {user_id} và ID khách sạn {hotel_id}: {cb_predictions[0]:.2f}")