import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
from sklearn.preprocessing import StandardScaler

st.title('Adidas Fashion Retail Analysis')

url = "Data_Cleaned.csv"
df = pd.read_csv(url)

st.subheader("dataset")
st.write(df.head())

st.subheader('Distribusi harga barang')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['selling_price'], bins=20, kde=True, color='blue', ax=ax)
ax.set_title('Distribusi harga barang')
ax.set_xlabel('Harga')
ax.set_ylabel('Jumlah')
st.pyplot(fig)

sales_counts = df['selling_price'].value_counts().sort_index()
total_saless = sales_counts.sum()

sales_counts_rounded = sales_counts.round().astype(int)
sales_percentages = (sales_counts / total_saless) * 100

st.write("Distribusi harga barang:")
st.write(sales_counts_rounded)

st.write('Visualisasi diatas memberikan gambaran tentang berapa banyak produk Adidas yang tersedia pada berbagai rentang harga jual. Rentang harga tersebut berkisar dari 9 dollar hingga 240 dollar. Distribusi ini menunjukkan variasi dalam jumlah produk yang tersedia untuk setiap harga, dengan beberapa harga memiliki hanya satu produk yang tersedia, sementara yang lain memiliki puluhan bahkan ratusan produk.')
st.write('1. Optimasi Persediaan untuk Memenuhi Permintaan Pasar: Dengan memahami pola distribusi produk berdasarkan harga, kita dapat mengoptimalkan persediaan untuk mencocokkan permintaan pasar yang sebenarnya. Fokusnya bisa pada meningkatkan stok untuk rentang harga yang paling diminati, seperti 20 dollar hingga 50 dollar, sementara kita juga mempertimbangkan untuk menyesuaikan stok untuk produk dengan permintaan yang lebih rendah atau harga yang kurang diminati.')
st.write('2. Analisis Mendalam Produk dengan Harga Ekstrim: Produk dengan harga yang sangat rendah atau tinggi membutuhkan evaluasi khusus. Kita akan menganalisis kinerja penjualan produk di kedua ujung spektrum harga ini untuk memahami alasan di baliknya. Apakah produk dengan harga rendah memberikan margin keuntungan yang memadai, atau produk dengan harga tinggi menarik pelanggan khusus yang perlu dipelajari lebih lanjut.')
st.write('3. Pengembangan Strategi Penetapan Harga yang Lebih Efektif: Distribusi ini memberikan wawasan berharga tentang preferensi harga pelanggan kita. Kita akan menggunakan df ini untuk mengembangkan strategi penetapan harga yang lebih efektif, termasuk penawaran diskon yang ditargetkan atau menetapkan harga yang lebih kompetitif untuk produk dalam kisaran harga yang paling diminati.')

st.subheader('Distribusi rating rata-rata')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['average_rating'], bins=20, kde=True, color='blue', ax=ax)
ax.set_title('Distribusi rating rata-rata')
ax.set_xlabel('rating')
ax.set_ylabel('Jumlah')
st.pyplot(fig)

rating_counts = df['average_rating'].value_counts().sort_index()
total_ratings = rating_counts.sum()

rating_counts_rounded = rating_counts.round().astype(int)
rating_percentages = (rating_counts / total_ratings) * 100

st.write("Distribusi rating rata-rata:")
st.write(rating_counts_rounded)

st.write("Visualisasi diatas menunjukkan distribusi jumlah produk Adidas berdasarkan ratingnya. Rentang rating berada antara 1 hingga 5, dengan jumlah produk yang bervariasi untuk setiap rating. Rating 4.8 memiliki jumlah produk terbanyak, sedangkan rating 3 memiliki jumlah produk yang paling sedikit.")
st.write('1. Fokus pada Produk dengan Rating Tinggi: Produk dengan rating 4.8 dan 4.9 memiliki jumlah yang signifikan. Ini menunjukkan bahwa produk-produk ini mungkin memiliki kualitas yang tinggi dan mendapat sambutan positif dari pelanggan. Perhatikan dan kembangkan produk-produk ini lebih lanjut untuk mempertahankan atau meningkatkan rating mereka.')
st.write('2. Perbaiki Produk dengan Rating Rendah: Produk dengan rating di bawah 4, terutama yang mendekati rating 3, membutuhkan perhatian khusus. Lakukan analisis mendalam terhadap produk-produk ini untuk memahami alasan di balik rating rendahnya. Perbaiki kekurangan produk dan lakukan perbaikan produk atau layanan untuk meningkatkan pengalaman pelanggan.')
st.write('3. Tingkatkan Interaksi dan Umpan Balik Pelanggan: Mendorong pelanggan untuk memberikan ulasan dan rating mereka dapat membantu meningkatkan transparansi dan kepercayaan pelanggan. Melalui umpan balik yang diberikan, Anda dapat memperoleh wawasan berharga untuk memperbaiki produk Anda dan meningkatkan kepuasan pelanggan.')

availability_count = df['availability_InStock'].sum(), df['availability_OutOfStock'].sum()
total_products = sum(availability_count)
percentage_in_stock = (availability_count[0] / total_products) * 100
percentage_out_of_stock = (availability_count[1] / total_products) * 100

st.subheader("Ketersediaan Barang")

fig, ax = plt.subplots()
ax.pie(availability_count, labels=['Tersedia', 'Tidak Tersedia'], autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig)

st.write(f"Tersedia: {percentage_in_stock:.2f}%")
st.write(f"Tidak Tersedia: {percentage_out_of_stock:.2f}%")

st.write("Data menunjukkan bahwa sebagian besar produk Adidas (99.64%) tersedia dalam stok, sedangkan hanya sebagian kecil (0.36%) yang tidak tersedia atau di luar stok. Hal ini menunjukkan bahwa sebagian besar produk Adidas cenderung tersedia bagi konsumen yang ingin membelinya.")
st.write(' Karena sebagian besar produk tersedia dalam stok, fokuskan upaya untuk memastikan ketersediaan produk yang stabil dan memadai. Lakukan pemantauan secara teratur terhadap stok yang menipis dan pastikan untuk melakukan restok yang tepat waktu agar tidak kehilangan pelanggan karena ketersediaan produk.')

color_counts = df.filter(like='color_').sum()
color_percentages = (color_counts / total_products) * 100

# Plot bar chart for color
plt.figure(figsize=(10, 6))
color_counts.sort_values(ascending=False).plot(kind='bar')
plt.xlabel('warna')
plt.ylabel('Jumlah Produk')
plt.xticks(rotation=45, ha='right')
color_chart = plt.gcf()

# Display the chart using st.pyplot()
st.subheader("Distribusi Warna Produk")
st.pyplot(color_chart)
st.write("Presentase Produk Berdasarkan Warna:")
for color, percentage in color_percentages.items():
    st.write(f"- {color}: {percentage:.2f}%")

st.write("Data presentase warna produk Adidas menunjukkan distribusi variasi warna untuk produk-produk mereka. Warna putih merupakan warna yang paling dominan dengan presentase sebesar 26.27%, diikuti oleh warna hitam dengan presentase 22.13%. Sementara itu, warna cokelat, perak, dan biru memiliki presentase yang lebih rendah, masing-masing kurang dari 1%.")
st.write('1. Fokus pada Warna yang Populer: Warna putih dan hitam merupakan warna yang paling dominan dalam produk Adidas. Fokuskan upaya produksi dan pemasaran pada produk dengan warna-warna ini karena mereka mungkin memiliki permintaan yang lebih tinggi dari pelanggan.')
st.write('2. Pantau Tren Warna: Lakukan pemantauan secara berkala terhadap tren warna yang muncul di pasar. Jika terjadi perubahan tren, seperti peningkatan popularitas warna tertentu, maka pertimbangkan untuk menyesuaikan produksi dan stok produk Anda untuk mengikuti tren tersebut.')

category_counts = df.filter(like='category_').sum()

# Calculate total number of products
total_products = category_counts.sum()

# Calculate percentage for each category
category_percentages = (category_counts / total_products) * 100

# Plot pie chart for category
plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Distribution of Product Categories")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
category_chart = plt.gcf()

# Display the chart using st.pyplot()
st.subheader("Kategori Produk")
st.pyplot(category_chart)

st.write("Data presentase produk Adidas menunjukkan distribusi produk berdasarkan kategori, dengan kategori Shoes menjadi yang paling dominan dengan 50.41% dari total produk, diikuti oleh Clothing dengan 39.88%, dan Accessories dengan 9.70%.")
st.write('1. Fokus pada Produk Sepatu: Kategori "Shoes" merupakan yang paling dominan, sehingga fokuskan upaya pemasaran dan pengembangan produk pada produk sepatu. Perhatikan tren dan preferensi pelanggan dalam hal desain, teknologi, dan kenyamanan untuk terus memperkuat posisi Anda di pasar sepatu.')
st.write('2. Pertimbangkan Diversifikasi: Meskipun produk sepatu mendominasi, jangan abaikan kategori lainnya. Pertimbangkan untuk diversifikasi portofolio Anda dengan mengembangkan produk-produk baru dalam kategori "Clothing" dan "Accessories". Ini dapat membantu menarik lebih banyak pelanggan dan mengurangi risiko bergantung pada satu kategori produk saja.')

breadcrumb_counts = df.filter(like='breadcrumbs_').sum()

# Calculate total number of products
total_products = breadcrumb_counts.sum()

# Calculate percentage for each breadcrumb
breadcrumb_percentages = (breadcrumb_counts / total_products) * 100

# Plot bar chart for breadcrumbs
plt.figure(figsize=(10, 6))
breadcrumb_counts.sort_values(ascending=False).plot(kind='bar')
plt.xlabel('Breadcrumb')
plt.ylabel('Number of Products')
plt.xticks(rotation=45, ha='right')
breadcrumb_chart = plt.gcf()

st.subheader("Distribusi Produk berdasarkan breadcrumbs")
st.pyplot(breadcrumb_chart)
st.write("Presentasi Produk Berdasarkan breadcrumb:")
for breadcrumb, percentage in breadcrumb_percentages.items():
    st.write(f"- {breadcrumb}: {percentage:.2f}%")

st.write('Data presentase produk Adidas berdasarkan breadcrumbs menunjukkan distribusi produk dalam berbagai kategori dan subkategori. Beberapa breadcrumbs memiliki persentase yang lebih tinggi, seperti "breadcrumbs_Men/Shoes" dengan 16.80% dan "breadcrumbs_Women/Clothing" dengan 20.83%, sementara yang lain memiliki persentase yang lebih rendah.')
st.write('1. Fokus pada Kategori yang Paling Dominan: Kategori "breadcrumbs_Women/Clothing" memiliki persentase tertinggi, menunjukkan bahwa produk pakaian wanita adalah area yang penting untuk difokuskan. Pertimbangkan untuk mengembangkan lebih banyak produk dalam kategori ini dan meluncurkan kampanye pemasaran yang ditargetkan untuk menarik pelanggan wanita.')
st.write('2. Penyesuaian Penawaran Produk: Berdasarkan persentase breadcrumbs, Anda dapat menyesuaikan penawaran produk Anda untuk mencerminkan preferensi pelanggan. Misalnya, jika kategori "breadcrumbs_Kids/Shoes" memiliki persentase yang tinggi, pertimbangkan untuk menawarkan lebih banyak variasi sepatu anak-anak atau mengembangkan koleksi sepatu yang lebih luas.')

st.subheader("Agglomerative Clustering")
def load_data(file_path):
    return pd.read_csv(file_path)

# Perform Agglomerative Clustering
def perform_clustering(data, n_clusters):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    pca = PCA(n_components=2)  # Reduksi dimensi menjadi 2
    reduced_data = pca.fit_transform(scaled_data)

    model = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = model.fit_predict(reduced_data)

    return clusters, reduced_data

# Sidebar for selecting number of clusters
st.sidebar.header('Select Number of Clusters')
n_clusters = st.sidebar.slider('Jumlah Cluster', min_value=2, max_value=10, value=2, step=1)

# Load data
file_path = 'Data_Cleaned.csv'
data = load_data(file_path)

# Perform Agglomerative Clustering
clusters, reduced_data = perform_clustering(data, n_clusters)

# Add cluster information to the DataFrame
data['Cluster'] = clusters

# Display the clustering results
st.write(f"Hasil Clusters dengan jumlah {n_clusters} Clusters:")
st.write(data)

# Visualize clustered data points
plt.figure(figsize=(10, 8))
for cluster_num in range(n_clusters):
    cluster_data = reduced_data[data['Cluster'] == cluster_num]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_num}')

plt.title('Visualisasi Kluster')
plt.legend()

st.pyplot(plt.gcf())