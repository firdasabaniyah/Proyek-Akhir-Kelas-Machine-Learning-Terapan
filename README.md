# Laporan Proyek Machine Learning - Firda Sa'baniyah

## Domain Proyek

Domain proyek yang dipilih dalam proyek akhir machine learning terapan ini adalah membuat _movie recommendation system_. 

Gambar 1. Movie

![17___film](https://user-images.githubusercontent.com/111235408/194676810-f8675b7c-6f78-414c-a107-ebb81be328a0.jpg)


_Recommendation system_ telah menjadi lazim dalam beberapa tahun terakhir karena menangani masalah kelebihan informasi dengan menyarankan pengguna produk yang paling relevan dari sejumlah besar data. Untuk produk media, _movie recommendation_ kolaboratif _online_ berupaya membantu pengguna mengakses film pilihan mereka dengan menangkap tetangga yang persis sama di antara pengguna atau film dari peringkat umum historis mereka. Namun, karena data yang jarang, pemilihan tetangga menjadi lebih sulit dengan meningkatnya film dan pengguna dengan cepat. 
 _Movie recommendation system_ merupakan sistem yang merekomendasikan film kepada penonton atau pengguna lainnya, rekomendasi ini contohnya diterapkan pada situs seperti netflix, iqiyi, dan wetv. Dalam proyek ini, diusulkan sistem rekomendasi film berbasis model _collaborative filtering_ dan _based filtering_.  Sistem rekomendasi yang saya buat ini didasarkan pada peferensi kesukaan pengguna dimasa lalu, serta rating dari film tersebut.
Hasil eksperimen pada dataset _recommendation movie_ menunjukkan bahwa pendekatan yang diusulkan dapat memberikan kinerja tinggi dalam hal akurasi, dan menghasilkan rekomendasi film yang lebih andal dan personal jika dibandingkan dengan metode yang ada.


- Film adalah salah satu media hiburan yang populer di masyarakat. Banyaknya judul-judul yang telah rilis membuat masyarakat kesulitan untuk menemukan film mana yang mereka ingin tonton. Karena hal tersebut masalah ini perlu untuk diatasi. Sehingga informasi mengenai film akan memudahkan masyarakat untuk menemukan film yang cocok dengan preferensi _user_, oleh sebab itu _user_ perlu sebuah sistem yang dapat memberikan rekomendasi film.  
- Sistem rekomendasi adalah sistem yang mampu memberikan rekomendasi item-item yang mungkin disukai oleh pengguna. Metode _Collaborative Filtering_ merupakan salah satu metode pada sistem rekomendasi. Metode ini memanfaatkan penilaian pengguna lain berupa rating atau umpan balik lain untuk memprediksi item yang mungkin diminati. _Collaborative filtering (CF)_ yang digunakan adalah metode _User Based Collaborative Filtering_. Penerapan dilakukan pada website merekomendasikan film-film untuk _user_ menonton.
- _Collaborative filtering_ merupakan sebuah metode dalam membuat prediksi dengan cara menyaring informasi item dari opini pengguna lain. Ide utama dalam sistem rekomendasi _collaborative filtering_ adalah untuk memanfaatkan riwayat opini pengguna aktif lain untuk memprediksi item yang mungkin akan disukai/diminati
oleh seorang pengguna. Implementasi yang paling sederhana dari pendekatan ini adalah membuat rekomendasi kepada pengguna aktif berdasarkan item yang disukai pengguna lain dengan riwayat selera yang serupa (Ricci, Rokach, & Shapira, 2011). 



## Business Understanding

### Problem Statements

- Bagaimana cara menemukan rekomendasi yang tepat untuk seseorang dengan pilihan genre film yang ia suka?
- Metode apa saja yang digunakan untuk membuat sistem rekomendasi?
- Bagaimana cara merekomendasikan pengguna yang memiliki kesamaan dalam menyukai suatu genre film?

### Goals

- Dapat dengan cara membuat sistem rekomendasi yang akurat berdasarkan aktivitas pengguna pada masa lalu.
- Metode yang akan digunakan pada sistem rekomendasi film ini adalah _Collaborative Filtering_ dan _Content Based Filtering_
- Dengan membangun sebuah sistem rekomendasi film yang akurat berdasarkan _ratings_ dan aktivitas pengguna lain pada masa lalu.
 

### Solution statements
    
Solusi yang saya buat yaitu dengan menggunakan 2 algoritma Machine Learning sistem rekomendasi,yaitu :
-  _Content Based Filtering_ adalah algoritma yang merekomendasikan item serupa dengan apa yang disukai pengguna, berdasarkan tindakan mereka sebelumnya atau umpan balik eksplisit.
 _Content Based Filtering_ mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna. Algoritma ini bekerja dengan menyarankan item serupa yang pernah disukai di masa lalu atau sedang dilihat di masa kini kepada pengguna. Semakin banyak informasi yang diberikan pengguna, semakin baik akurasi sistem rekomendasi.


- _Collaborative Filtering_, adalah algoritma yang bergantung pada pendapat komunitas pengguna. Dia tidak memerlukan atribut untuk setiap itemnya.
Metode _Collaborative Filtering_ merupakan salah satu metode pada sistem rekomendasi. Metode ini memanfaatkan penilaian pengguna lain berupa rating atau umpan balik lain untuk memprediksi item yang mungkin diminati. 

Bahasa sederhananya, algoritma _Content Based Filtering_ digunakan untuk merekemondesikan film berdasarkan aktivitas pengguna pada masa lalu, sedangkan algoritma _Collaborative Filtering_ digunakan untuk merekomendasikan film berdasarkan rating yang paling tinggi.

Saya menggunakan dataset dari kaggle, untuk dapat menggunakan dataset tersebut saya mengimport file kredensial kaggle dalam bentuk json terlebih dahulu. Maksud dari file kredensial ini adalah untuk mengatur _permission_ dataset yang nantinya akan saya _upload_. Saya juga mengimport _library pandas_ yang dibutuhkan untuk _upload_ dataset Movie Recommendation Data menggunakan kode API. Untuk dapat melihat dataset Movie Recommendation Data yang terdapat pada gambar 2 dataset Movie Recommendation Data.

Gambar 2. dataset Movie Recommendation Data

![Reommendation](https://user-images.githubusercontent.com/111235408/194678835-edc34e43-fd9f-45ad-95c2-ed1cdb8baeaa.png)


## Data Understanding

Setelah mendownload dataset tahap selanjutnya adalah melakukan data understanding. Agar lebih memudahkan pembaca, saya membuat tabel yang dapat dilihat di Tabel 1. data understanding

Tabel 1. data understanding

| Sumber | [Kaggle Dataset : Movie Recommendation Data](https://www.kaggle.com/datasets/rohan4050/movie-recommendation-data)|
| ------------ |---------------| 
| Lisensi | Unknown |
| Kategori | Industri Hiburan |
| Jenis dan Ukuran Berkas | ZIP 14 MB |

Dataset yang digunakan pada proyek akhir machine learning ini adalah data Movie Recommendation Data yang didapat dari situs kaggle. Link dataset tersebut, berikut tautan Dataset : Movie Recommendation Data](https://www.kaggle.com/datasets/rohan4050/movie-recommendation-data)


### Variabel-variabel pada Movie Recommendation dataset adalah sebagai berikut:

- Linkss : list link movie.
- Moviie : list movie yang tersedia.
- Ratingg :list penilaian yang diberikan penonton terhadap movie.
- Tagg : list kata kunci dari movie.

Berikut beberapa tahapan Data Understanding diantaranya sebagai berikut:

- Meload Dataset ke dalam sebuah Dataframe menggunakan pandas.
- df.info() digunakan untuk mengecek tipe kolom pada dataset.
- df.isna().sum() digunakan untuk mengecek apakah ada kolom yg kosong, pada dataset ini nilai kosong tidak ditemukan.
- df.describe() digunakan untuk mendapatkan info mengenai dataset terhadap nilai rata-rata, median, banyaknya data, nilai Q1 hingga Q3 dan lain-lain.
- len(nama_variable.unique())menghitung panjang data unique dari variable tertentu.

Tahap eksplorasi penting untuk memahami variabel-variabel pada data serta korelasi antar variabel. Pemahaman terhadap variabel pada data dan korelasinya akan membantu saya dalam menentukan pendekatan atau algoritma yang cocok untuk data saya. Saya akan melakukan eksplorasi data terhadap variabel Linkss, variabel Moviie, variabel Ratingg dan variabel Tagg.
Variabel Moviie dan ratingg akan digunakan pada model rekomendasi saya. Sedangkan, variabel Linkss dan Tagg hanya untuk melihat bagaimana linkss akses dan tagg  yang digunakan oleh pengguna. 

Mari kita mulai eksplorasi!

- Variabel Moviie

Tabel 2. Variabel Moviie

| Column | Non-Null | Count | Dtype |
| ------------ |---------------|---------------|---------------|
| movieId | 9742 | non-null | int64 |
| title |  9742 | non-null | object |
| genres | 9742 | non-null | object |

Dari tabel 2 Variable Moviie yang tertera diatas terdapat 3 kolom yang ada di variabel moviie yaitu movieId, title, dan genres. MovieId memiliki tipe data int64 sedangkan title dan genres memiliki tipe data object.

- variabel Linkss

Tabel 3. variabel Linkss

| Column | Non-Null | Count | Dtype |
| ------------ |---------------|---------------|---------------|
| movieId | 9742 | non-null | int64 |
| imdbId |  9742 | non-null | int64 |
| tmdbId | 9742 | non-null | float64 |

Dari tabel 3 Variable Linkss yang tertera diatas terdapat 3 kolom yang ada di variabel Linkss yaitu movieId, imdbId, dan tmdbId. MovieId dan imdbId memiliki tipe data int64 sedangkan tmdbId memiliki tipe data float64.

- Variabel Ratingg

Tabel 4. Variabel Ratingg

| Column | Non-Null | Count | Dtype |
| ------------ |---------------|---------------|---------------|
| userId | 100836 | non-null | int64 |
| movieId | 100836 | non-null | int64 |
| rating | 100836 | non-null | float64 |
| timestamp | 100836 | non-null | int64 |

Dari tabel 4 Variable Ratingg yang tertera diatas terdapat 4 kolom yang ada di variabel Ratingg yaitu userId, movieId, rating, dan timestamp. UserId, movieId, dan timestamp memiliki tipe data int64 sedangkan rating memiliki tipe data float64.

Untuk melihat rating describe dan memudahkan dalam melihatnya saya visualisasikan menggunakan Table 5 visualisasi rating describe dibawah.

Table. 5 visualisasi rating describe

|  | userId | movieId | rating | timestamp |
| ------------ |---------------|---------------|---------------|---------------|
| count | 100836.000000 | 100836.000000 | 100836.000000 | 1.008360e+05 |
| mean | 326.127564 | 19435.295718 | 3.501557 | 1.205946e+09 |
| std | 182.618491 | 35530.987199 | 1.042529 | 2.162610e+0 |
| min | 1.000000 | 1.000000 | 0.500000 | 8.281246e+08 |
| 25% | 177.000000 | 1199.000000 | 3.000000 | 1.019124e+09 |
| 50% | 325.000000 | 2991.000000 | 3.500000 | 	1.186087e+09 |
| 75% | 477.000000 | 8122.000000 | 4.000000 | 1.435994e+09 |
| max | 610.000000 | 193609.000000 | 5.000000 | 1.537799e+09 |


Kemudian untuk melihat visualisasi dari banyaknya genre film atau menampilkan rata-rata genre yg paling banyak muncul pada dataset.

Gambar 3 visualisasi genre

![visualisasi data](https://user-images.githubusercontent.com/111235408/194400480-e955951c-50e6-473a-b31f-9aaa5f230296.png)

Dalam tampilan Gambar 3 visualisasi genre terlihat bahwa film dengan genre commedi dan drama paling banyak diminati oleh pengguna.


## Data Preprocessing

- Menggabungkan Movie

Menggabungkan beberapa file menggunakan fungsi concatenate berdasarkan pada movieId, dengan menggabungkan seluruh data pada variabel movie_all. Beberapa variabel yang digabungkan adalah sebagai berikut :

   - linkk
   - moviie
   - ratingg
   - tagg
   
Setelah itu mengurutkan data dan menghapus data yang sama menggunakan np. sort , sehingga dihasilkan jumlah seluruh data movie berdasarkan movieId terdapat 9742.

- Menggabungkan Seluruh User

Menggabungkan beberapa file menggunakan fungsi concatenate berdasarkan pada userId, menggabungkan seluruh data pada variabel user_all. Beberapa variabel yang digabungkan adalah sebagai berikut :

   - rating
   - tagg

Setelah itu mengurutkan data dan menghapus data yang sama menggunakan np. sort , sehingga dihasilkan jumlah seluruh user 610.

Menggabungkan file linkk, moviie, ratingg, dan tagg ke dalam dataframe moviie_info. Serta menggabungkan ratingg dengan dataframe moviie_info berdasarkan nilai movieId. Menggunakan fungsi concat.

Setelah itu, mengecek missing value menggunakan fungsi isnull terhadap variabel moviie. Berikut ini Tabel. 6 Missing Value memudahkan dalam memahami variabel mana saja yang terdapat missing value :

Tabel. 6 Missing Value

|  |  | 
| ------------ |---------------|
| userId_x | 0 | 
| movieId | 0 | 
| rating_x | 0 | 
| timestamp_x | 0 | 
| imdbId | 6258749 | 
| tmdbId | 6258749 | 
| title | 6258749 | 
| genres | 6258749 | 
| userId_y | 201672 | 
| rating_y | 434885 | 
| timestamp_y | 201672 | 
| tag | 6126372 | 

Terlihat bahwa terdapat banyak missing value. Maka langkah selanjutnya adalah membersihkan missing value.

- Menggabungkan rating dengan berdasar movieId

Terdapat 9724 rows × 8 columns dalam penggabungan rating dengan berdasar movieId.

- Menggabungkan Data dengan Fitur Nama Movie

Mendefinisikan variabel all_moviie_rate dengan variabel ratingg.
  
Terdapat 100836 rows × 4 columns dalam penggabungan Data dengan Fitur Nama Movie.


## Data Preparation

Setelah proses penggabungan maka akan saya cek lagi datanya apakah ada missing value atau tidak.  Dengan menjalankan kode berikut.
all_moviie.isnull().sum(). Dan ternyata hasilnya dalam tabel berikut ini :

Tabel. 7 cek missing value

| variabel | Banyaknya data |
| ------------ |---------------| 
| userId  | 0 |
| movieId | 0 |
| rating | 0 |
| timestamp | 0 |
| title | 0 |
| genres | 0 |
| tag | 0 |

Perhatikanlah, sudah tidak terdapat missing value lagi setealah dilakukan penggabungan-penggabungan terhadap variabel. Selanjutnya, saya hanya akan menggunakan data unik untuk dimasukkan ke dalam proses pemodelan. Oleh karena itu, saya perlu menghapus data yang duplikat dengan fungsi drop_duplicates().

- Proses dalam tahap data preparation adalah dengan menghapus Missing Value yang terdapat pada variabel.
- Alasannya agar model yang dibuat memiliki tingkat prediksi yang bagus. 

Mengembangkan sistem rekomendasi dengan pendekatan content based filtering. Tapi sebelumnya, mari cek lagi data yang kita miliki dan assign dataframe dari tahap sebelumnya ke dalam variabel data, sebagai berikut:

Tabel. 8 assign dataframe 

|  | movieId | title | genres |
| ------------ |---------------|---------------|---------------|  
| 8501 | 113705 | Two Days, One Night (Deux jours, une nuit) (2014) | Drama |
| 9424 | 165969 | HyperNormalisation (2016) | Documentary |
| 7291 | 75803 | Our Family Wedding (2010) | Comedy |
| 2309 | 3061 | Holiday Inn (1942) | Comedy|Musical |
| 9019 | 140525 | Secret in Their Eyes (2015) | Crime|Drama|Mystery |

Dan pada dasarnya, method dropna() bisa digunakan untuk menghapus baris atau kolom yang mengandung missing values. Saya hanya perlu menentukan axis-nya, dimana 0 untuk menghapus baris dan 1 untuk menghapus kolom.

## Modeling

Tahapan yang dilakukan pada fungsi tersebut ialah sebagai berikut.

- Mengambil indeks dari judul film yang telah didefinisikan sebelumnnya.
- Mengambil skor kemiripan dengan semua film.
- Mengurutkan film berdasarkan skor kemiripan.
- Mengambil 19 judul berdasarkan kemiripan dari 1-20 karena urutan 0 memberikan indeks yang sama dengan judul film yang diinput.
- Mengambil judul film dari skor kemiripan.
- Mengembalikan 19 rekomendasi judul film dari kemiripan skor yang telah diurutkan dan menampilkan genre dari 19 rekomendasi film tersebut.

Berikut top-4 recommemdation berdasarkan genre dari judul film "Wonderful, Horrible Life of Leni Riefenstahl, The (Macht der Bilder: Leni Riefenstahl, Die) (1993)"

Tabel 1.3 top-4 recommemdation film

| Judul | genres |
| ------------ |---------------| 
| Wonderful, Horrible Life of Leni Riefenstahl, The (Macht der Bilder: Leni Riefenstahl, Die) (1993) | Documentary |
|  ![rekomendasi](https://user-images.githubusercontent.com/111235408/194407721-36b406b9-d912-4add-963d-a13b6d152050.png) |   |
| Dengan hasil yang diberikan di atas berdasarkan judul film "Wonderful, Horrible Life of Leni Riefenstahl, The (Macht der Bilder: Leni Riefenstahl, Die) (1993)" dengan genre Documentary maka didapatkan 4 rekomendasi judul film dengan genre yang serupa ataupun mirip. |  |

Setelah dilakukan pra-pemrosesan pada dataset, langkah selanjutnya adalah modeling terhadap data. Pada tahap ini Model machine learning yang digunakan pada sistem rekomendasi ini adalah model content-based filtering dengan simlarty measure yang digunakan adalah Cosine Similarity.

Model content-based filtering ini bekerja dengan mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna. Metode ini bekerja dengan menyarankan item serupa yang pernah disukai sebelumnya atau sedang dilihat sekarang kepada pengguna berdasrakan kategori tertentu dari item yang dinilai oleh pengguna dengan menggunakan similarity tertentu.

Sedangkan cosine similarity adalah salah satu teknik mengukur kesamaan yang bekerja dengan mengukur kesamaan antara dua vektor dan menentukan apakah kedua vektor tersebut menunjuk ke arah yang sama dengan menghitung sudut cosinus antara dua vektor. Semakin kecil sudut cosinus, semakin besar nilai cosine similarity. cara kerja dari fungsi cosine similiraty yaitu dengan melakukan perhitungan yang sering digunakan untuk menghitung kemiripan diantara item-item. 

Secara umum, fungsi similarity adalah fungsi yang menerima dua buah obyek berupa bilangan riil (0 dan 1) dan mengembalikan nilai kemiripan (similarity) antara kedua obyek tersebut berupa bilangan riil. Cosine similarity merupakan salah satu metode pengukuran kemiripan yang populer. Metode ini digunakan untuk menghitung nilai kosinus sudut antara dua vektor dan biasanya digunakan untuk mengukur kemiripan antara dua teks/dokumen. Fungsi cosine similarity antara item A dan item B ditunjukkan.

Selain itu saya juga menggunakan metode Collaborative Filtering, dimana Collaborative Filtering merupakan algoritma yang bergantung pada pendapat komunitas pengguna. Dia tidak memerlukan atribut untuk setiap itemnya. Metode Collaborative Filtering merupakan salah satu metode pada sistem rekomendasi. Metode ini memanfaatkan penilaian pengguna lain berupa rating atau umpan balik lain untuk memprediksi item yang mungkin diminati. 



- Ketika seseorang memasuki rental VCD seringkali, ia mengalami kebimbangan disebabkan oleh begitu banyaknya pilihan film yang tersedia. Mereka yang sebelumnya tidak memiliki cukup informasi seperti dari membaca review-review film dan mereka yang memang belum memiliki tujuan pasti akan menyewa judul film apa, membutuhkan bentuk rekomendasi dari member-member lainnya. Rekomendasi yang diinginkan adalah yang bersifat personal dan yang dapat sedikit di luar dugaan, kemungkinan film yang sama sekali tidak terpikirkan namun ternyata menarik dan sesuai seleranya. Collaborative filtering memungkinkan munculnya item yang memiliki karakteristik sama sekali berbeda dari item-item yang pernah dipilih sebelumnya namun ternyata menarik bagi user bersangkutan, karena rekomendasi didasarkan pada preferensi user-user lain. Feedback yang ditangkap secara implisit berupa data biner dengan hanya didasarkan pada perilaku seorang member apakah dia menyewa (‘1’) ataukah belum menyewa (‘0’) judul film tertentu. Metode collaborative filtering yang digunakan adalah user-based collaborative filtering, item-based collaborative filtering, dan item-based collaborative filtering yang dikombinasikan dengan fitur konten. Hasil dari pengujian ketiga metode menunjukkan bahwa pada penggunaan user-based collaborative filtering terjadi kesalahan prediksi rata-rata sebanyak 58,8%; pada item-based collaborative filtering terjadi kesalahan prediksi rata-rata sebanyak 24,9%; sedangkan pada item-based collaborative filtering yang dikombinasikan dengan fitur konten terjadi kesalahan prediksi rata-rata sebanyak 24,4%. Pengkombinasian collaborative filtering dengan fitur konten mengakibatkan hasil rekomendasi yang muncul tidak lagi memiliki karakteristik rekomendasi hasil collaborative filtering.

Referensi : [Penerapan Metode Collaborative Filtering Menggunakan Rating Implisit pada Sistem Perekomendasi Pemilihan Film Di Rental VCD](https://digilib.uns.ac.id/dokumen/detail/26091)


- Kedua model menurut saya sudah bekerja dengan bagus, dan menurut saya model terbaik yaitu Collaborative filtering. Karena merekomendasikan film berdasarkan genre yang sama dan sesuai.


## Evaluation

Pada proyek ini, Metric yang digunakan pada sistem rekomendasi judul film berdasarkan genre adalah accuracy precision. Precision adalah metrik yang membandingkan rasio prediksi benar atau positif dengan keseluruhan hasil yang diprediksi positif dengan rumus :

Tabel 1.4 Rumus Metric accuracy precision

| Rumus | 
| ------------ | 
| Precission = TP/(TP + FP)  |
| keterangan:
TP = True Positif (prediksi positif dan hal tersebut benar)
FP = False Positif (prediksi positif dan hal tersebut salah) |

Alasan accuracy Precision dipilih adalah karena metrik ini dapat membandingkan rasio prediksi benar atau positif dengan keseluruhan hasil yang diprediksi positif. Dalam hal ini adalah rasio item yang direkomendasikan memiliki genre yang mirip atau serupa dibandingkan dengan genre dari judul film yang diinput.

Berikut adalah hasil evaluasi :

Gambar 1.7 Hasil evaluasi

![final](https://user-images.githubusercontent.com/111235408/194409935-95dac870-a85a-45f8-b376-a8645832b1f2.png)

# Referensi
- [Sistem Rekomendasi Film Menggunakan Metode User Based Collaborative Filtering](https://openlibrarypublications.telkomuniversity.ac.id/index.php/engineering/article/view/16513) 
- [Kelas Machine Learning Terapan](https://www.dicoding.com/academies/319/tutorials/17116)
- [Penerapan Metode Collaborative Filtering Menggunakan Rating Implisit pada Sistem Perekomendasi Pemilihan Film Di Rental VCD](https://digilib.uns.ac.id/dokumen/detail/26091)
