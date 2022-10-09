# Laporan Proyek Machine Learning - Firda Sa'baniyah

## Domain Proyek

Domain proyek yang dipilih dalam proyek akhir _machine learning_ terapan ini adalah membuat _movie recommendation system_. 

![17___film](https://user-images.githubusercontent.com/111235408/194676810-f8675b7c-6f78-414c-a107-ebb81be328a0.jpg)

Gambar 1. Movie

_Recommendation system_ telah menjadi lazim dalam beberapa tahun terakhir karena menangani masalah kelebihan informasi dengan menyarankan pengguna produk yang paling relevan dari sejumlah besar data. Untuk produk media, _movie recommendation_ kolaboratif _online_ berupaya membantu pengguna mengakses film pilihan mereka dengan menangkap tetangga yang persis sama di antara pengguna atau film dari peringkat umum historis mereka. Namun, karena data yang jarang, pemilihan tetangga menjadi lebih sulit dengan meningkatnya film dan pengguna dengan cepat. 
 _Movie recommendation system_ merupakan sistem yang merekomendasikan film kepada penonton atau pengguna lainnya, rekomendasi ini contohnya diterapkan pada situs seperti netflix, iqiyi, dan wetv. Dalam proyek ini, diusulkan sistem rekomendasi film berbasis model _collaborative filtering_ dan _based filtering_.  Sistem rekomendasi yang saya buat ini didasarkan pada peferensi kesukaan pengguna dimasa lalu, serta rating dari film tersebut.
Hasil eksperimen pada _dataset recommendation movie_ menunjukkan bahwa pendekatan yang diusulkan dapat memberikan kinerja tinggi dalam hal akurasi, dan menghasilkan rekomendasi film yang lebih andal dan personal jika dibandingkan dengan metode yang ada.

- Film adalah salah satu media hiburan yang populer di masyarakat. Banyaknya judul-judul yang telah rilis membuat masyarakat kesulitan untuk menemukan film mana yang mereka ingin tonton. Karena hal tersebut masalah ini perlu untuk diatasi. Sehingga informasi mengenai film akan memudahkan masyarakat untuk menemukan film yang cocok dengan preferensi _user_, oleh sebab itu _user_ perlu sebuah sistem yang dapat memberikan rekomendasi film.  
- Sistem rekomendasi adalah sistem yang mampu memberikan rekomendasi _item_ yang mungkin disukai oleh pengguna. Metode _Collaborative Filtering_ merupakan salah satu metode pada sistem rekomendasi. Metode ini memanfaatkan penilaian pengguna lain berupa rating atau umpan balik lain untuk memprediksi item yang mungkin diminati. _Collaborative filtering (CF)_ yang digunakan adalah metode _User Based Collaborative Filtering_. Penerapan dilakukan pada _website_ merekomendasikan film-film untuk _user_ menonton.
- _Collaborative filtering_ merupakan sebuah metode dalam membuat prediksi dengan cara menyaring informasi item dari opini pengguna lain. Ide utama dalam sistem rekomendasi _collaborative filtering_ adalah untuk memanfaatkan riwayat opini pengguna aktif lain untuk memprediksi item yang mungkin akan disukai/diminati
oleh seorang pengguna. Implementasi yang paling sederhana dari pendekatan ini adalah membuat rekomendasi kepada pengguna aktif berdasarkan item yang disukai pengguna lain dengan riwayat selera yang serupa. [[1]](https://openlibrarypublications.telkomuniversity.ac.id/index.php/engineering/article/view/16513)


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
    
Solusi yang saya buat yaitu dengan menggunakan 2 algoritma _Machine Learning_ sistem rekomendasi,yaitu :
-  _Content Based Filtering_ adalah algoritma yang merekomendasikan item serupa dengan apa yang disukai pengguna, berdasarkan tindakan mereka sebelumnya atau umpan balik eksplisit.
 _Content Based Filtering_ mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna. Algoritma ini bekerja dengan menyarankan item serupa yang pernah disukai di masa lalu atau sedang dilihat di masa kini kepada pengguna. Semakin banyak informasi yang diberikan pengguna, semakin baik akurasi sistem rekomendasi.

- _Collaborative Filtering_, adalah algoritma yang bergantung pada pendapat komunitas pengguna. Dia tidak memerlukan atribut untuk setiap itemnya.
Metode _Collaborative Filtering_ merupakan salah satu metode pada sistem rekomendasi. Metode ini memanfaatkan penilaian pengguna lain berupa rating atau umpan balik lain untuk memprediksi item yang mungkin diminati. 

Bahasa sederhananya, algoritma _Content Based Filtering_ digunakan untuk merekomendasikan film berdasarkan aktivitas pengguna pada masa lalu, sedangkan algoritma _Collaborative Filtering_ digunakan untuk merekomendasikan film berdasarkan rating yang paling tinggi.

Saya menggunakan _dataset_ dari kaggle, untuk dapat menggunakan dataset tersebut saya mengimpor file kredensial kaggle dalam bentuk json terlebih dahulu. Maksud dari file kredensial ini adalah untuk mengatur _permission_ _dataset_ yang nantinya akan saya _upload_. Saya juga mengimport _library pandas_ yang dibutuhkan untuk _upload_ dataset _Movie Recommendation Data_ menggunakan kode API. Untuk dapat melihat dataset _Movie Recommendation Data_ yang terdapat pada gambar 2 dataset _Movie Recommendation Data_.

![Reommendation](https://user-images.githubusercontent.com/111235408/194678835-edc34e43-fd9f-45ad-95c2-ed1cdb8baeaa.png)

Gambar 2. _Dataset Movie Recommendation Data_

## Data Understanding

Setelah _mendownload dataset_ tahap selanjutnya adalah melakukan data _understanding_. Agar lebih memudahkan pembaca, saya membuat tabel yang dapat dilihat di Tabel 1. data _understanding_.

Tabel 1. Data _Understanding_

| Sumber | [Kaggle Dataset : Movie Recommendation Data](https://www.kaggle.com/datasets/rohan4050/movie-recommendation-data)|
| ------------ |---------------| 
| Lisensi | Unknown |
| Kategori | Industri Hiburan |
| Jenis dan Ukuran Berkas | ZIP 14 MB |

_Dataset_ yang digunakan pada proyek akhir _machine learning_ ini adalah _dataset_ _Movie Recommendation Data_ yang didapat dari situs kaggle. Link _dataset_ tersebut, berikut tautan _Dataset_ : [Movie Recommendation Data](https://www.kaggle.com/datasets/rohan4050/movie-recommendation-data)


### Variabel-variabel pada Movie Recommendation dataset adalah sebagai berikut:

- Linkss : list link film.
- Moviie : list film yang tersedia.
- Ratingg :list penilaian yang di berikan penonton terhadap film.
- Tagg : list kata kunci dari film.

Berikut beberapa tahapan Data _Understanding_ diantaranya sebagai berikut:

- Meload _dataset_ ke dalam sebuah _dataframe_ menggunakan _pandas_.
- df.info() digunakan untuk mengecek tipe kolom pada _dataset_.
- df.isna().sum() digunakan untuk mengecek apakah ada kolom yg kosong, pada _dataset_ ini nilai kosong tidak ditemukan.
- df.describe() digunakan untuk mendapatkan info mengenai _dataset_ terhadap nilai rata-rata, median, banyaknya data, nilai Q1 hingga Q3 dan lain-lain.
- len (nama_variable.unique()) menghitung panjang data _unique_ dari variable tertentu.

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

Dari tabel 2. Variable Moviie yang tertera diatas terdapat 3 kolom yang ada di variabel moviie yaitu movieId, title, dan genres. MovieId memiliki tipe data int64 sedangkan _title_ dan _genres_ memiliki tipe data _object_.

- variabel Linkss

Tabel 3. variabel Linkss

| Column | Non-Null | Count | Dtype |
| ------------ |---------------|---------------|---------------|
| movieId | 9742 | non-null | int64 |
| imdbId |  9742 | non-null | int64 |
| tmdbId | 9742 | non-null | float64 |

Dari tabel 3. Variable Linkss yang tertera diatas terdapat 3 kolom yang ada di variabel Linkss yaitu movieId, imdbId, dan tmdbId. MovieId dan imdbId memiliki tipe data int64 sedangkan tmdbId memiliki tipe data float64.

- Variabel Ratingg

Tabel 4. Variabel Ratingg

| Column | Non-Null | Count | Dtype |
| ------------ |---------------|---------------|---------------|
| userId | 100836 | non-null | int64 |
| movieId | 100836 | non-null | int64 |
| rating | 100836 | non-null | float64 |
| timestamp | 100836 | non-null | int64 |

Dari tabel 4. Variable Ratingg yang tertera diatas terdapat 4 kolom yang ada di variabel Ratingg yaitu userId, movieId, rating, dan timestamp. UserId, movieId, dan timestamp memiliki tipe data int64 sedangkan rating memiliki tipe data float64.

Untuk melihat rating _describe_ dan memudahkan dalam melihatnya saya visualisasikan menggunakan Table 5. Visualisasi Rating _Describe_ dibawah.

Table 5. Visualisasi Rating _Describe_

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


Kemudian untuk melihat visualisasi dari banyaknya _genre film_ atau menampilkan rata-rata _genre_ yg paling banyak muncul pada _dataset_.

![visualisasi data](https://user-images.githubusercontent.com/111235408/194400480-e955951c-50e6-473a-b31f-9aaa5f230296.png)

Gambar 3. Visualisasi _Genre_

Dalam tampilan Gambar 3 visualisasi _genre_ terlihat bahwa film dengan _genre commedi_ dan drama paling banyak diminati oleh pengguna.


## Data Preprocessing

- Menggabungkan Movie

Menggabungkan beberapa file menggunakan fungsi _concatenate_ berdasarkan pada movieId, dengan menggabungkan seluruh data pada variabel movie_all. Beberapa variabel yang digabungkan adalah sebagai berikut :

   - linkk
   - moviie
   - ratingg
   - tagg
   
Setelah itu mengurutkan data dan menghapus data yang sama menggunakan np. sort , sehingga dihasilkan jumlah seluruh data film berdasarkan movieId terdapat 9742.

- Menggabungkan Seluruh User

Menggabungkan beberapa file menggunakan fungsi _concatenate_ berdasarkan pada userId, menggabungkan seluruh data pada variabel user_all. Beberapa variabel yang digabungkan adalah sebagai berikut :

   - rating
   - tagg

Setelah itu mengurutkan data dan menghapus data yang sama menggunakan np. sort , sehingga dihasilkan jumlah seluruh user 610.

Menggabungkan file linkk, moviie, ratingg, dan tagg ke dalam dataframe moviie_info. Serta menggabungkan ratingg dengan dataframe moviie_info berdasarkan nilai movieId. Menggunakan fungsi concat.

Setelah itu, mengecek missing value menggunakan fungsi _isnull_ terhadap variabel moviie. Berikut ini Tabel. 6 _Missing Value_ memudahkan dalam memahami variabel mana saja yang terdapat _missing value_ :

Tabel 6. _Missing Value_

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

Terlihat bahwa terdapat banyak _missing value_. Maka langkah selanjutnya adalah membersihkan _missing value_.

- Menggabungkan rating dengan berdasar movieId

Terdapat 9724 rows × 8 columns dalam penggabungan rating dengan berdasar movieId.

- Menggabungkan Data dengan Fitur Nama Moviie

Mendefinisikan variabel all_moviie_rate dengan variabel ratingg.
  
Terdapat 100836 rows × 4 columns dalam penggabungan Data dengan Fitur Nama Moviie.


## Data Preparation

Setelah proses penggabungan maka akan saya cek lagi datanya apakah ada _missing value_ atau tidak.  Dengan menjalankan kode berikut. _All_moviie.isnull().sum()_. Dan ternyata hasilnya dalam tabel berikut ini :

Tabel 7. cek _missing value_

| variabel | Banyaknya data |
| ------------ |---------------| 
| userId  | 0 |
| movieId | 0 |
| rating | 0 |
| timestamp | 0 |
| title | 0 |
| genres | 0 |
| tag | 0 |

Perhatikanlah, sudah tidak terdapat _missing value_ lagi setelah dilakukan penggabungan-penggabungan terhadap variabel. Selanjutnya, saya hanya akan menggunakan data unik untuk dimasukkan ke dalam proses pemodelan. Oleh karena itu, saya perlu menghapus data yang duplikat dengan fungsi drop_duplicates(). Dan pada dasarnya, _method dropna()_ bisa digunakan untuk menghapus baris atau kolom yang mengandung _missing values_. Saya hanya perlu menentukan _axis-nya_, dimana 0 untuk menghapus baris dan 1 untuk menghapus kolom.

- Proses dalam tahap data _preparation_ adalah dengan menghapus _Missing Value_ yang terdapat pada variabel.
- Alasannya agar model yang dibuat memiliki tingkat prediksi yang bagus. 

Mengembangkan sistem rekomendasi dengan pendekatan _content based filtering_. Tapi sebelumnya, mari cek lagi data yang kita miliki dan _assign dataframe_ dari tahap sebelumnya ke dalam variabel data, sebagai berikut:

Tabel 8. _Assign Dataframe_

|  | movieId | title | genres |
| ------------ |---------------|---------------|---------------|  
| 8501 | 113705 | Two Days, One Night (Deux jours, une nuit) (2014) | Drama |
| 9424 | 165969 | HyperNormalisation (2016) | Documentary |
| 7291 | 75803 | Our Family Wedding (2010) | Comedy |
| 2309 | 3061 | Holiday Inn (1942) | Comedy|Musical |
| 9019 | 140525 | Secret in Their Eyes (2015) | Crime|Drama|Mystery |

## Training dan Validasi metode _Collaborative Filtering_

Tahap persiapan ini penting dilakukan agar data siap digunakan untuk pemodelan. Namun sebelumnya, saya perlu membagi data untuk training dan validasi terlebih dahulu.
Bagi data train dan validasi dengan komposisi 80:20. Namun sebelumnya, saya perlu memetakan (mapping) data genre dan movie menjadi satu value terlebih dahulu. Lalu, membuat rating dalam skala 0 sampai 1 agar mudah dalam melakukan proses training. 

![pembagian data](https://user-images.githubusercontent.com/111235408/194687162-9caaf75f-9e78-49d0-a6ff-4396f4cc57fb.png)

Gambar 4. Output Pembagian Data

## Modeling

### Model Development dengan Content Based Filtering

### TF-IDF Vectorizer

Pada tahap ini, saya akan membangun sistem rekomendasi sederhana berdasarkan _genre film_. Teknik ini digunakan pada sistem rekomendasi untuk menemukan representasi fitur penting dari setiap kategori _genre_. Menggunakan fungsi _tfidfvectorizer()_ dari _library sklearn_. Selain itu saya juga melakukan Inisialisasi _TfidfVectorizer_ dalam hal ini terdapat (1554, 24) shape yang ditemukan berarti nilai 1554 merupakan ukuran data dan 24 merupakan matrik kategori _genre_. 

Perhitungan idf pada data _genre_ dengan cara melakukan fit lalu ditransformasikan ke bentuk _matrix_ dan melihat ukuran _matrix_ tfidf dengan menggunakan fungsi _todense()_. Selanjutnya, saya lihat _matriks tf-idf_ untuk beberapa _genre_. Dapat dilihat pada Gambar. 3 _matrix tfidf_ dibawah.

![matrix](https://user-images.githubusercontent.com/111235408/194685122-60f0aa0d-06ea-4b55-9e86-59dfe0371168.png)

Gambar 5. _Matrix Tfidf_

Selanjutnya, saya akan menghitung derajat kesamaan (_similarity degree_) antar _genre_ dengan teknik _cosine similarity_. Di sini, saya menggunakan fungsi _cosine_similarity_ dari _library_ sklearn. Berikut Gambar. 4 _Cosine Simmilarity_.

![cosine](https://user-images.githubusercontent.com/111235408/194685484-bc41f47b-f601-4435-92e6-94f322ba9e7b.png)

Gambar 6. _Cosine Simmilarity_

Pada tahapan ini, saya menghitung _cosine similarity dataframe _tfidf matrix_ yang saya peroleh pada tahapan sebelumnya. Dengan satu baris kode untuk memanggil fungsi _cosine similarity_ dari _library sklearn_, saya telah berhasil menghitung kesamaan (_similarity_) antar _genre_. Kode di atas menghasilkan keluaran berupa matriks kesamaan dalam bentuk _array_. 

Dengan _cosine similarity_, saya berhasil mengidentifikasi kesamaan antara satu _genre film_ dengan _genre film_ lainnya. _Shape_ (1554, 1554) merupakan ukuran matriks _similarity_ dari data yang saya miliki. Berdasarkan data yang ada, matriks di atas sebenarnya berukuran 1554 _genre_ x 1554 _genre_ (masing-masing dalam sumbu X dan Y). Artinya, saya mengidentifikasi tingkat kesamaan pada 1554 _genre film_.  

Nah, dengan data kesamaan (_similarity_) _genre_ yang diperoleh dari kode sebelumnya, saya akan merekomendasikan daftar judul film yang mirip (_similar_) dengan _genre_ yang sebelumnya pernah melayani pengguna. 


### Model Development dengan Collaborative Filtering

Sebelumnya saya telah menerapkan teknik _content based filtering_ pada data. Teknik ini merekomendasikan _item_ yang mirip dengan preferensi pengguna di masa lalu. Selanjutnya saya akan menerapkan teknik _collaborative filtering_ untuk membuat sistem rekomendasi. Teknik ini membutuhkan data rating dari _user_. 

_Goal_ proyek kali ini adalah menghasilkan rekomendasi sejumlah judul film yang sesuai dengan preferensi pengguna berdasarkan rating yang telah diberikan sebelumnya. Dari data rating pengguna, saya akan mengidentifikasi restoran-restoran yang mirip dan belum pernah dikunjungi oleh pengguna untuk direkomendasikan.

### Proses Training 

Pada tahap ini, model menghitung skor kecocokan antara judul dan _genre_ dengan teknik _embedding_. Pertama, saya melakukan proses _embedding_ terhadap data _genre_ dan moviie. Selanjutnya, lakukan operasi perkalian _dot product_ antara _embedding genre_ dan _movie_. Selain itu, saya juga dapat menambahkan bias untuk setiap judul dan _genre_. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi _sigmoid_. 

Di sini, saya membuat _class RecommenderNet_ dengan keras model class. Kode _class RecommenderNet_ ini terinspirasi dari tutorial dalam situs Keras dengan beberapa adaptasi sesuai kasus yang sedang saya selesaikan. 

Selanjutnya, lakukan proses _compile_ terhadap model. Model ini menggunakan _Binary Crossentropy_ untuk menghitung _loss function_, _Adam (Adaptive Moment Estimation)_ sebagai _optimizer_, dan _root mean squared error (RMSE)_ sebagai _metrics evaluation_. Langkah berikutnya, mulailah proses _training_. 

### _Modelling_ dan _Resault_

Berikut hasil Top-N tertera pada Tabel 10. Hasil Top-N

Tabel 9. Hasil Top-N

| movie_name | Gentleman's Agreement (1947) | Lady Vengeance (Sympathy for Lady Vengeance) (Chinjeolhan geumjassi) (2005) | Deadpool 2 (2018) | Better Off Dead... (1985) | Vertigo (1958) |
| ------------ |---------------|---------------|---------------|---------------|---------------| 
| Monster (2003) | 0.462197	 | 0.615050 | 0.000000 | 0.000000 | 0.137931 |
| Coming Home (1978)	 | 0.366404 | 0.104159 | 0.000000 | 0.000000 | 0.109344 |
| Thor: Ragnarok (2017) | 0.000000 | 0.000000 | 0.819244 | 0.000000 | 0.000000 |
| Follow the Fleet (1936) | 0.000000 | 0.000000 | 0.148325 | 0.642388 | 0.234118 |
| Scott Pilgrim vs. the World (2010) | 0.000000 | 0.000000 | 0.318270 | 0.477664 | 0.174085 |
| True Lies (1994) | 0.000000 | 0.217058 | 0.368857 | 0.553586 | 0.429617 |
| Grosse Pointe Blank (1997) | 0.000000 | 0.370178 | 0.169571 | 0.734405 | 0.267654 |
| Maltese Falcon, The (a.k.a. Dangerous Female) (1931) | 0.000000 | 0.624750 | 0.000000 | 0.000000 | 0.655849 |
| Dumbo (1941) | 0.229088 | 0.000000 | 0.065124 | 0.000000 | 0.068366 |
| Jungle2Jungle (a.k.a. Jungle 2 Jungle) (1997) | 0.000000 | 0.000000 | 0.173914 | 0.310132 | 0.000000 |

## Evaluation

### Mendapatkan Rekomendasi film

- Content Based Filtering

Untuk mendapatkan rekomendasi judul film, pertama saya ambil sampel _user_ secara acak dan definisikan variabel moviie_not_visited yang merupakan daftar judul film yang belum pernah dikunjungi oleh pengguna. Hal ini karena daftar moviie_not_visited inilah yang akan menjadi film yang akan rekomendasikan. 

Sebelumnya, pengguna telah memberi rating pada beberapa film yang telah mereka kunjungi. saya menggunakan rating ini untuk membuat rekomendasi film yang mungkin cocok untuk pengguna. Nah, film yang akan direkomendasikan tentulah film yang belum pernah ditonton oleh pengguna. 

Variabel moviie_not_visited diperoleh dengan menggunakan operator bitwise (~) pada variabel _moviie_visited_by_user_.

Selanjutnya, untuk memperoleh rekomendasi film, gunakan fungsi _model.predict()_ dari _library Keras_.

Tabel 10. Hasil Rekomendasi

| Showing recommendations for users : 448 (Movie with high ratings from user) |  |
| ------------ |---------------| 
|  Toy Story (1995) | (Adventure, Animation, Animation, Children, Comedy, Fantasy) |
| Rating Bull (1980) | (Drama) | 
| Annie Hal (1977) | (Comedy, Romance) | 
| Back to The Future (1985) | (Adventure, Comed, Sci-Fi) | 

Tabel 11. Top 10 Movie Recommendation

| Top 10 Movie Recommendation |  | 
| ------------ |---------------| 
| Secrets & Lies (1996) | Drama |
| Shallwe Dance (1973) | (Comedy, Musical, Romance) |
| Streetcar Named Desire, A (1951) | Drama |
| Buena Vista Social Club (1999) | (Documentary, Musical) |
| Guess Who's Coming to Dinner (1967) | Drama |
| Witness for the Prosecution (1957) | (Drama, Mystery, Thriller) |
| Adam's Rib (1949) | (Comedy, Romance) |
| Strada, La (1954) | Drama |
| Wild Parrots of Telegraph Hill, The (2003) | Documentary |
| Captain Fantastic (2016) | Drama |

- Collaborative Filtering

Membuat fungsi moviie_recommendations dengan beberapa parameter sebagai berikut:

Nama_moviie : Nama judul dari moviie tersebut (index kemiripan _dataframe_).
_Similarity_data_ : _Dataframe_ mengenai similarity yang telah kita didefinisikan sebelumnya
_Items_ : Nama dan fitur yang digunakan untuk mendefinisikan kemiripan, dalam hal ini adalah _movie_name_ dan _genre_.
k : Banyak rekomendasi yang ingin diberikan.

Hasil rekomendasi terdapat pada tabel 9. Hasil Rekomendasi

Tabel 12. Hasil Rekomendasi 

|  | movie_name | genres |
| ------------ |---------------|---------------| 
| 0 | Eyes of Tammy Faye, The (2000) | Documentary |
| 1 | Why We Fight (2005) | Documentary |
| 2 | Anne Frank Remembered (1995) | Documentary |
| 3 | Deliver Us from Evil (2006) | Documentary |
| 5 | Stone Reader (2002) | Documentary |


### Visualisasi Metrik

Untuk melihat visualisasi proses _training_, saya plot metrik evaluasi dengan _matplotlib_. 

![model metrik](https://user-images.githubusercontent.com/111235408/194691167-b59a2337-8ccf-4f31-b8f4-d45a87ec5d48.png)

Gambar 7. Visualisasi Metrik

Perhatikanlah pada Gambar. 10 visualisasi Metrik , proses training model cukup smooth dan model konvergen pada epochs sekitar 30. Dari proses ini, saya memperoleh nilai error akhir sebesar sekitar 0.1956 dan error pada data validasi sebesar 0.6132. Nilai tersebut cukup bagus untuk sistem rekomendasi. 

Sampai di tahap ini, saya telah berhasil membuat sistem rekomendasi dengan dua teknik, yaitu _Content based Filtering_ dan _Collaborative Filtering_. Sistem rekomendasi yang saya buat telah berhasil memberikan sejumlah rekomendasi film yang sesuai dengan preferensi pengguna. 

Setiap teknik membutuhkan data yang berbeda dan bekerja dengan cara yang berbeda pula. Misalnya, pada teknik _collaborative filtering_, saya membutuhkan data rating dari pengguna. Sedangkan, pada _content based filtering_, data rating tidak diperlukan. 

Berikut kelebihan serta kekurangan dari ke-2 model :

Model _content-based filtering_ ini bekerja dengan mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna. Metode ini bekerja dengan menyarankan item serupa yang pernah disukai sebelumnya atau sedang dilihat sekarang kepada pengguna berdasrakan kategori tertentu dari item yang dinilai oleh pengguna dengan menggunakan _similarity_ tertentu.

Sedangkan _cosine similarity_ adalah salah satu teknik mengukur kesamaan yang bekerja dengan mengukur kesamaan antara dua vektor dan menentukan apakah kedua vektor tersebut menunjuk ke arah yang sama dengan menghitung sudut _cosinus_ antara dua vektor. Semakin kecil sudut _cosinus_, semakin besar nilai _cosine similarity_. Cara kerja dari fungsi _cosine similiraty_ yaitu dengan melakukan perhitungan yang sering digunakan untuk menghitung kemiripan diantara item-item. 

Secara umum, fungsi _similarity_ adalah fungsi yang menerima dua buah obyek berupa bilangan riil (0 dan 1) dan mengembalikan nilai kemiripan (similarity) antara kedua obyek tersebut berupa bilangan _riil_. _Cosine similarity_ merupakan salah satu metode pengukuran kemiripan yang populer. Metode ini digunakan untuk menghitung nilai kosinus sudut antara dua vektor dan biasanya digunakan untuk mengukur kemiripan antara dua teks/dokumen. Fungsi cosine similarity antara item A dan item B ditunjukkan.

Selain itu saya juga menggunakan metode _Collaborative Filtering_, dimana _Collaborative Filtering_ merupakan algoritma yang bergantung pada pendapat komunitas pengguna. Dia tidak memerlukan atribut untuk setiap itemnya. Metode _Collaborative Filtering_ merupakan salah satu metode pada sistem rekomendasi. Metode ini memanfaatkan penilaian pengguna lain berupa rating atau umpan balik lain untuk memprediksi _item_ yang mungkin diminati. 

- Ketika seseorang memasuki rental VCD seringkali, ia mengalami kebimbangan disebabkan oleh begitu banyaknya pilihan film yang tersedia. Mereka yang sebelumnya tidak memiliki cukup informasi seperti dari membaca review-review film dan mereka yang memang belum memiliki tujuan pasti akan menyewa judul film apa, membutuhkan bentuk rekomendasi dari member-member lainnya. Rekomendasi yang diinginkan adalah yang bersifat personal dan yang dapat sedikit di luar dugaan, kemungkinan film yang sama sekali tidak terpikirkan namun ternyata menarik dan sesuai seleranya. _Collaborative filtering_ memungkinkan munculnya _item_ yang memiliki karakteristik sama sekali berbeda dari item-item yang pernah dipilih sebelumnya namun ternyata menarik bagi user bersangkutan, karena rekomendasi didasarkan pada preferensi user-user lain. _Feedback_ yang ditangkap secara implisit berupa data biner dengan hanya didasarkan pada perilaku seorang _member_ apakah dia menyewa (‘1’) ataukah belum menyewa (‘0’) judul film tertentu. Metode collaborative filtering yang digunakan adalah user-based collaborative filtering, item-based collaborative filtering, dan item-based _collaborative filtering_ yang dikombinasikan dengan fitur konten. Hasil dari pengujian ketiga metode menunjukkan bahwa pada penggunaan _user-based _collaborative filtering_ terjadi kesalahan prediksi rata-rata sebanyak 58,8%; pada _item-based collaborative filtering_ terjadi kesalahan prediksi rata-rata sebanyak 24,9%; sedangkan pada _item-based collaborative filtering_ yang dikombinasikan dengan fitur konten terjadi kesalahan prediksi rata-rata sebanyak 24,4%. Pengkombinasian _collaborative filtering_ dengan fitur konten mengakibatkan hasil rekomendasi yang muncul tidak lagi memiliki karakteristik rekomendasi hasil _collaborative filtering_. [[2]](https://digilib.uns.ac.id/dokumen/detail/26091)


# Referensi
- [[1]][Sistem Rekomendasi Film Menggunakan Metode User Based Collaborative Filtering](https://openlibrarypublications.telkomuniversity.ac.id/index.php/engineering/article/view/16513)Nugraha, D., Purboyo, T. W., & Nugrahaeni, R. A. (2021). Sistem Rekomendasi Film Menggunakan Metode User Based Collaborative Filtering. eProceedings of Engineering, 8(5).
- [[2]][Penerapan Metode Collaborative Filtering Menggunakan Rating Implisit pada Sistem Perekomendasi Pemilihan Film Di Rental VCD](https://digilib.uns.ac.id/dokumen/detail/26091)Dzumiroh, L. (2012). Penerapan Metode Collaborative Filtering Menggunakan Rating Implisit pada Sistem Perekomendasi Pemilihan Film di Rental VCD.
