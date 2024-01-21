import itertools  # Importing itertools for efficient iteration
import imageio  # Importing imageio for image input/output
import numpy as np  # Importing NumPy for numerical operations
from scipy.sparse import lil_matrix, csr_matrix  # Importing sparse matrices for memory efficiency
from scipy.spatial.distance import cityblock  # Importing Manhattan distance for heuristic
import matplotlib.pyplot as plt  # Importing matplotlib for visualization
import heapq  # Importing heapq for the priority queue implementation
import time  # Importing time for measuring elapsed time

# Function to load an image from a file path
def load_image(file_path): # - file_path (str): Resim dosyasının bulunduğu dosya yolunu içeren bir string.
    # Fonksiyon: load_image
    # Açıklama: Belirtilen dosya yolundan bir resim dosyası okur, imageio kullanarak.
    return imageio.v2.imread(file_path)
# Dönüş:
# - ndarray: NumPy dizisi olarak resim verisi.

# Function to extract the first channel if the image is RGB
def create_flat_image(img):
    if len(img.shape) == 3:
        return img[:, :, 0]
    else:
        return img
    # Fonksiyon: create_flat_image
    # Açıklama: Eğer resim üç kanallı (RGB) ise, sadece ilk kanalı (kırmızı) alır ve düzleştirilmiş bir görüntü oluşturur.
    # Parametreler:
    # - img (ndarray): Giriş olarak verilen resim verisi, NumPy dizisi olarak.
    # Dönüş:
    # - ndarray: İki boyutlu bir NumPy dizisi olarak düzleştirilmiş görüntü.


# Function to convert 2D coordinates to a 1D index in a flattened array
def to_index(y, x, width):
    return y * width + x
# Fonksiyon: to_index
# Açıklama: İki boyutlu koordinatları, belirtilen genişlikteki bir düzleştirilmiş dizideki 1D dizin olarak dönüştürür.
# Parametreler:
# - y (int): Yükseklik (satır) koordinatı.
# - x (int): Genişlik (sütun) koordinatı.
# - width (int): Düzleştirilmiş dizinin genişliği.
# Dönüş:
# - int: 1D dizindeki indeksi temsil eden bir tamsayı.



# Function to convert a 1D index to 2D coordinates
def to_coordinates(index, width):
    y, x = divmod(index, width)
    return y, x
# Fonksiyon: to_coordinates
# Açıklama: Düzleştirilmiş bir dizindeki 1D dizini, belirtilen genişlikteki iki boyutlu koordinatlara dönüştürür.
# Parametreler:
# - index (int): Düzleştirilmiş dizideki 1D dizin.
# - width (int): Düzleştirilmiş dizinin genişliği.
# Dönüş:
# - tuple: Yükseklik (satır) ve genişlik (sütun) koordinatlarını içeren bir tuple.


# Function to build an adjacency matrix representing the connectivity of pixels
def build_adjacency_matrix(img):
    img_size = img.size
    adjacency = lil_matrix((img_size, img_size), dtype=bool)
    # Fonksiyon: build_adjacency_matrix
    # Açıklama: Resimdeki pikseller arasındaki bağlantıyı temsil eden bir komşuluk matrisi oluşturur.
    # Parametreler:
    # - img (ndarray): Giriş olarak verilen resim verisi, NumPy dizisi olarak.
    # Dönüş:
    # - csr_matrix: Pikseller arasındaki bağlantıyı temsil eden seyrek bir matris (Compressed Sparse Row formatında).

    # Define 8-connected neighborhood directions
    directions = list(itertools.product([0, 1, -1], [0, 1, -1]))

    # Iterate over the image pixels
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if not img[i, j]:
                continue
                # Döngü: Piksel İterasyonu
                # Açıklama: Resimdeki pikselleri gezerek, siyah pikseller hariç olanları işler.
                # İterasyonlar:
                # - i: Yükseklik (satır) indeksi, 1'den resmin yüksekliği eksi bir değere kadar.
                # - j: Genişlik (sütun) indeksi, 1'den resmin genişliği eksi bir değere kadar.
                # Koşul:
                # - Eğer piksel siyah değilse devam et, aksi halde geç.

            # Connect the pixel to its neighbors in the adjacency matrix
            for y_diff, x_diff in directions:
                if img[i + y_diff, j + x_diff]:
                    adjacency[to_index(i, j, img.shape[1]),
                              to_index(i + y_diff, j + x_diff, img.shape[1])] = True

    return adjacency.tocsr()
# Döngü: Komşuluk Bağlantıları Oluşturma
# Açıklama: Her pikseli, 8 yönlü komşularıyla bağlantı kurarak komşuluk matrisini oluşturur.
# İterasyonlar:
# - y_diff, x_diff: Yükseklik (satır) ve genişlik (sütun) yönlü farklarını temsil eden tuple.
# Koşul:
# - Eğer pikselin komşusu siyah ise, komşuluk matrisinde bağlantıyı işaretle (True).
# Dönüş:
# - csr_matrix: Pikseller arasındaki bağlantıyı temsil eden seyrek bir matris (Compressed Sparse Row formatında).


# Heuristic function: Manhattan distance between two pixels
def heuristic_cost_estimate(node, goal, flat_img):
    y1, x1 = to_coordinates(node, flat_img.shape[1])
    y2, x2 = to_coordinates(goal, flat_img.shape[1])
    return cityblock((y1, x1), (y2, x2))
# Fonksiyon: heuristic_cost_estimate
# Açıklama: İki piksel arasındaki Manhattan mesafesini hesaplayan sezgisel bir maliyet tahmini yapar.
# Parametreler:
# - node (int): Başlangıç pikselinin 1D dizindeki indeksi.
# - goal (int): Hedef pikselinin 1D dizindeki indeksi.
# - flat_img (ndarray): Düzleştirilmiş resim verisi, NumPy dizisi olarak.
# Dönüş:
# - int: İki piksel arasındaki Manhattan mesafesi.


# A* algorithm to find the path between two pixels in the image
def find_path_a_star(img, source, target):
    adjacency = build_adjacency_matrix(img)

    start_time = time.time()
    queue = [(0, source)]
    closed_set = set()
    predecessors = {source: None}
    g_score = {source: 0}
    # Fonksiyon: find_path_a_star
    # Açıklama: A* algoritması kullanarak iki piksel arasındaki yolu bulur ve zamanı ölçer.
    # Parametreler:
    # - img (ndarray): Giriş olarak verilen resim verisi, NumPy dizisi olarak.
    # - source (int): Başlangıç pikselinin 1D dizindeki indeksi.
    # - target (int): Hedef pikselinin 1D dizindeki indeksi.
    # Dönüş:
    # - tuple: Düzleştirilmiş yolu ve algoritmanın çalışma süresini içeren bir tuple.

    while queue:
        current_cost, current_node = heapq.heappop(queue)

        if current_node == target:
            break

        closed_set.add(current_node)
        # Döngü: A* Algoritması İterasyonları
        # Açıklama: Öncelikli kuyruk yapısını kullanarak A* algoritmasının her bir iterasyonunu gerçekleştirir.
        # İterasyonlar:
        # - current_cost: Şu anki pikselin maliyeti, öncelikli kuyruktan çıkarıldığında elde edilir.
        # - current_node: Şu anki pikselin 1D dizindeki indeksi, öncelikli kuyruktan çıkarıldığında elde edilir.
        # Koşul:
        # - Eğer şu anki piksel hedef piksel ise döngüden çık.
        # İzleme:
        # - closed_set: Zaten değerlendirilmiş piksellerin kümesi.

        # Iterate over neighbors of the current pixel
        for neighbor in adjacency[current_node].nonzero()[1]:
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current_node] + 1

            # Update the score if the new path to the neighbor is shorter
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic_cost_estimate(neighbor, target, img)
                heapq.heappush(queue, (f_score, neighbor))
                predecessors[neighbor] = current_node

    pixel_index = target
    pixels_path = []
    # Döngü: Komşu Pikselleri İterasyonu ve Maliyet Güncelleme
    # Açıklama: Şu anki pikselin komşularını gezerek, maliyet güncellemeleri yapar ve öncelikli kuyruğa ekler.
    # İterasyonlar:
    # - neighbor: Şu anki pikselin komşuları, komşuluk matrisinden elde edilir.
    # Koşul:
    # - Eğer komşu piksel zaten değerlendirilmişse geç.
    # Maliyet Güncelleme:
    # - Eğer yeni yol komşu piksele daha kısa ise, maliyeti güncelle ve öncelikli kuyruğa ekle.
    # İzleme:
    # - g_score: Başlangıç pikselinden her piksele olan maliyetin toplamını içeren sözlük.
    # - f_score: A* algoritmasındaki toplam maliyet, g_score ve sezgisel tahminin toplamı.
    # - predecessors: Her pikselin önceki pikselini içeren sözlük.
    # - pixel_index: Hedef pikselin 1D dizindeki indeksi.
    # - pixels_path: Başlangıç pikselinden hedef piksele olan yolun 1D dizin indekslerini içeren liste.

    # Reconstruct the path from the target to the source
    while pixel_index != source:
        pixels_path.append(pixel_index)
        pixel_index = predecessors[pixel_index]

    pixels_path.reverse()

    smoothed_path = []

    # Interpolate the path for smoother visualization
    for pixel_index in pixels_path:
        i, j = to_coordinates(pixel_index, img.shape[1])
        smoothed_path.extend(zip(np.linspace(i, i + 1, 100), np.linspace(j, j + 1, 100)))

    smoothed_path = np.array(smoothed_path).astype(int)
    end_time = time.time()

    return smoothed_path, end_time - start_time
# Döngü: Yolu Geri Oluşturma ve Düzleştirme
# Açıklama: Hedef pikselden başlangıç pikseline kadar olan yolu önce geriye oluşturur, sonra düzleştirir.
# İterasyonlar:
# - pixel_index: Hedef pikselden başlangıç pikseline kadar olan yolu temsil eden piksellerin 1D dizindeki indeksleri.
# Geri Oluşturma:
# - Her bir pikselin önceki pikselini kullanarak yolu geriye oluşturur.
# Düzleştirme:
# - Her bir pikselin koordinatlarını alarak, 100 ara nokta ile bir çizgi oluşturur.
# - Oluşturulan çizgileri birleştirerek düzleştirilmiş bir yolu elde eder.
# İzleme:
# - smoothed_path: Başlangıç pikselinden hedef piksele olan düzleştirilmiş yol.
# - i, j: Her bir pikselin yükseklik (satır) ve genişlik (sütun) koordinatları.
# - np.linspace: İki koordinat arasında belirtilen sayıda eşit aralıklı ara nokta oluşturan NumPy fonksiyonu.
# - end_time: Algoritmanın tamamlanma zamanı.


# Function to visualize the original image and the computed path
def visualize_path(original_img, path):
    plt.imshow(original_img, cmap='gray' if len(original_img.shape) == 2 else None)

    path_array = np.array(path)
    plt.plot(path_array[:, 1], path_array[:, 0], color='blue', linewidth=7) # Yol kalınlığı "linewidth" operatörü ile değiştirilebilir. Her görsel için farklı linewidth kullanılabilir.

    plt.show()
    # Fonksiyon: visualize_path
    # Açıklama: Orijinal resmi ve hesaplanan yolu görselleştirir.
    # Parametreler:
    # - original_img (ndarray): Görselleştirilecek orijinal resim verisi, NumPy dizisi olarak.
    # - path (ndarray): Görselleştirilecek yolu temsil eden düzleştirilmiş piksel koordinatları, NumPy dizisi olarak.
    # Görselleştirme:
    # - Orijinal resmi gri tonlama (grayscale) veya renkli olarak gösterir.
    # - Hesaplanan yolu mavi renkte ve belirtilen kalınlıkta (linewidth) çizer.
    # - Kalınlık, görselin büyüklüğüne ve kullanıcı tercihine göre değiştirilebilir.


# Main function
def main():
    file_path = r"E:\Desktop\University classes and homeworks\Season 4 Episode 2\FENG-498\Images\183.jpg"  # Path to the image file
    original_img = load_image(file_path)
    flat_img = create_flat_image(original_img)

    source_y = 39
    source_x = 429
    source = to_index(source_y, source_x, original_img.shape[1])

    target_y = 442
    target_x = 930
    target = to_index(target_y, target_x, original_img.shape[1])

    # Find the path and measure the elapsed time
    path, elapsed_time = find_path_a_star(flat_img, source, target)

    # Visualize the original image with the computed path
    visualize_path(original_img, path)
    # Fonksiyon: main
    # Açıklama: Ana program akışını yönetir. Resmi yükler, başlangıç ve hedef noktalarını belirler, yolu hesaplar ve görselleştirir.
    # İşlemler:
    # - file_path: Resim dosyasının dosya yolunu içeren bir string.
    # - original_img: Resim dosyasını yükleyerek elde edilen orijinal resim verisi.
    # - flat_img: Orijinal resmin düzleştirilmiş halini oluşturan resim verisi.
    # - source_y, source_x: Başlangıç noktasının yükseklik (satır) ve genişlik (sütun) koordinatları.
    # - source: Başlangıç noktasının 1D dizindeki indeksi.
    # - target_y, target_x: Hedef noktasının yükseklik (satır) ve genişlik (sütun) koordinatları.
    # - target: Hedef noktasının 1D dizindeki indeksi.
    # - path: Başlangıç ve hedef noktaları arasındaki yolu içeren düzleştirilmiş piksel koordinatları.
    # - elapsed_time: Algoritmanın çalışma süresi.
    # - visualize_path: Hesaplanan yolu ve orijinal resmi görselleştiren fonksiyon.

    print(f"Elapsed time: {elapsed_time} seconds")

# Run the main function if the script is executed
if __name__ == "__main__":
    main()
    # Çıktı: Geçen Zamanı Yazdırma
    # Açıklama: Hesaplanan yolun bulunma süresini ekrana yazdırır.
    # İzleme:
    # - elapsed_time: Algoritmanın çalışma süresi.
    # Koşul:
    # - Eğer bu betik dosyası doğrudan çalıştırılıyorsa, main fonksiyonunu çalıştır.

