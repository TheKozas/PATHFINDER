import itertools  # Iterasyon işlemleri için itertools modülü import ediliyor.
import imageio  # Resim okuma işlemleri için imageio modülü import ediliyor.
import numpy as np  # Sayısal hesaplamalar için numpy modülü import ediliyor.
from scipy.sparse import dok_matrix  # Sparse matris işlemleri için dok_matrix sınıfı import ediliyor.
from scipy.sparse.csgraph import dijkstra  # Dijkstra algoritması için dijkstra fonksiyonu import ediliyor.
import matplotlib.pyplot as plt  # Grafik çizimleri için matplotlib modülü import ediliyor.
import time  # Zaman ölçümleri için time modülü import ediliyor.

def load_image(file_path):
    # Belirtilen dosya yolundaki resmi yükle ve döndür.
    return imageio.v2.imread(file_path)

def create_flat_image(img):
    # Eğer resim üç kanallı (RGB) ise sadece birinci kanalı al, yoksa aynı resmi kullan.
    if len(img.shape) == 3:
        return img[:, :, 0]
    else:
        return img

def to_index(y, x, width):
    # Koordinatları matris indisine dönüştür.
    return y * width + x

def to_coordinates(index, width):
    # Matris indisini koordinatlara dönüştür.
    y, x = divmod(index, width)
    return y, x

def build_adjacency_matrix(img):
    # Verilen resmin bitişiklik matrisini oluştur ve dok_matrix formatında sakla.
    img_size = img.size
    adjacency = dok_matrix((img_size, img_size), dtype=bool)

    # Pikseller arası bitişiklik ilişkilerini kontrol et.
    # Her yönde (0, 1, -1) birer birim kayma sağlayan koordinat çiftlerini içeren bir liste oluşturuluyor.
    directions = list(itertools.product([0, 1, -1], [0, 1, -1]))

    # Resmin kenar piksellerini kontrol etmek ve bitişiklik ilişkilerini oluşturmak için iki döngü kullanılıyor.
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            # Eğer piksel siyah (0) ise, bu pikseli görmezden gel ve diğerlerine geç.
            if not img[i, j]:
                continue

            # Her bir yönde (komşu pikseller) dolaşarak bitişiklik ilişkilerini kontrol et.
            for y_diff, x_diff in directions:
                # Eğer komşu piksel siyah (1) ise, bu iki piksel arasında bir bitişiklik ilişkisi vardır.
                if img[i + y_diff, j + x_diff]:
                    # Bitişiklik matrisinde ilgili indisleri True olarak işaretle.
                    adjacency[to_index(i, j, img.shape[1]),
                    to_index(i + y_diff, j + x_diff, img.shape[1])] = True

    # Oluşturulan bitişiklik matrisi döndürülüyor.
    return adjacency

def find_path(img, source, target):
    # Verilen resimdeki başlangıç ve hedef noktalar arasındaki en kısa yolu bul.
    adjacency = build_adjacency_matrix(img)

    start_time = time.time()  # Zaman ölçümü başlatılıyor.
    # Dijkstra algoritması kullanılarak en kısa yolu ve önceki düğümleri bul.
    _, predecessors = dijkstra(adjacency, directed=False, indices=[source],
                               unweighted=True, return_predecessors=True)
    # Burada, adjacency matrisi üzerinde Dijkstra algoritması kullanılıyor.
    # directed=False, çizgesinin yönlendirilmemiş olduğunu belirtir.
    # indices=[source], başlangıç düğümünü belirtir.
    # unweighted=True, ağırlıkların kullanılmadığını ve tüm kenarların aynı uzunluğa sahip olduğunu belirtir.
    # return_predecessors=True, Dijkstra'nın önceki düğümleri de döndürmesini sağlar.

    # Dijkstra algoritması sonucunda elde edilen en kısa yolu ve önceki düğümleri içeren
    # predecessors değişkeni kullanılabilir. Bu bilgiler daha sonra en kısa yolun oluşturulması için kullanılacaktır.

    print("Shape of predecessors array:", predecessors.shape)

    pixel_index = target
    pixels_path = []
    while pixel_index != source:
        # Hedeften başlayarak geriye doğru yol üzerindeki pikselleri bul.
        pixels_path.append(pixel_index)

        # Eğer pixel_index geçerli bir aralıkta değilse, hata mesajı yazdır ve döngüyü kır.
        if pixel_index < 0 or pixel_index >= predecessors.shape[1]:
            print(f"Invalid pixel_index: {pixel_index}")
            break

        # Bir önceki düğümü bulmak için predecessors matrisini kullan.
        pixel_index = predecessors[0, pixel_index]

    # Yolu içeren piksel dizisi oluşturuluyor.
    smoothed_path = []
    for pixel_index in pixels_path:
        # Bulunan yolun her pikselini daha düzgün bir yola dönüştür.
        i, j = to_coordinates(pixel_index, img.shape[1])

        # Yolu daha düzgün hale getirmek için piksel aralıklarını lineer bir şekilde böler.
        smoothed_path.extend(zip(np.linspace(i, i + 1, 100), np.linspace(j, j + 1, 100)))

    # Elde edilen düzgün yolu içeren array'i oluştur ve tamsayıya dönüştür.
    smoothed_path = np.array(smoothed_path).astype(int)
    end_time = time.time()  # Zaman ölçümü sona eriyor.

    # Düzgün yolu ve işlem süresini döndür.
    return smoothed_path, end_time - start_time


def visualize_path(original_img, path):
    # Verilen resmi ve bulunan yolu görselleştir.

    # Resmi matplotlib kütüphanesini kullanarak görselleştir. Eğer resim siyah-beyaz (2D) ise 'gray',
    # renkli (3D) ise renkleri koru şeklinde renklendirme yapılır.
    plt.imshow(original_img, cmap='gray' if len(original_img.shape) == 2 else None)

    # Yolu içeren array'i kullanarak grafiği çiz. Yol, mavi renkte, kalınlığı 3 piksel olarak belirtilmiştir.
    path_array = np.array(path)
    plt.plot(path_array[:, 1], path_array[:, 0], color='blue', linewidth=3)

    # Grafiği göster.
    plt.show()

def main():
    # Ana program fonksiyonu

    # İşlenecek resmin dosya yolu belirleniyor.
    file_path = r"E:\Desktop\University classes and homeworks\Season 4 Episode 2\FENG-498\Images\183.jpg"  # Sabit dosya yolu

    # Belirtilen dosya yolundaki resim yükleniyor.
    original_img = load_image(file_path)

    # Eğer resim üç kanallı (RGB) ise sadece birinci kanalı al, yoksa aynı resmi kullan.
    flat_img = create_flat_image(original_img)

    # Başlangıç ve hedef piksel koordinatları belirleniyor ve bu koordinatlar indekslere dönüştürülüyor.
    source_y = 39
    source_x = 429
    source = to_index(source_y, source_x, original_img.shape[1])

    target_y = 442
    target_x = 930
    target = to_index(target_y, target_x, original_img.shape[1])

    # En kısa yolu ve geçen süreyi hesapla.
    path, elapsed_time = find_path(flat_img, source, target)

    # Bulunan yolu ve resmi görselleştir.
    visualize_path(original_img, path)

    # Hesaplanan işlem süresini ekrana yazdır.
    print(f"Elapsed time: {elapsed_time} seconds")


if __name__ == "__main__":
    main()
