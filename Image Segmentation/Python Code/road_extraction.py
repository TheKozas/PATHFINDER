import cv2
import numpy as np
from matplotlib import pyplot as plt

# Görüntüleri yükleyelim
sat_img_path = 'C:/Users/ata_k/Desktop/Bitirme/Data Set/archive/train/104_sat.jpg'
mask_img_path = 'C:/Users/ata_k/Desktop/Bitirme/Data Set/archive/train/104_mask.png'

# Uydu görüntüsü ve maske görüntüsünü okuyalım
sat_image = cv2.imread(sat_img_path)
mask_image = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)

# Görüntülerin boyutlarını kontrol edelim
sat_image_shape = sat_image.shape[:2]
mask_image_shape = mask_image.shape

# Görüntülerin boyutlarını karşılaştıralım ve eşleşip eşleşmediğine bakalım
matching_dimensions = sat_image_shape == mask_image_shape

# Eşleşme durumuna göre devam edip etmeyeceğimize karar verelim
if not matching_dimensions:
    print(f"Görüntü boyutları uyuşmuyor. Uydu görüntüsü boyutları: {sat_image_shape}, Maske görüntüsü boyutları: {mask_image_shape}")
else:
    # İlk görselleştirme (Uydu ve Maske Görüntüsü)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(sat_image, cv2.COLOR_BGR2RGB))
    plt.title('Uydu Görüntüsü')
    plt.subplot(2, 2, 2)
    plt.imshow(mask_image, cmap='gray')
    plt.title('Maske Görüntüsü')

    # Kenar algılama için Canny algoritmasını kullanacağız
    lower_threshold = 50
    upper_threshold = 150
    edges = cv2.Canny(sat_image, lower_threshold, upper_threshold)

    # İkinci görselleştirme (Canny Kenar Algılama Sonucu)
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(sat_image, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal Uydu Görüntüsü')
    plt.subplot(2, 2, 4)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Kenar Algılama Sonucu')

    # Hough Çizgi Algılama işlemi için parametreleri belirleyelim
    rho = 1  # çizgilerin çözünürlüğü (pixel cinsinden)
    theta = np.pi / 180  # açı çözünürlüğü (radyan cinsinden)
    threshold = 50  # eşik değer, bu değerden daha fazla kesişen noktaları olan çizgiler tespit edilecek
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=100, maxLineGap=10)

    # Uydu görüntüsünün üzerine çizgileri çizelim
    sat_image_with_lines = sat_image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(sat_image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Üçüncü görselleştirme (Hough Çizgi Algılama Sonucu)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(sat_image_with_lines, cv2.COLOR_BGR2RGB))
    plt.title('Uydu Görüntüsü Üzerine Çizgiler')
    plt.subplot(1, 2, 2)
    plt.imshow(mask_image, cmap='gray')
    plt.title('Gerçek Maske Görüntüsü')

    # Morfolojik işlemler için çekirdek (kernel) tanımlayalım
    kernel = np.ones((5, 5), np.uint8)

    # Canny kenar algılama sonucunu iyileştirmek için morfolojik kapatma işlemi uygulayalım
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Kapatma işlemi sonrası elde edilen maskenin üzerinde genişletme işlemi uygulayalım
    dilation = cv2.dilate(closing, kernel, iterations=2)

    # Maskenin üzerine orijinal uydu görüntüsünü uygulayarak yolları çıkaralım
    road_extracted = cv2.bitwise_and(sat_image, sat_image, mask=dilation)

    # Sonuçları görselleştirelim
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(sat_image, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal Uydu Görüntüsü')
    plt.subplot(1, 3, 2)
    plt.imshow(dilation, cmap='gray')
    plt.title('Maskelenmiş Görüntü')
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(road_extracted, cv2.COLOR_BGR2RGB))
    plt.title('Çıkarılmış Yol')
    plt.show()
