import pathlib
import cv2
import os

# Membuat direktori untuk menyimpan hasil foto jika belum ada
output_dir = "hasil_foto"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Menentukan path ke file Haar Cascade untuk deteksi wajah
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

# Membuat instance classifier
clf = cv2.CascadeClassifier(str(cascade_path))

# Mengaktifkan kamera utama (biasanya webcam internal)
camera = cv2.VideoCapture(0)

# Variabel untuk melacak status deteksi dan penomoran file
img_counter = 0
face_detected_previously = False

print("✅ Kamera siap. Arahkan wajah ke kamera untuk memfoto otomatis.")
print("ℹ️ Tekan tombol 'q' untuk keluar.")

while True:
    # Membaca frame dari kamera
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)
    if not _:
        print("❌ Gagal membaca frame dari kamera. Keluar...")
        break

    # Mengubah frame menjadi grayscale untuk efisiensi deteksi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Melakukan deteksi wajah
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Memeriksa apakah ada wajah yang terdeteksi di frame saat ini
    is_face_detected_now = len(faces) > 0

    # LOGIKA UTAMA: Ambil foto HANYA saat wajah pertama kali terdeteksi
    if is_face_detected_now and not face_detected_previously:
        # Tentukan nama file dan simpan gambar
        img_name = f"{output_dir}/wajah_terdeteksi_{img_counter}.png"
        cv2.imwrite(img_name, frame)
        print(f"📸 Foto berhasil disimpan: {img_name}")
        img_counter += 1

    # Perbarui status deteksi untuk frame berikutnya
    face_detected_previously = is_face_detected_now

    # Menggambar kotak hijau dan teks di sekitar setiap wajah yang terdeteksi
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frame, "Wajah Terdeteksi", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Menampilkan jendela dengan output kamera
    cv2.imshow("Aplikasi Foto Otomatis", frame)

    # Berhenti jika tombol 'q' ditekan
    if cv2.waitKey(1) == ord("q"):
        break

# Melepaskan kamera dan menutup semua jendela
camera.release()
cv2.destroyAllWindows()
print("Program ditutup.")
