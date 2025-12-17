🖼️ Image Captioning with Transformer (Flickr8k)
1. Giới thiệu

Project này xây dựng hệ thống Image Captioning – sinh mô tả ngôn ngữ tự nhiên cho ảnh – dựa trên kiến trúc Encoder–Decoder với Transformer, huấn luyện và đánh giá trên dataset Flickr8k.

Ứng dụng cho phép:

Load ảnh bất kỳ

Sinh caption bằng Beam Search

Chạy bằng GUI (Tkinter) hoặc inference bằng code

2. Cấu trúc thư mục
.
├── models/                 # Encoder, Decoder, CaptionModel
├── datasets/               # Flickr8kDataset, transforms
├── train/                  # Trainer, training loop
├── utils/                  # Vocabulary, preprocessing
├── saved/model/             # Model đã train (.pth)
├── sample/                 # Ảnh test
├── caption_gui_app.py       # Ứng dụng GUI
├── requirements.txt
├── README.md
└── config.py

3. Kích hoạt môi trường ảo
🔹 Tạo môi trường ảo
python -m venv venv

🔹 Kích hoạt

Windows

venv\Scripts\activate


Linux / macOS

source venv/bin/activate

4. Cài đặt thư viện
🔹 Cài từ requirements.txt
pip install -r requirements.txt

🔹 requirements.txt (tối thiểu)
torch
torchvision
Pillow
tqdm
nltk


⚠️ Với GPU CUDA, cài PyTorch theo hướng dẫn tại:
https://pytorch.org/get-started/locally/

5. Tải dataset Flickr8k
🔹 Cách khuyến nghị: Kaggle API
Cài Kaggle
pip install kaggle

Tải dataset
kaggle datasets download -d adityajn105/flickr8k
unzip flickr8k.zip -d data/flickr8k


📁 Cấu trúc sau khi giải nén:

data/flickr8k/
├── Images/
├── captions.txt

6. Chuẩn bị tài nguyên NLP (NLTK)

Project sử dụng NLTK để tokenize caption.

python -m nltk.downloader punkt

7. Chạy ứng dụng GUI
python caption_gui_app.py

Chức năng:

Upload ảnh

Hiển thị ảnh

Sinh caption bằng Beam Search

8. Mô tả kiến trúc mô hình
🔹 Encoder

Sử dụng CNN pretrained (hoặc embedding layer)

Trích xuất đặc trưng ảnh

🔹 Decoder

Transformer Decoder

Sinh chuỗi từ dựa trên:

Feature ảnh

Các từ đã sinh trước đó

🔹 Quy trình
Image → Encoder → Feature Vector
       ↓
   Transformer Decoder → Caption

9. Beam Search Decoding

Thay vì Greedy Search, project sử dụng Beam Search để cải thiện chất lượng caption.

Giữ lại K câu ứng viên tốt nhất

Chọn câu có log-probability cao nhất

Ưu điểm:

Caption tự nhiên hơn

Ít lỗi ngữ nghĩa

10. Loss Function

Cross Entropy Loss

Dự đoán từ tiếp theo dựa trên ground truth

11. Đánh giá mô hình

Quan sát chất lượng caption sinh ra

So sánh Greedy vs Beam Search

Đánh giá định tính (qualitative)

12. Công nghệ sử dụng
Thành phần	Công nghệ
Framework	PyTorch
Vision	Torchvision
NLP	NLTK
GUI	Tkinter
Dataset	Flickr8k
