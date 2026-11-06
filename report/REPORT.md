# BÁO CÁO FORMAL EVALUATION - LAB DAY 1: DATA OVERVIEW & YOLO11

- **Học viên:** [Hoàng Bá Đạt]
- ** 2A202603100 / Account:** [hdat2031-dotcom]
- **Môi trường thực thi:** Google Colab (Python 3.10+, PyTorch, Ultralytics v8.4.145)
- **Thiết bị (Device):** GPU (CUDA) / CPU
- **Tập dữ liệu:** 3 ảnh COCO công khai chuẩn (`traffic` id=210273, `kitchen` id=397133, `dining` id=166918)

---

## 1. Phân Loại Ảnh (Image Classification)

### 1.1. Bằng chứng thực nghiệm (Evidence Record)
- **Tệp nguồn:** `day1_lab_outputs/classification_predictions.json` và biểu đồ `visuals/classification_top5.png`.
- **Mẫu kiểm tra:** `sample = traffic` (COCO ID: `210273`, SHA256: `3ec23de63592...`).
- **Mô hình & Checkpoint:** `yolo11n-cls.pt` (ImageNet-1K taxonomy, 1000 classes).
- **Top-5 Dự đoán từ mô hình:**
  - `rank = 1`: `class_name = cab` | `score = 0.510`
  - `rank = 2`: `class_name = minibus` | `score = 0.160`
  - `rank = 3`: `class_name = police_van` | `score = 0.082`
  - `rank = 4`: `class_name = recreational_vehicle` | `score = 0.055`
  - `rank = 5`: `class_name = streetcar` | `score = 0.048`

### 1.2. Phân tích chuyên môn & Phân biệt Prediction vs Ground Truth
- **Bản chất của `rank = 1`:** Điểm số $0.510$ phản ánh xác suất chuẩn hóa (Softmax confidence score) cao nhất mà mô hình tính toán được trong không gian 1.000 nhãn của ImageNet, **hoàn toàn không phải là Ground Truth**.
- **Lỗi do bài toán đơn nhãn (Single-label Classification Limitation):**
  - Khung cảnh thực tế là một đại lộ đông đúc với nhiều chủ thể cùng xuất hiện: các xe buýt công cộng cỡ lớn màu xanh (transit bus), xe khách nhỏ (minibus), ô tô gia đình và người đi bộ trên cầu vượt.
  - Không có một chiếc taxi đơn lẻ nào chiếm toàn bộ khung hình. Việc mô hình dự đoán nhãn `cab` cho toàn bộ bức ảnh thể hiện sự bất lực của bài toán phân loại đơn nhãn khi áp dụng vào khung cảnh thế giới thực phức tạp (complex multi-object scene).

---

## 2. Phát Hiện Vật Thể (Object Detection)

### 2.1. Bằng chứng thực nghiệm (Evidence Record)
- **Tệp nguồn:** `day1_lab_outputs/detection_predictions.json` và hình ảnh `visuals/detection_predictions.png`.
- **Mẫu kiểm tra:** `sample = kitchen` (COCO ID: `397133`, SHA256: `09e1d25c75f7...`).
- **Mô hình & Checkpoint:** `yolo11n.pt` (huấn luyện trên COCO 80 lớp, định dạng tọa độ `bbox_xyxy`).
- **Các phát hiện tiêu biểu:**
  - `person` (người đầu bếp quay lưng): `score = 0.91`
  - `person` (cánh tay phía bên trái): `score = 0.61`
  - `oven` (bếp đun phía sau bên trái): `score = 0.69`
  - `oven` (khối tủ/lò sắt bên phải): `score = 0.63`
  - `bowl` (khuôn bánh oval trên bàn): `score = 0.72`
  - `bowl` (bát trên giá treo): `score = 0.70`, `bowl` (bát trên bàn): `score = 0.50`, `bowl = 0.38`
  - `cup` (hộp/ly trên bàn): `score = 0.45`

### 2.2. Phân tích lỗi & Ảnh hưởng của Confidence Threshold
- **Vật thể bị bỏ sót (False Negatives):**
  - Toàn bộ dàn xoong chảo bằng đồng và muôi sắt treo trên trần và vách tường bị bỏ sót hoàn toàn.
  - Cây cán bột (rolling pin) và các đĩa bánh tạc nướng trên bàn không được gắn nhãn.
  - Chiếc bàn làm bánh bằng gỗ lớn ở tiền cảnh (`dining table`) hoàn toàn bị mô hình bỏ sót trong tác vụ Detection.
- **Dự đoán nhầm / Mơ hồ (False Positives & Ambiguities):**
  - **Lỗi nhận diện một phần cơ thể:** Ở mép trái, chỉ có một phần cánh tay của người thợ phụ thò vào bàn, nhưng mô hình lại dự đoán cả một bounding box là `person (0.61)`.
  - **Khung chứa nhãn mơ hồ:** Khuôn làm bánh dạng oval có gân nổi được dự đoán là `bowl (0.72)`.
- **Tác động của Confidence Threshold:**
  - Nếu tăng threshold lên $0.50$: Các vật thể như `bowl (0.38)`, `bowl (0.46)`, `cup (0.45)` sẽ biến mất, làm tăng tỷ lệ bỏ sót (**False Negative Rate**).
  - Nếu hạ threshold xuống $0.20$: Sẽ xuất hiện thêm nhiều bounding box nhiễu ở khu vực tường gạch và các xoong chảo chồng chéo, làm tăng tỷ lệ báo động giả (**False Positive Rate**).

---

## 3. Phân Đoạn Cá Thể (Instance Segmentation)

### 3.1. Bằng chứng thực nghiệm (Evidence Record)
- **Tệp nguồn:** `day1_lab_outputs/segmentation_predictions.json` và hình ảnh `visuals/segmentation_prediction.png`.
- **Mẫu kiểm tra:** `sample = kitchen` (COCO ID: `397133`).
- **Mô hình & Checkpoint:** `yolo11n-seg.pt` (Dự đoán đồng thời Bounding Box và Polygon Mask).
- **Các thực thể phân đoạn đáng chú ý:**
  - `potted plant`: `score = 0.63` (khu vực lá khô treo góc trên bên trái).
  - `dining table`: `score = 0.58` (mặt bàn làm bánh ở trung tâm).
  - `spoon`: `score = 0.49` (chiếc muôi treo ở tường bên phải).
  - `person`: `score = 0.90` (người đầu bếp chính).
  - `oven`: `score = 0.49`, các `bowl = 0.74, 0.70, 0.53, 0.37`.

### 3.2. So sánh Bounding Box vs Instance Segmentation & Lỗi Nhận Diện Sai
- **Lỗi nhận thức nghiêm trọng (Hallucination / False Positive):**
  - Chùm lá thuốc lá/thảo mộc khô treo trên giàn bếp bị mô hình gán nhãn và phủ mặt nạ polygon màu cam là `potted plant 0.63`. Đây là sai sót nghiêm trọng do mô hình học máy bị đánh lừa bởi kết cấu dạng lá cây (foliage texture), dù trong thực tế không hề có chậu cây (pot) hay đất trồng.
- **Ưu thế vượt trội của Polygon Mask:**
  - Đối với `person (0.90)`: Mặt nạ polygon màu xanh lam ôm khít viền cơ thể, bờ vai và hai ống quần, loại bỏ hoàn toàn phần khoảng trống nền gạch giữa hai chân (điều mà Bounding Box hình chữ nhật không thể làm được).
  - Khả năng phục hồi vật thể: Ở tác vụ Detection, chiếc bàn chế biến bị bỏ sót, nhưng ở tác vụ Segmentation, mô hình đã nhận diện chính xác diện tích mặt bàn với nhãn `dining table (0.58)`.
  - Chiếc muôi treo tường được bọc sát ranh giới với nhãn `spoon (0.49)`.

---

## 4. Đề Xuất Quy Chuẩn Dữ Liệu & Kiểm Soát Chất Lượng (QC Guidelines)

Dựa trên các lỗi thực tế quan sát được từ 3 mô hình, nhóm đề xuất quy chuẩn gán nhãn thực tế như sau:

### 4.1. Quy tắc dành cho Chuyên viên gán nhãn (Annotator)
1. **Quy tắc bộ phận cơ thể (Partial Person Rule):** Chỉ gán nhãn `person` khi nhìn thấy tối thiểu phần đầu và thân trên (hoặc tối thiểu 30% diện tích cơ thể người). Không được đóng khung nhãn `person` nếu chỉ nhìn thấy một cánh tay hoặc bàn tay độc lập; những trường hợp này bỏ qua hoặc gắn thuộc tính `body_part` nếu có quy định riêng.
2. **Quy tắc vật thể che khuất (Occlusion Handling):** Khi các vật thể xếp chồng lên nhau (như các bát, khuôn bánh trên bàn làm bếp), annotator phải vẽ polygon bao sát đúng ranh giới nhìn thấy được thực tế (visible boundary), không tự ý phỏng đoán phần diện tích bị che khuất.
3. **Quy định ranh giới nhãn `potted plant`:** Chỉ gán nhãn `potted plant` khi hội tụ đủ 2 điều kiện: Có cây xanh/hoa và nằm trong một chậu/bình chứa xác định. Tuyệt đối không gán cho hoa khô, cành cây treo hoặc rau củ phơi trên bếp.

### 4.2. Quy tắc dành cho Kiểm định viên (Reviewer / QC) & Quy trình Escalation
1. **Kiểm tra độ khít của ranh giới (Boundary Fit Check):** Từ chối (Reject) bất kỳ polygon nào cắt lẹm quá 5% vào thân vật thể hoặc chứa quá 10% diện tích pixel nền không liên quan.
2. **Xử lý xung đột nhãn (Label Ambiguity & Escalation):**
   - Trường hợp vật thể nằm ở ranh giới giữa 2 lớp (ví dụ: khuôn làm bánh oval giữa `bowl` và `baking_pan`): Reviewer không tự ý quyết định mà phải gắn cờ (Flag) và chuyển lên cấp Quản lý dự án dữ liệu (Data Project Lead) thông qua quy trình Escalation để cập nhật tài liệu Guideline chính thức cho toàn đội ngũ.

---

## 5. Vòng Đời Dữ Liệu & Tính Tái Lập (Lifecycle & Reproducibility)

### 5.1. Vòng đời dữ liệu (Data Lifecycle)
- Thí nghiệm khẳng định nguyên lý cốt lõi: **Mô hình AI chỉ phản ánh những gì dữ liệu đã định nghĩa.**
- Để cải thiện mô hình, không thể dùng trực tiếp output của mô hình làm Ground Truth mà phải qua quy trình:
  $$\text{Dữ liệu thực tế} \longrightarrow \text{Guideline chuẩn hóa} \longrightarrow \text{Annotator gán nhãn} \longrightarrow \text{QC phê duyệt} \longrightarrow \text{Ground Truth} \longrightarrow \text{Huấn luyện & Đánh giá lỗi}.$$

### 5.2. Tính tái lập (Reproducibility)
- **Hệ điều hành:** Linux (Google Colab Environment).
- **Thư viện cố định:** `ultralytics == 8.4.145`, `torch == 2.x`.
- **Mã Checkpoint SHA256 chính thức:**
  - `yolo11n-cls.pt`: `c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7`
  - `yolo11n.pt`: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`
  - `yolo11n-seg.pt`: `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`
- **Bộ mẫu kiểm định:** Đầy đủ 3 mã COCO hash chuẩn (`traffic: 3ec23...`, `kitchen: 09e1d...`, `dining: a7f84...`).
- Không có bất kỳ sự thay đổi nào đối với code gốc, threshold hoặc cấu hình bài lab.
