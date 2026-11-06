# BÁO CÁO FORMAL EVALUATION - LAB DAY 1: DATA OVERVIEW & YOLO11

- **Học viên:** Hoàng Bá Đạt
- **Mã học viên / GitHub Account:** 2A202603100 / hdat2031-dotcom
- **Môi trường thực thi:** Google Colab (Python 3.10+, PyTorch 2.x, Ultralytics v8.4.145)
- **Thiết bị phần cứng (Hardware):** GPU (NVIDIA T4 CUDA) / CPU
- **Tập dữ liệu kiểm thử:** 3 ảnh COCO công khai chuẩn (`traffic` id=210273, `kitchen` id=397133, `dining` id=166918)
- **Quy tắc tuân thủ:** Giữ nguyên 100% checkpoint gốc, threshold mặc định, không tự sửa mã nguồn; không dùng dữ liệu ngoài.

---

## 1. Phân Loại Ảnh (Image Classification)

### 1.1. Bằng chứng thực nghiệm (Evidence Record)
- **Tệp nguồn:** `day1_lab_outputs/classification_predictions.json` và biểu đồ trực quan `visuals/classification_top5.png`.
- **Mẫu kiểm tra:** `sample = traffic` (COCO ID: `210273`, mã băm SHA256: `3ec23de63592...`).
- **Mô hình & Checkpoint:** `yolo11n-cls.pt` (huấn luyện trên không gian 1.000 lớp ImageNet-1K).
- **Bảng Top-5 Dự đoán từ mô hình:**

| Thứ hạng (Rank) | Nhãn dự đoán (`class_name`) | Điểm tin cậy (`score`) | Phần trăm (%) |
| :---: | :--- | :---: | :---: |
| **Top 1** | `cab` (Taxi) | **0.510** | 51.0% |
| **Top 2** | `minibus` (Xe khách nhỏ) | **0.160** | 16.0% |
| **Top 3** | `police_van` (Xe cảnh sát) | **0.082** | 8.2% |
| **Top 4** | `recreational_vehicle` (Xe cắm trại) | **0.055** | 5.5% |
| **Top 5** | `streetcar` (Tàu điện đường phố) | **0.048** | 4.8% |

### 1.2. Phân tích chuyên môn & Phân biệt Prediction vs Ground Truth
- **Bản chất của điểm số Top 1 (`score = 0.510`):**
  - Điểm số $0.510$ thực chất chỉ là xác suất chuẩn hóa qua hàm Softmax tại tầng đầu ra cuối cùng của mạng nơ-ron trong không gian 1.000 lớp ImageNet.
  - Điểm số này **hoàn toàn là dự đoán của mô hình (Prediction)**, **không phải là sự thật thực tế (Ground Truth)**.
- **Hạn chế cốt lõi của bài toán phân loại đơn nhãn (Single-label Limitation):**
  - Khung cảnh thực tế của bức ảnh là một đại lộ giao thông đông đúc (multi-object scene) gồm nhiều chủ thể cùng xuất hiện: các xe buýt công cộng màu xanh cỡ lớn (`transit bus`), xe khách nhỏ (`minibus`), nhiều xe ô tô con cá nhân và dòng người đi bộ trên cầu vượt.
  - Trong ảnh không có một chiếc xe taxi (`cab`) độc lập nào chiếm trọn khung hình. Việc mô hình bắt buộc phải gán một nhãn duy nhất cho toàn bộ bức ảnh đã bộc lộ sự bất lực cố hữu của bài toán Phân loại đơn nhãn khi tiếp cận các cảnh phức tạp trong thế giới thực.

---

## 2. Phát Hiện Vật Thể (Object Detection)

### 2.1. Bằng chứng thực nghiệm (Evidence Record)
- **Tệp nguồn:** `day1_lab_outputs/detection_predictions.json` và hình ảnh `visuals/detection_predictions.png`.
- **Mẫu kiểm tra:** `sample = kitchen` (COCO ID: `397133`, mã băm SHA256: `09e1d25c75f7...`).
- **Mô hình & Checkpoint:** `yolo11n.pt` (huấn luyện trên tập dữ liệu COCO 80 lớp, định dạng tọa độ hộp bao `bbox_xyxy`).
- **Danh sách phát hiện tiêu biểu trích xuất từ JSON:**

| Lớp đối tượng (`class_name`) | Vị trí / Đối tượng thực tế | Điểm tin cậy (`score`) |
| :--- | :--- | :---: |
| `person` | Người đầu bếp chính đứng quay lưng | **0.91** |
| `person` | Cánh tay thợ phụ thò vào mép trái bàn | **0.61** |
| `oven` | Cụm bếp đun phía sau bên trái | **0.69** |
| `oven` | Khối tủ/lò sấy kim loại bên phải | **0.63** |
| `bowl` | Khuôn bánh oval có gân nổi trên bàn | **0.72** |
| `bowl` | Bát kim loại trên giá treo tường | **0.70** |
| `bowl` | Bát đặt trên mặt bàn bếp | **0.50** |
| `bowl` | Bát nhỏ ở góc bàn | **0.38** |
| `cup` | Hộp/ly kim loại trên bàn | **0.45** |

### 2.2. Phân tích lỗi & Ảnh hưởng của Ngưỡng tin cậy (Confidence Threshold)
- **Vật thể bị bỏ sót (False Negatives - Báo động trượt):**
  - **Dàn dụng cụ nhà bếp treo tường:** Toàn bộ dãy xoong chảo bằng đồng, nắp vung và muôi kim loại treo trên trần và vách tường hậu cảnh bị mô hình bỏ sót hoàn toàn (không có bounding box nào).
  - **Dụng cụ làm bánh:** Cây cán bột bằng gỗ (`rolling pin`) và các đĩa bánh nướng trên bàn không được nhận diện.
  - **Mặt bàn bếp lớn:** Chiếc bàn gỗ chế biến cỡ lớn ở tiền cảnh (`dining table`) hoàn toàn vắng bóng trong kết quả Detection.
- **Dự đoán nhầm / Báo động giả (False Positives & Label Ambiguities):**
  - **Lỗi nhận diện bộ phận cơ thể thành toàn bộ người:** Ở mép trái khung hình, chỉ có một phần cánh tay và bàn tay của người thợ phụ thò vào bàn, nhưng mô hình lại vẽ một bounding box và gán nhãn là `person` với độ tin cậy khá cao ($0.61$).
  - **Nhãn phân loại mơ hồ:** Chiếc khuôn nướng bánh dạng oval có gân nổi được mô hình nhận diện là `bowl` ($0.72$) do thiếu nhãn chuyên biệt trong 80 lớp COCO.
- **Tác động của Confidence Threshold:**
  - Nếu **tăng threshold lên $0.50$:** Các đối tượng có điểm tin cậy trung bình yếu như `bowl (0.38)`, `cup (0.45)` sẽ bị lọc bỏ, làm giảm báo động giả nhưng **làm tăng mạnh tỷ lệ bỏ sót (False Negative Rate tăng, Recall giảm)**.
  - Nếu **hạ threshold xuống $0.20$:** Mô hình sẽ giữ lại được nhiều vật thể nhỏ, nhưng sẽ sinh ra hàng loạt bounding box nhiễu (noise) ở các mảng gạch men tường và cụm xoong chảo phức tạp phía trên (**False Positive Rate tăng, Precision giảm)**.

---

## 3. Phân Đoạn Cá Thể (Instance Segmentation)

### 3.1. Bằng chứng thực nghiệm (Evidence Record)
- **Tệp nguồn:** `day1_lab_outputs/segmentation_predictions.json` và hình ảnh `visuals/segmentation_prediction.png`.
- **Mẫu kiểm tra:** `sample = kitchen` (COCO ID: `397133`).
- **Mô hình & Checkpoint:** `yolo11n-seg.pt` (Dự đoán đồng thời Bounding Box và Polygon Mask).
- **Các thực thể phân đoạn ghi nhận trong JSON:**

| Lớp đối tượng (`class_name`) | Vị trí / Chi tiết phân đoạn | Điểm tin cậy (`score`) | Dạng hình học |
| :--- | :--- | :---: | :---: |
| `potted plant` | Chùm lá thảo mộc khô treo giàn bếp góc trên | **0.63** | Polygon Mask |
| `dining table` | Toàn bộ diện tích mặt bàn làm bánh | **0.58** | Polygon Mask |
| `spoon` | Chiếc muôi kim loại treo trên vách tường | **0.49** | Polygon Mask |
| `person` | Người đầu bếp chính ở trung tâm | **0.90** | Polygon Mask |
| `oven` | Bếp lò phía sau | **0.49** | Polygon Mask |
| `bowl` | Các bát trên giá và trên bàn | **0.74, 0.70, 0.53, 0.37** | Polygon Mask |

### 3.2. So sánh Bounding Box vs Instance Segmentation & Lỗi Nhận Diện Sai
- **Lỗi nhận thức nghiêm trọng (Hallucination / False Positive):**
  - Chùm lá thuốc lá/thảo mộc phơi khô treo lủng lẳng trên giàn bếp bị mô hình bọc mặt nạ polygon màu cam và gán nhãn `potted plant (0.63)`.
  - **Nguyên nhân kỹ thuật:** Mạng nơ-ron tích chập (CNN/Backbone) đã bị đánh lừa bởi đặc trưng kết cấu dạng tán lá cây (`foliage texture`), trong khi thực tế không hề tồn tại chậu cây (`pot`) hay đất trồng. Đây là bằng chứng điển hình về việc mô hình chỉ bắt chước hoa văn bề mặt mà không có hiểu biết ngữ cảnh thực sự.
- **Ưu thế vượt trội của Polygon Mask so với Bounding Box:**
  - **Độ chính xác không gian (Pixel-level accuracy):** Với `person (0.90)`, mặt nạ polygon màu xanh lam bám sát từng đường viền bờ vai, cánh tay áo và hai ống quần, loại bỏ hoàn toàn mảng pixel nền sàn gạch nằm giữa hai chân người (điều mà bounding box hình chữ nhật chắc chắn sẽ bao thừa).
  - **Khả năng phục hồi vật thể:** Chiếc bàn gỗ chế biến bị bỏ sót hoàn toàn ở Detection thì ở Segmentation đã được mô hình khôi phục với nhãn `dining table (0.58)`.
  - Chiếc muôi treo tường được bọc viền chuẩn xác với nhãn `spoon (0.49)`.

---

## 4. Đề Xuất Quy Chuẩn Dữ Liệu & Kiểm Soát Chất Lượng (QC Guidelines)

Để khắc phục các sai sót thực tế quan sát được từ mô hình YOLO11, nhóm đề xuất bộ quy chuẩn gán nhãn (Annotation Guideline) và quy trình kiểm soát chất lượng (QC SOP) như sau:

### 4.1. Quy tắc dành cho Chuyên viên gán nhãn (Annotator Guidelines)
1. **Quy tắc bộ phận cơ thể (Partial Person Rule):**
   - Chỉ được gán nhãn `person` khi quan sát thấy tối thiểu phần đầu và thân trên (hoặc chiếm từ $30\%$ diện tích cơ thể người trở lên).
   - Nếu trong khung hình chỉ xuất hiện một phần cánh tay, bàn tay hoặc một cẳng chân đơn lẻ thò vào mép ảnh, annotator **tuyệt đối không gán nhãn `person`** mà phải bỏ qua (hoặc gắn nhãn thuộc tính phụ `body_part` nếu dự án có yêu cầu).
2. **Quy tắc xử lý vật thể che khuất (Occlusion Handling):**
   - Đối với các vật thể xếp chồng hoặc che lấp lẫn nhau (như chồng bát đĩa, khay bánh trên bàn), annotator chỉ vẽ polygon bao quanh diện tích pixel thực tế nhìn thấy được (**visible boundary**), không được tự ý phỏng đoán hay vẽ lấn sang vùng bị che khuất.
3. **Quy chuẩn định danh `potted plant`:**
   - Nhãn `potted plant` chỉ được phê duyệt khi hội tụ đủ 2 yếu tố: (1) Có cành/lá/hoa thực vật và (2) Nằm trong một chậu, bình hoặc bồn chứa cây xác định.
   - Hoa khô treo tường, thảo mộc phơi trên giàn hoặc rau củ quả đặt trên bàn tuyệt đối không được gán nhãn `potted plant`.

### 4.2. Quy tắc dành cho Kiểm định viên (Reviewer / QC) & Quy trình Escalation
1. **Tiêu chuẩn kiểm định độ khít đường viền (Boundary Fit Standard):**
   - Từ chối phê duyệt (Reject) bất kỳ polygon nào cắt lẹm quá $5\%$ vào diện tích vật thể, hoặc bao chứa quá $10\%$ diện tích pixel của hậu cảnh xung quanh.
2. **Quy trình xử lý trường hợp mơ hồ (Label Ambiguity & Escalation):**
   - Khi gặp vật thể nằm ở ranh giới phân loại không rõ ràng (ví dụ: chiếc khuôn nướng bánh oval nằm giữa `bowl` và `baking_pan`):
   - Reviewer không được tự ý đưa ra quyết định chủ quan mà phải gắn cờ cảnh báo (**Flag**) và chuyển tiếp (**Escalate**) lên Quản lý dự án dữ liệu (Data Project Lead).
   - Data Project Lead có trách nhiệm hội chẩn và cập nhật ngay định nghĩa bổ sung vào tài liệu Guideline chính thức cho toàn bộ đội ngũ.

---

## 5. Vòng Đời Dữ Liệu & Tính Tái Lập (Lifecycle & Reproducibility)

### 5.1. Vòng đời dữ liệu (Data Lifecycle)
- Kết quả thí nghiệm khẳng định định luật cốt lõi trong Trí tuệ Nhân tạo: **"Mô hình AI chỉ phản ánh những gì tập dữ liệu huấn luyện đã định nghĩa."**
- Không thể lấy trực tiếp output dự đoán của mô hình để làm nhãn chuẩn, mà bắt buộc phải tuân thủ chu trình khép kín:
  $$\text{Dữ liệu thực tế} \longrightarrow \text{Guideline chuẩn hóa} \longrightarrow \text{Annotator gán nhãn} \longrightarrow \text{QC kiểm duyệt} \longrightarrow \text{Ground Truth} \longrightarrow \text{Huấn luyện & Đánh giá lỗi}.$$

### 5.2. Tính tái lập thực nghiệm (Reproducibility)
- **Môi trường:** Google Colab Linux (Ubuntu 22.04 LTS, Python 3.10+, PyTorch 2.x).
- **Thư viện cố định:** `ultralytics == 8.4.145`.
- **Mã băm SHA256 chính thức của các Checkpoint:**
  - `yolo11n-cls.pt`: `c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7`
  - `yolo11n.pt`: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`
  - `yolo11n-seg.pt`: `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`
- **Mã băm SHA256 của 3 ảnh mẫu kiểm định:**
  - `traffic` (`id=210273`): `3ec23de63592...`
  - `kitchen` (`id=397133`): `09e1d25c75f7...`
  - `dining` (`id=166918`): `a7f8475871b6...`
- **Cam kết:** Báo cáo giữ nguyên 100% mã nguồn gốc, không thay đổi checkpoint và không thay đổi bất kỳ ngưỡng threshold nào theo đúng quy tắc bài lab.
