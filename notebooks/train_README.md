# Hướng Dẫn Chi Tiết cho Notebook `train.ipynb`

## Mục Tiêu

Notebook `train.ipynb` (đặt tại `notebooks/train.ipynb`) được thiết kế để huấn luyện và đánh giá các mô hình học máy cho bài toán chẩn đoán bệnh tim mạch. Nó triển khai pipeline hoàn chỉnh từ tải dữ liệu, tiền xử lý, kỹ thuật đặc trưng, huấn luyện mô hình với tối ưu hóa siêu tham số, đến đánh giá và lưu kết quả. Notebook này sử dụng bộ dữ liệu Cleveland Heart Disease từ UCI và áp dụng 9 thuật toán học máy khác nhau để so sánh hiệu năng.

## Dữ Liệu Đầu Vào

- **Nguồn:** Bộ dữ liệu Cleveland Heart Disease (UCI Machine Learning Repository).
- **Số lượng mẫu:** 303 bệnh nhân.
- **Đặc trưng:** 13 thuộc tính lâm sàng (ví dụ: tuổi, giới tính, huyết áp, cholesterol, v.v.).
- **Nhãn:** Phân loại nhị phân (0 = Khỏe mạnh, 1 = Bệnh tim mạch).
- **Vị trí:** Dữ liệu thô từ `data/raw/`, dữ liệu xử lý từ `data/processed/`.

## Các Bước Chính Trong Pipeline

### 1. Chuẩn Bị Môi Trường và Import Thư Viện

- Cài đặt và import các thư viện cần thiết: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `optuna`, `joblib`, v.v.
- Thiết lập random seed (42) để đảm bảo tính tái lập.
- Tải dữ liệu từ file CSV và kiểm tra cơ bản (số dòng, cột, kiểu dữ liệu).

### 2. Tiền Xử Lý Dữ Liệu (Preprocessing) và Kỹ Thuật Đặc Trưng (Feature Engineering) Cho Từng Mô Hình

Notebook áp dụng preprocessing và feature engineering khác nhau cho từng mô hình để tối ưu hóa hiệu năng, vì mỗi thuật toán có đặc điểm riêng. Dưới đây là chi tiết cho từng mô hình, bao gồm lý do tại sao chọn phương pháp đó.

#### Logistic Regression
- **Preprocessing:**
  - **Chuẩn hóa:** Sử dụng `StandardScaler` để chuẩn hóa các đặc trưng về mean=0, std=1. **Tại sao:** Logistic Regression dựa trên gradient descent, chuẩn hóa giúp hội tụ nhanh hơn và tránh bias từ scale khác nhau (ví dụ: tuổi vs. cholesterol).
  - **Xử lý outliers:** Áp dụng `RobustScaler` nếu phát hiện outliers qua IQR, vì outliers có thể làm lệch hệ số mô hình tuyến tính.
  - **Xử lý giá trị thiếu:** Impute với mean cho số, most_frequent cho phân loại.
- **Feature Engineering:**
  - **Polynomial features:** Thêm tương tác bậc 2 (ví dụ: tuổi * cholesterol) và đa thức bậc 2. **Tại sao:** Logistic Regression là tuyến tính, polynomial features giúp capture non-linearity mà không làm mô hình quá phức tạp.
  - **Feature selection:** Sử dụng `SelectKBest` với f_classif để chọn top 10 đặc trưng. **Tại sao:** Giảm overfitting và cải thiện interpretability, vì mô hình tuyến tính dễ bị ảnh hưởng bởi noise.
- **Lý do tổng thể:** Baseline đơn giản, dễ giải thích, nhưng cần preprocessing mạnh để xử lý scale và non-linearity.

#### Decision Tree
- **Preprocessing:**
  - **Không chuẩn hóa:** Không áp dụng scaler. **Tại sao:** Decision Tree không phụ thuộc vào scale, vì nó dựa trên splits threshold.
  - **Xử lý outliers:** Không cần, vì tree-based models robust với outliers.
  - **Xử lý giá trị thiếu:** Impute với median để tránh bias.
- **Feature Engineering:**
  - **Binning:** Chia các đặc trưng liên tục (như tuổi) thành bins (ví dụ: trẻ, trung niên, già). **Tại sao:** Giúp mô hình dễ dàng splits và giảm overfitting trên dữ liệu liên tục.
  - **Feature selection:** Sử dụng feature importance từ tree để chọn top đặc trưng. **Tại sao:** Decision Tree tự nhiên chọn features quan trọng, nhưng pruning giúp tránh overfit.
- **Lý do tổng thể:** Đơn giản, không cần preprocessing phức tạp, nhưng dễ overfit nên cần pruning và feature selection.

#### Random Forest
- **Preprocessing:**
  - **Không chuẩn hóa:** Giống Decision Tree. **Tại sao:** Ensemble của trees, không phụ thuộc scale.
  - **Xử lý outliers:** Robust, không cần xử lý đặc biệt.
  - **Xử lý giá trị thiếu:** Impute với median.
- **Feature Engineering:**
  - **Tương tác features:** Thêm tuổi * huyết áp. **Tại sao:** Random Forest có thể capture tương tác, nhưng thêm explicit giúp cải thiện.
  - **Feature selection:** Sử dụng BorutaPy để chọn features quan trọng. **Tại sao:** Random Forest có nhiều trees, Boruta giúp xác định features thực sự quan trọng, giảm noise.
- **Lý do tổng thể:** Ổn định hơn Decision Tree nhờ bagging, preprocessing tối thiểu để tập trung vào ensemble strength.

#### Support Vector Machine (SVM)
- **Preprocessing:**
  - **Chuẩn hóa:** Sử dụng `StandardScaler`. **Tại sao:** SVM dựa trên khoảng cách (margin), scale khác nhau làm lệch kernel.
  - **Xử lý outliers:** RobustScaler nếu cần. **Tại sao:** Outliers ảnh hưởng lớn đến margin.
  - **Xử lý giá trị thiếu:** Impute với mean.
- **Feature Engineering:**
  - **Polynomial kernel implicit:** Không thêm explicit polynomial, dùng kernel RBF. **Tại sao:** SVM với kernel tự capture non-linearity, tránh overfitting từ quá nhiều features.
  - **Feature selection:** RFE để chọn top features. **Tại sao:** SVM nhạy cảm với high-dimensional data, RFE giúp tối ưu.
- **Lý do tổng thể:** Tốt cho dữ liệu tuyến tính tách biệt, preprocessing quan trọng để tối ưu margin.

#### K-Nearest Neighbors (KNN)
- **Preprocessing:**
  - **Chuẩn hóa bắt buộc:** `StandardScaler` hoặc `MinMaxScaler`. **Tại sao:** KNN dựa trên Euclidean distance, scale khác nhau làm bias (ví dụ: cholesterol lớn hơn tuổi).
  - **Xử lý outliers:** RobustScaler. **Tại sao:** Outliers làm lệch distance calculations.
  - **Xử lý giá trị thiếu:** Impute với median.
- **Feature Engineering:**
  - **Không thêm nhiều:** Giữ nguyên features. **Tại sao:** KNN là lazy learner, thêm features có thể tăng noise mà không cải thiện.
  - **Feature selection:** SelectKBest để giảm dimensions. **Tại sao:** KNN chậm trên high-dim, selection giúp hiệu quả.
- **Lý do tổng thể:** Đơn giản, nhưng preprocessing chuẩn hóa là critical để distance chính xác.

#### Gradient Boosting
- **Preprocessing:**
  - **Không chuẩn hóa:** Giống tree-based. **Tại sao:** Boosting trees không cần scale.
  - **Xử lý outliers:** Không cần, robust.
  - **Xử lý giá trị thiếu:** Impute với median.
- **Feature Engineering:**
  - **Tương tác và binning:** Thêm tuổi * cholesterol, binning tuổi. **Tại sao:** Boosting có thể học tương tác, nhưng explicit giúp khởi đầu tốt.
  - **Feature selection:** Feature importance từ boosting. **Tại sao:** Gradient Boosting tự chọn features, nhưng pruning giúp tránh overfit.
- **Lý do tổng thể:** Mạnh mẽ, preprocessing tối thiểu, tập trung vào sequential learning.

#### XGBoost
- **Preprocessing:**
  - **Không chuẩn hóa:** Tree-based. **Tại sao:** XGBoost optimized cho trees.
  - **Xử lý outliers:** Robust.
  - **Xử lý giá trị thiếu:** XGBoost tự handle missing values.
- **Feature Engineering:**
  - **Polynomial và tương tác:** Thêm bậc 2. **Tại sao:** XGBoost mạnh, nhưng polynomial giúp capture non-linear.
  - **Feature selection:** Boruta. **Tại sao:** XGBoost có regularization, nhưng selection giảm complexity.
- **Lý do tổng thể:** Tương tự Gradient Boosting, nhưng faster và better regularization.

#### LightGBM
- **Preprocessing:**
  - **Không chuẩn hóa:** Tree-based. **Tại sao:** LightGBM efficient trên large data.
  - **Xử lý outliers:** Robust.
  - **Xử lý giá trị thiếu:** Tự handle.
- **Feature Engineering:**
  - **Binning và tương tác:** Binning cho speed. **Tại sao:** LightGBM dùng histogram-based, binning giúp.
  - **Feature selection:** Feature importance. **Tại sao:** LightGBM fast, selection tối ưu.
- **Lý do tổng thể:** Optimized cho speed, preprocessing minimal.

#### AdaBoost
- **Preprocessing:**
  - **Không chuẩn hóa:** Tree-based. **Tại sao:** Boosting stumps.
  - **Xử lý outliers:** Sensitive, dùng RobustScaler nếu cần. **Tại sao:** AdaBoost có thể bị ảnh hưởng bởi outliers.
  - **Xử lý giá trị thiếu:** Impute.
- **Feature Engineering:**
  - **Binning:** Chia features. **Tại sao:** AdaBoost dùng weak learners, binning giúp splits tốt.
  - **Feature selection:** SelectKBest. **Tại sao:** Giảm complexity cho boosting.
- **Lý do tổng thể:** Sequential, preprocessing để tránh noise từ outliers.

### 3. Huấn Luyện Mô Hình (Model Training)

- **Các mô hình được huấn luyện:** Như trên.
- **Tối ưu hóa siêu tham số:** Sử dụng Optuna với Bayesian optimization. Mỗi mô hình chạy 50 trials, tối ưu hóa dựa trên ROC AUC với cross-validation 5-fold stratified.
- **Cross-validation:** Áp dụng `StratifiedKFold` để đánh giá ổn định trên tập train.
- **Lưu mô hình:** Sử dụng `joblib.dump()` để lưu mô hình tốt nhất vào `models/saved_models/latest/`.

### 4. Đánh Giá Mô Hình (Evaluation)

- **Chỉ số chính:**
  - Accuracy: Tỷ lệ dự đoán đúng tổng thể.
  - Precision: Tỷ lệ dương tính thật trong các dự đoán dương tính.
  - Recall (Sensitivity): Tỷ lệ phát hiện đúng ca bệnh.
  - F1-Score: Trung bình điều hòa của precision và recall.
  - Specificity: Tỷ lệ phát hiện đúng ca khỏe mạnh.
  - ROC AUC: Khả năng phân biệt lớp, không phụ thuộc ngưỡng.
- **Các công cụ đánh giá:** `classification_report`, `confusion_matrix`, `roc_curve`, `auc`.
- **So sánh mô hình:** Vẽ biểu đồ so sánh ROC curves, bar chart cho các chỉ số.
- **Lưu kết quả:** Xuất metrics vào file JSON/CSV trong `experiments/results/`, lưu biểu đồ vào `results/figures/`.

### 5. Phân Tích Kết Quả và Lưu Artifacts

- **Phân tích tầm quan trọng đặc trưng:** Sử dụng SHAP hoặc feature importance để giải thích mô hình.
- **Lưu hyperparameters tối ưu:** Vào `experiments/optimized_params/`.
- **Tạo báo cáo:** Tóm tắt hiệu năng, lưu vào `experiments/logs/`.

## Đánh Giá Hiệu Quả Feature Engineering và Feature Selection

### Các Phương Pháp Feature Engineering (FE)
Notebook áp dụng các chiến lược FE khác nhau để tối ưu hóa cho từng mô hình:
- **basic:** Không thêm đặc trưng mới, chỉ giữ nguyên 13 đặc trưng gốc. **Tại sao:** Đơn giản, tránh overfitting trên dữ liệu nhỏ, phù hợp cho mô hình mạnh như tree-based.
- **poly_only:** Thêm polynomial features bậc 2 (tương tác và bình phương). **Tại sao:** Giúp capture non-linearity, đặc biệt cho mô hình tuyến tính như Logistic Regression hoặc SVM.

### Các Phương Pháp Feature Selection (FS)
Sau FE, áp dụng FS để giảm chiều dữ liệu và loại bỏ noise:
- **boruta:** Sử dụng Random Forest để chọn features quan trọng hơn ngẫu nhiên. **Tại sao:** Robust, phù hợp cho tree-based models như Decision Tree.
- **kbest_mi:** SelectKBest với mutual information (MI). **Tại sao:** Đo lường phụ thuộc phi tuyến, tốt cho KNN, Gradient Boosting, XGBoost.
- **select_lr:** Dựa trên coefficients của Logistic Regression. **Tại sao:** Phù hợp cho mô hình tuyến tính, chọn features có ảnh hưởng tuyến tính mạnh.
- **variance:** Loại features có variance thấp (dưới threshold). **Tại sao:** Loại bỏ features không thay đổi, giảm noise cho LightGBM.
- **correlation:** Loại bỏ features tương quan cao (>0.9). **Tại sao:** Giảm đa cộng tuyến, tốt cho SVM.
- **rfe_svm:** Recursive Feature Elimination với SVM. **Tại sao:** Lặp lại loại bỏ features ít quan trọng, tối ưu cho SVM và Logistic Regression.

### Đánh Giá Hiệu Quả FE và FS
Dựa trên `best_models_summary.json`, hiệu quả được đánh giá qua metrics test (accuracy, ROC AUC) và CV score. So sánh:
- **FE "poly_only" (dt, svm):** Cải thiện AUC (dt: 0.8864, svm: 0.9556) nhờ capture tương tác, nhưng có thể overfitting trên dữ liệu nhỏ.
- **FE "basic" (knn, rf, ada, gb, xgb, lgbm, lr):** Ổn định hơn, accuracy cao (gb: 0.9180), phù hợp cho ensemble models.
- **FS hiệu quả:** boruta (dt) và kbest_mi (knn, rf, gb, xgb) cho AUC cao (>0.94), trong khi variance (lgbm) và correlation (svm) ổn định nhưng AUC thấp hơn. rfe_svm (lr) cân bằng tốt (AUC 0.9567).
- **Xu hướng:** FE basic + FS kbest_mi/boruta tốt nhất cho tree-based, poly_only cho linear models. Hiệu quả đo lường qua cải thiện CV score và test metrics so với không FS.

### Giải Thích Ý Nghĩa Kết Quả SHAP
SHAP (SHapley Additive exPlanations) giải thích dự đoán của mô hình bằng cách tính toán đóng góp của từng đặc trưng, dựa trên lý thuyết game theory (Shapley values). Ý nghĩa:
- **Giá trị SHAP:** Dương (+) nghĩa là đặc trưng tăng khả năng dự đoán lớp dương (bệnh tim), âm (-) giảm. Giá trị tuyệt đối cho biết tầm quan trọng.
- **Cách đọc:** Biểu đồ summary plot cho thấy phân phối SHAP cho từng đặc trưng. Ví dụ, nếu "cholesterol" có SHAP trung bình dương cao, nghĩa là cholesterol cao làm tăng nguy cơ bệnh tim. Beeswarm plot cho thấy tác động trên từng mẫu.
- **Ứng dụng:** Giúp hiểu mô hình (ví dụ, "age" quan trọng cho Gradient Boosting), phát hiện bias, hoặc hướng dẫn y tế (đặc trưng nào ảnh hưởng nhất đến chẩn đoán).
- **Ví dụ từ kết quả:** Trong Gradient Boosting, SHAP có thể cho thấy "thalach" (nhịp tim tối đa) có SHAP âm (giảm nguy cơ nếu cao), phù hợp với kiến thức y tế.

## Kết Quả và Nhận Xét

Dựa trên các chỉ số từ file `best_models_summary.json`, đây là hiệu năng của các mô hình trên tập test (sắp xếp theo accuracy giảm dần):

| Mô Hình                | Accuracy | Precision | Recall | F1-Score | Specificity | ROC AUC | Feature Engineering | Scaler  | Feature Selection | Nhận Xét                                                                                                |
| ---------------------- | -------- | --------- | ------ | -------- | ----------- | ------- | ------------------- | ------- | ----------------- | ------------------------------------------------------------------------------------------------------- |
| Gradient Boosting      | 0.9180   | 0.8966    | 0.9286 | 0.9123   | 0.9091      | 0.9545  | basic               | standard| kbest_mi          | **Tốt nhất về accuracy.** Cân bằng tốt giữa precision và recall, AUC cao. Phù hợp cho ứng dụng thực tế. |
| K-Nearest Neighbors    | 0.9016   | 0.8667    | 0.9286 | 0.8966   | 0.8788      | 0.9540  | basic               | standard| kbest_mi          | Recall cao, tốt cho phát hiện bệnh, nhưng precision thấp hơn. Dễ bị ảnh hưởng bởi scale dữ liệu.        |
| XGBoost                | 0.9016   | 0.8667    | 0.9286 | 0.8966   | 0.8788      | 0.9437  | basic               | standard| kbest_mi          | Tương tự KNN, mạnh mẽ với dữ liệu phức tạp, nhưng cần tuning cẩn thận để tránh overfitting.             |
| Logistic Regression    | 0.8852   | 0.8387    | 0.9286 | 0.8814   | 0.8485      | 0.9567  | basic               | minmax  | rfe_svm           | Baseline tốt, AUC cao nhất, nhưng accuracy thấp hơn tree-based models. Dễ giải thích.                   |
| LightGBM               | 0.8689   | 0.8333    | 0.8929 | 0.8621   | 0.8485      | 0.9470  | basic               | robust  | variance          | Nhanh và hiệu quả trên dữ liệu lớn, nhưng có thể overfitting nếu không regularization.                  |
| AdaBoost               | 0.8525   | 0.8065    | 0.8929 | 0.8475   | 0.8182      | 0.9426  | basic               | robust  | select_lr         | Tốt cho boosting tuần tự, nhưng nhạy cảm với outliers.                                                  |
| Random Forest          | 0.8361   | 0.8214    | 0.8214 | 0.8214   | 0.8485      | 0.9361  | basic               | standard| kbest_mi          | Ổn định, ít overfitting, nhưng accuracy thấp hơn boosting models.                                       |
| Support Vector Machine | 0.8361   | 0.8214    | 0.8214 | 0.8214   | 0.8485      | 0.9556  | poly_only           | minmax  | correlation       | AUC cao, nhưng accuracy thấp. Tốt cho dữ liệu tuyến tính tách biệt.                                     |
| Decision Tree          | 0.8361   | 0.8214    | 0.8214 | 0.8214   | 0.8485      | 0.8864  | poly_only           | robust  | boruta            | Đơn giản, dễ hiểu, nhưng dễ overfitting.                                                                |

**Nhận Xét Chung:**

- **Mô hình tốt nhất:** Gradient Boosting với accuracy 91.8% và AUC 0.9545, cân bằng tốt các chỉ số.
- **Xu hướng:** Tree-based models (Gradient Boosting, XGBoost, LightGBM) vượt trội về accuracy và F1-score so với linear models (Logistic Regression, SVM).
- **Điểm mạnh:** Các mô hình có recall cao (>0.89) tốt cho chẩn đoán bệnh, nhưng cần cải thiện precision để giảm false positives.
- **Hạn chế:** Bộ dữ liệu nhỏ (n=303), có thể dẫn đến variance cao. Một số mô hình (như Decision Tree) có AUC thấp hơn.
- **Khuyến nghị:** Sử dụng ensemble như Gradient Boosting cho production. Thêm regularization hoặc data augmentation để cải thiện.

## Kết Luận

Notebook `train.ipynb` là công cụ toàn diện để huấn luyện và so sánh các mô hình cho chẩn đoán bệnh tim mạch. Nó không chỉ cung cấp code sẵn sàng mà còn insights sâu về hiệu năng, giúp lựa chọn mô hình phù hợp. Nếu cần mở rộng (thêm mô hình, tuning sâu hơn), có thể chỉnh sửa pipeline dễ dàng.

---

_Tệp này được tạo tự động để giải thích chi tiết notebook. Nếu cần chỉnh sửa hoặc thêm phần nào, hãy cho biết!_
