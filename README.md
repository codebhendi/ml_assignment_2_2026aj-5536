Github repository link: https://github.com/codebhendi/ml_assignment_2_2026aj-5536

StreamLit app link: https://rq47b94pwurc2yqpuxshfs.streamlit.app/


## Problem Statement
Customer churn poses a significant financial challenge in the e-commerce sector, where retaining existing users is considerably more cost-effective than acquiring new ones. Predicting churn requires processing complex behavioral data—such as order history, tenure, payment preferences, and customer satisfaction—which contains mixed data types, missing values, and class imbalance.
To address this, this project constructed a production-grade machine learning workflow using an E-Commerce Churn Dataset satisfying all assignment constraints ($>500$ instances and $>12$ features). The primary goal was to evaluate and benchmark 5 distinct machine learning algorithms—Logistic Regression, Decision Tree Classifier, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Support Vector Machine (SVM)—under strict data leakage prevention protocols. Furthermore, the project aimed to decouple training from inference by serializing the best-performing preprocessor and model artifacts using Joblib and deploying a lightweight, interactive evaluation dashboard on Streamlit Cloud.

## Dataset Description
The E-Commerce Customer Churn Dataset represents customer churn data collected from an online retail platform. It captures historical customer demographics, purchasing behaviors, platform usage, and service interactions over a observed tenure period.Domain: E-Commerce / Customer Relationship Management (CRM)Problem Type: Binary ClassificationTotal Instances (Rows): 5,630 rows (exceeds the >500 instance assignment constraint)Total Attributes (Columns): 20 feature attributes + 1 target column (exceeds the >12 feature assignment constraint)

Target variable: Churn
Binary indicator representing whether a customer left the platform
83.2 % of dataset continue to make purchase and rest has left the platform

#### Numerical Features (14 Attributes)
| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `Tenure` | Continuous (`float64`) | Months customer has been associated with platform *(Contains missing values)* |
| `CityTier` | Discrete (`int64`) | Tier rating of customer's city (1, 2, or 3) |
| `WarehouseToHome` | Continuous (`float64`) | Distance from warehouse to delivery address in km *(Contains missing values)* |
| `HourSpendOnApp` | Continuous (`float64`) | Average hours spent on app/website daily *(Contains missing values)* |
| `NumberOfDeviceRegistered` | Discrete (`int64`) | Total devices registered to account |
| `SatisfactionScore` | Discrete (`int64`) | Customer rating score (1 to 5) |
| `NumberOfAddress` | Discrete (`int64`) | Total registered delivery addresses |
| `Complain` | Binary (`int64`) | Complaint raised in last month (`1` = Yes, `0` = No) |
| `OrderAmountHikeFromlastYear`| Continuous (`float64`) | Percentage increase in order value year-over-year *(Contains missing values)* |
| `CouponUsed` | Continuous (`float64`) | Count of discount coupons redeemed *(Contains missing values)* |
| `OrderCount` | Continuous (`float64`) | Total count of orders placed *(Contains missing values)* |
| `DaySinceLastOrder` | Continuous (`float64`) | Days elapsed since last order *(Contains missing values)* |
| `CashbackAmount` | Continuous (`float64`) | Average monthly cashback earned |
| `CustomerID` | Identifier (`int64`) | Unique customer ID *(Dropped during preprocessing)* |

#### Categorical Features (6 Attributes)
| Feature Name | Data Type | Categories / Values | Description |
| :--- | :--- | :--- | :--- |
| `PreferredLoginDevice` | Text (`object`) | *Mobile Phone, Phone, Computer* | Primary device used for login |
| `PreferredPaymentMode` | Text (`object`) | *Debit Card, Credit Card, E-Wallet, UPI, COD, Net Banking* | Primary payment method |
| `Gender` | Text (`object`) | *Male, Female* | Customer gender |
| `PreferedOrderCat` | Text (`object`) | *Laptop & Accessory, Mobile Phone, Fashion, Grocery, Others* | Most ordered product category |
| `MaritalStatus` | Text (`object`) | *Single, Married, Divorced* | Customer marital status |

---

We have removed the index and customer id columns as they do not contribute to the classification

# Models Used
| Model | Accuracy | Precision | Recall | F1 Score | MCC | ROC AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **K-Nearest Neighbors** | 92.27% | 87.05% | 63.68% | 73.56% | 0.7032 | 80.88% |
| **Support Vector Machine** | 91.47% | 89.17% | 56.32% | 69.03% | 0.6667 | 77.46% |
| **Decision Tree** | 90.94% | 80.99% | 60.53% | 69.28% | 0.6503 | 78.82% |
| **Logistic Regression** | 88.81% | 74.24% | 51.58% | 60.87% | 0.5582 | 73.97% |
| **Gaussian Naive Bayes** | 70.07% | 32.20% | 70.00% | 44.11% | 0.3115 | 70.04% |

# Model Observation
| ML Model Name | Observation about Model Performance |
| :--- | :--- |
| **K-Nearest Neighbors (KNN)** | **Top Overall Performer:** Achieved the highest Accuracy (92.27%), F1-Score (73.56%), MCC (0.7032), and ROC AUC (80.88%). `StandardScaler` scaling on numeric features allowed distance metrics to effectively separate churned vs. retained clusters. |
| **Support Vector Machine (SVM)** | **Highest Precision:** Delivered the top Precision (89.17%) and second-highest MCC (0.6667). Ideal for minimizing False Positives (avoiding misidentifying loyal customers as churn risks), benefiting significantly from the RBF kernel and scaled input space. |
| **Decision Tree** | **Strong Non-Linear Classifier:** Ranked second in Recall (60.53%) and third in F1-Score (69.28%) and MCC (0.6503). Handled feature interactions naturally (e.g., tenure vs. satisfaction score) without assuming linear relationships. |
| **Logistic Regression** | **Moderate Linear Baseline:** Displayed solid Accuracy (88.81%) but suffered on Recall (51.58%) and MCC (0.5582), indicating that linear decision boundaries struggle to capture complex non-linear customer churn patterns. |
| **Gaussian Naive Bayes** | **Highest Recall / Weakest Precision:** Achieved the highest Recall (70.00%) among all models, effectively flagging churners, but suffered from extremely poor Precision (32.20%) and the lowest MCC (0.3115) due to its strict feature-independence assumption. |