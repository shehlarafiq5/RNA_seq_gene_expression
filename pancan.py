#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd

# Replace 'your_file.tsv' with the actual file name
df = pd.read_csv('data.csv')

# Display the first few rows
print(df.shape)


# In[8]:


# Check the shape of the dataframe
num_instances, num_features = df.shape

# Print the number of instances and features
print(f"Number of instances: {num_instances}")
print(f"Number of features: {num_features}")


# In[9]:



labels_df = pd.read_csv('labels.csv')  # Assuming labels are stored here
# Define a mapping
label_mapping = {'BRCA': 0, 'KIRC': 1, 'COAD': 2, 'LUAD': 3, 'PRAD': 4}

# Map the categorical labels to numeric values
labels_df['Label_Numeric'] = labels_df['Class'].map(label_mapping)

print(labels_df.head())

# Merge datasets based on a common column (replace 'Sample_ID' with actual key)
merged_df = pd.merge(df, labels_df, on='Sample_id')
print(merged_df.head())


# In[10]:


print(merged_df.shape)


# In[11]:


# Check the distribution of tumor types
import matplotlib.pyplot as plt
distribution = labels_df['Label_Numeric'].value_counts()

# Map numeric labels back to tumor type names for readability
distribution_named = labels_df['Class'].value_counts()

print("Numeric Distribution:")
print(distribution)

print("\nTumor Type Distribution:")
print(distribution_named)



# Bar plot for the tumor type distribution
distribution_named.plot(kind='bar', color='skyblue', edgecolor='black')
# plt.title('Distribution of Tumor Types')
plt.xlabel('Tumor Types')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[12]:



# Merge the features with the labels based on Sample_ID
merged_df = pd.merge(df, labels_df[['Sample_id', 'Label_Numeric']], on='Sample_id')

# Separate features (X) and target (y)
X = merged_df.drop(columns=['Sample_id', 'Label_Numeric'])  # Drop non-feature columns
y = merged_df['Label_Numeric']

# Check the shapes
print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(y)
print(X)


# In[13]:


from sklearn.model_selection import train_test_split

# Assuming you have your features (X) and target (y) defined
# X = X_normalized_filled  # Your feature set
y = y  # Your target variable

# Perform the train-test split (80% training, 20% testing, you can adjust the ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now you can perform preprocessing and feature selection only on the training set


# # PCA

# In[14]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Step 1: Fit PCA on the training data (normalize your features first if not done)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)

# Step 2: Transform the test data
X_test_pca = pca.transform(X_test)

# Step 3: Check how many components were selected
print(f"Number of components to retain 95% variance: {pca.n_components_}")

# Optional: explained variance plot
plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_)*100, marker='o')
plt.axhline(y=95, color='r', linestyle='--', label='95% variance')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance (%)')
# plt.title('Explained Variance by PCA Components')
plt.legend()
plt.grid(True)
plt.savefig("pca_variance.png", dpi=300)
plt.show()


# In[15]:


# Convert y_train integers to cancer type strings
# Mapping from class ID to cancer type
label_map = {0: 'BRCA', 1: 'KIRC', 2: 'COAD', 3: 'LUAD', 4: 'PRAD'}

cancer_labels = y_train.map(label_map)

# Create a DataFrame for the first 2 principal components
pca_df = pd.DataFrame(X_train_pca[:, :2], columns=['PC1', 'PC2'])
pca_df['Cancer Type'] = cancer_labels.values

# Scatter plot
plt.figure(figsize=(8, 4))
for cancer_type in pca_df['Cancer Type'].unique():
    subset = pca_df[pca_df['Cancer Type'] == cancer_type]
    plt.scatter(subset['PC1'], subset['PC2'], label=cancer_type, alpha=0.7)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
# plt.title('PCA Scatter Plot (First 2 Components) by Cancer Type')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('pca_scatter_plot.png', dpi=300) 
plt.show()


# In[16]:


from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit on training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train_pca)
X_test_scaled = scaler.transform(X_test_pca)


# In[17]:


print(X_train_scaled.shape)
print(X_test_scaled)


# In[18]:


print(y_train)


# In[19]:


print(y_test)


# # ML algorithms

# In[20]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc
)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from itertools import cycle


# In[21]:


classifiers = {
    'Neural Network': MLPClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}


# In[22]:


results = pd.DataFrame(columns=['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC'])


# In[23]:


# Binarize the output for ROC AUC calculations
y_test_binarized = label_binarize(y_test, classes=np.unique(y))

for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled) if hasattr(clf, "predict_proba") else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Calculate AUC-ROC
    if y_pred_proba is not None:
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(y_test_binarized.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        auc_roc = roc_auc["micro"]
    else:
        auc_roc = float('nan')  # Not available for classifiers without predict_proba
    
    # Append metrics to the results DataFrame
    results = results.append({
        'Classifier': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC-ROC': auc_roc
    }, ignore_index=True)


# In[24]:


# Set up the plot
plt.figure(figsize=(12, 10))

# Define colors
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink'])

for (name, clf), color in zip(classifiers.items(), colors):
    if hasattr(clf, "predict_proba"):
        y_pred_proba = clf.predict_proba(X_test_scaled)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(y_test_binarized.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.plot(fpr["micro"], tpr["micro"], color=color, lw=2,
                 label=f'{name} (AUC = {roc_auc["micro"]:.2f})')

# Plot random chance line
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')

# Set plot labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Multiple Classifiers')
plt.legend(loc="lower right")
plt.show()


# In[25]:


results.to_csv('model_performance_metrics.csv', index=False)


# In[26]:


print(y_test)


# In[27]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, classification_report
)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from itertools import cycle

# Classifiers dictionary
classifiers = {
    'Neural Network': MLPClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Replace with your actual data
# X, y = ...  # Your features and labels

# Encode labels if not already encoded
class_names = ['LUAD', 'BRCA', 'COAD', 'KIRC', 'PRAD']
y_binarized = label_binarize(y, classes=class_names)

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_test_binarized = label_binarize(y_test, classes=class_names)

# Store results
results = []
per_class_metrics = []

for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled) if hasattr(clf, "predict_proba") else None

    # Overall metrics
    overall_accuracy = accuracy_score(y_test, y_pred)
    overall_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    overall_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    overall_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    overall_auc = roc_auc_score(y_test_binarized, y_pred_proba, multi_class='ovr') if y_pred_proba is not None else np.nan

    results.append({
        'Classifier': name,
        'Accuracy': overall_accuracy,
        'Precision': overall_precision,
        'Recall': overall_recall,
        'F1 Score': overall_f1,
        'AUC-ROC': overall_auc
    })

    # Per-class metrics
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    for cls in class_names:
        if cls in report:
            per_class_metrics.append({
                'Classifier': name,
                'Class': cls,
                'Precision': report[cls]['precision'],
                'Recall': report[cls]['recall'],
                'F1 Score': report[cls]['f1-score'],
                'Support': report[cls]['support']
            })

# Convert to DataFrames
df_results = pd.DataFrame(results)
df_per_class = pd.DataFrame(per_class_metrics)

# Save to CSV
df_results.to_csv("ml_performance_without_balancing.csv", index=False)
df_per_class.to_csv("ml_per_class_metrics.csv", index=False)

print("Performance metrics saved.")


# In[28]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, auc
)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import UndefinedMetricWarning

# Suppress undefined metric warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Define classifiers
classifiers = {
    'Neural Network': MLPClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Classes (make sure these match your labels exactly)
classes = np.unique(y_train)
y_test_binarized = label_binarize(y_test, classes=classes)

# Store detailed results
detailed_results = []

for name, clf in classifiers.items():
    # Train
    clf.fit(X_train_scaled, y_train)

    # Predict
    y_pred = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled) if hasattr(clf, "predict_proba") else None

    # Overall accuracy
    overall_accuracy = accuracy_score(y_test, y_pred)

    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=classes, zero_division=0)

    # Per-class AUC
    auc_scores = []
    if y_pred_proba is not None:
        for i, cls in enumerate(classes):
            if len(np.unique(y_test_binarized[:, i])) < 2:
                auc_scores.append(np.nan)
            else:
                fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
                auc_scores.append(auc(fpr, tpr))
    else:
        auc_scores = [np.nan] * len(classes)

    # Store per-class results
    for i, cls in enumerate(classes):
        detailed_results.append({
            'Classifier': name,
            'Class': cls,
            'Accuracy': overall_accuracy,
            'Precision': precision[i],
            'Recall': recall[i],
            'F1 Score': f1[i],
            'AUC-ROC': auc_scores[i]
        })

# Save to CSV
detailed_df = pd.DataFrame(detailed_results)
detailed_df.to_csv("ml_performance_per_class_without_balancing.csv", index=False)

print("✅ Performance metrics saved to 'ml_performance_per_class_without_balancing.csv'")


# # bar chart

# In[ ]:


# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Define the performance metrics
# data = {
#     "Classifier": [
#         "Neural Network", "Logistic Regression", "Decision Tree",
#         "Naive Bayes", "Random Forest", "SVM", "K-Nearest Neighbors", "Gradient Boosting"
#     ],
#     "Accuracy": [1.0, 1.0, 0.9689, 0.6646, 1.0, 1.0, 1.0, 0.9876],
#     "Precision": [1.0, 1.0, 0.9720, 0.7328, 1.0, 1.0, 1.0, 0.9880],
#     "Recall": [1.0, 1.0, 0.9689, 0.6646, 1.0, 1.0, 1.0, 0.9876],
#     "F1 Score": [1.0, 1.0, 0.9693, 0.6604, 1.0, 1.0, 1.0, 0.9875],
#     "AUC-ROC": [1.0, 1.0, 0.9806, 0.7904, 1.0, 1.0, 1.0, 0.9999]
# }

# # Create DataFrame
# df = pd.DataFrame(data)

# # Set the bar width and positions
# bar_width = 0.15
# x = np.arange(len(df["Classifier"]))

# # Create the figure and axis
# fig, ax = plt.subplots(figsize=(10, 6))

# # Plot each metric as grouped bars
# metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
# colors = ["blue", "orange", "green", "red", "purple"]

# for i, metric in enumerate(metrics):
#     ax.bar(x + i * bar_width, df[metric], width=bar_width, label=metric, color=colors[i])

# # Set labels and title
# ax.set_xlabel("Classifier", fontsize=12)
# ax.set_ylabel("Performance Score", fontsize=12)
# ax.set_title("Comparison of Classifier Performance", fontsize=14)
# ax.set_xticks(x + bar_width * 2)
# ax.set_xticklabels(df["Classifier"], rotation=45, ha="right")

# # Add legend
# ax.legend()

# # Save as PNG file
# plt.tight_layout()
# plt.savefig("classifier_performance.png", dpi=300)
# plt.show()


# # horizontal chart

# In[ ]:


# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Define the performance metrics
# data = {
#     "Classifier": [
#         "Neural Network", "Logistic Regression", "Decision Tree",
#         "Naive Bayes", "Random Forest", "SVM", "K-Nearest Neighbors", "Gradient Boosting"
#     ],
#     "Accuracy": [1.0, 1.0, 0.9689, 0.6646, 1.0, 1.0, 1.0, 0.9876],
#     "Precision": [1.0, 1.0, 0.9720, 0.7328, 1.0, 1.0, 1.0, 0.9880],
#     "Recall": [1.0, 1.0, 0.9689, 0.6646, 1.0, 1.0, 1.0, 0.9876],
#     "F1 Score": [1.0, 1.0, 0.9693, 0.6604, 1.0, 1.0, 1.0, 0.9875],
#     "AUC-ROC": [1.0, 1.0, 0.9806, 0.7904, 1.0, 1.0, 1.0, 0.9999]
# }

# # Create DataFrame
# df = pd.DataFrame(data)

# # Define bar width and positions
# bar_width = 0.15
# y = np.arange(len(df["Classifier"]))

# # Create the figure and axis
# fig, ax = plt.subplots(figsize=(8, 6))

# # Plot each metric as horizontal bars
# metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
# colors = ["blue", "orange", "green", "red", "purple"]

# for i, metric in enumerate(metrics):
#     ax.barh(y + i * bar_width, df[metric], height=bar_width, label=metric, color=colors[i])

# # Set labels and title
# ax.set_ylabel("Classifier", fontsize=12)
# ax.set_xlabel("Performance Score", fontsize=12)
# ax.set_title("Classifier Performance Comparison", fontsize=14)
# ax.set_yticks(y + bar_width * 2)
# ax.set_yticklabels(df["Classifier"])

# # Add legend
# ax.legend(loc="lower right")

# # Save as PNG file
# plt.tight_layout()
# plt.savefig("horizontal_classifier_performance.png", dpi=300)
# plt.show()


# # heat map

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create DataFrame
df = pd.DataFrame(data).set_index("Classifier")

# Create heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".3f")

# Title and labels
plt.title("Classifier Performance Heatmap", fontsize=14)
plt.xlabel("Performance Metrics", fontsize=12)
plt.ylabel("Classifier", fontsize=12)

# Save as PNG
plt.savefig("heatmap_classifier_performance.png", dpi=300)
plt.show()


# # spider chart

# In[ ]:


# import numpy as np
# import matplotlib.pyplot as plt

# # Define metrics and classifiers
# metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
# classifiers = [
#     "Neural Network", "Logistic Regression", "Decision Tree",
#     "Naive Bayes", "Random Forest", "SVM", "K-Nearest Neighbors", "Gradient Boosting"
# ]

# # Convert data into an array
# values = np.array([data[metric] for metric in metrics])

# # Define angles for radar chart
# angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
# angles += angles[:1]  # Close the circle

# # Create the figure
# fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# # Plot each classifier
# for i, classifier in enumerate(classifiers):
#     classifier_values = values[:, i].tolist()
#     classifier_values += classifier_values[:1]  # Close the circle
#     ax.plot(angles, classifier_values, label=classifier, linewidth=2)

# # Set labels
# ax.set_xticks(angles[:-1])
# ax.set_xticklabels(metrics, fontsize=12)
# ax.set_yticklabels([])

# # Add legend
# ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))

# # Title and Save
# plt.title("Classifier Performance Radar Chart", fontsize=14)
# plt.savefig("radar_classifier_performance.png", dpi=300)
# plt.show()


# # SMOTE

# In[29]:


from imblearn.over_sampling import SMOTE
from collections import Counter

# Assume X and y are your features and labels
# smote = SMOTE(sampling_strategy={0: 300, 1: 300, 2: 300, 3: 300, 4: 300}, random_state=42)
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled,y_train)
# X_resampled, y_resampled = smote.fit_resample(X_train_pca,y_train)

# Print new class distribution
print("After SMOTE:", Counter(y_resampled))


# In[30]:


import matplotlib.pyplot as plt
from collections import Counter

# Mapping of labels to cancer types
label_map = {0: 'BRCA', 1: 'KIRC', 2: 'COAD', 3: 'LUAD', 4: 'PRAD'}

# Count the number of samples in each class after SMOTE
counter = Counter(y_resampled)

# Prepare data for plotting
cancer_types = [label_map[i] for i in sorted(counter.keys())]
counts = [counter[i] for i in sorted(counter.keys())]

# Plot
plt.figure(figsize=(8, 6))
bars = plt.bar(cancer_types, counts, color='skyblue')
# plt.title("Class Distribution After SMOTE")
plt.xlabel("Cancer Type")
plt.ylabel("Number of Samples")
plt.grid(axis='y', linestyle='', alpha=0.7)

# Adding count labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# In[31]:


import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# # Assume you have X (features) and y (labels)
# # Splitting dataset before applying SMOTE to prevent data leakage
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# # Feature scaling
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Apply SMOTE
# smote = SMOTE(sampling_strategy={0: 300, 1: 300, 2: 300, 3: 300, 4: 300}, random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
# print("After SMOTE:", Counter(y_resampled))

# Define classifiers
classifiers = {
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Store results
results = []

# Train and evaluate classifiers
for name, clf in classifiers.items():
    clf.fit(X_resampled, y_resampled)  # Train on balanced data
    y_pred = clf.predict(X_test_scaled)  # Predict on test data
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    auc_roc = roc_auc_score(y_test, clf.predict_proba(X_test_scaled), multi_class="ovr")

    # Append results
    results.append([name, accuracy, precision, recall, f1, auc_roc])

# Convert results to DataFrame
df_results = pd.DataFrame(results, columns=["Classifier", "Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"])

# Save to CSV
df_results.to_csv("ml_performance_balanced.csv", index=False)

print("Saved performance metrics to 'ml_performance_balanced.csv'")


# In[32]:


ml=pd.read_csv("ml_performance_balanced.csv")
print(ml)


# In[33]:


from sklearn.metrics import classification_report, roc_auc_score

print(f"\nClassifier: {name}")

# Predict on test data
y_pred = clf.predict(X_test_scaled)
y_proba = clf.predict_proba(X_test_scaled)

# Average metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
auc_roc = roc_auc_score(y_test, y_proba, multi_class="ovr")

# Append overall metrics
results.append([name, accuracy, precision, recall, f1, auc_roc])

# Per-class performance
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["LUAD", "BRCA", "COAD", "KIRC", "PRAD"]))


# In[34]:


from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
import pandas as pd

# Binarize the output for multi-class AUC-ROC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
n_classes = y_test_bin.shape[1]

# Store per-class results
per_class_results = []

for name, clf in classifiers.items():
    clf.fit(X_resampled, y_resampled)
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)

    # Get classification report as dictionary
    report = classification_report(
        y_test, y_pred, output_dict=True, target_names=["LUAD", "BRCA", "COAD", "KIRC", "PRAD"]
    )

    # Extract overall accuracy (global)
    overall_accuracy = report["accuracy"]

    # Calculate per-class AUC-ROC
    auc_scores = roc_auc_score(y_test_bin, y_proba, average=None, multi_class="ovr")

    # For each class, append metrics
    for i, cancer_type in enumerate(["LUAD", "BRCA", "COAD", "KIRC", "PRAD"]):
        precision = report[cancer_type]["precision"]
        recall = report[cancer_type]["recall"]
        f1 = report[cancer_type]["f1-score"]
        support = report[cancer_type]["support"]
        auc = auc_scores[i]

        per_class_results.append([
            name, cancer_type, overall_accuracy, precision, recall, f1, auc, support
        ])

# Create DataFrame
df_per_class = pd.DataFrame(per_class_results, columns=[
    "Classifier", "Cancer Type", "Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC", "Support"
])

# Save to CSV
df_per_class.to_csv("ml_performance_per_class.csv", index=False)

print("Saved detailed performance per cancer type to 'ml_performance_per_class.csv'")


# In[35]:


ml1=pd.read_csv("model_performance_metrics.csv")
print(ml1)


# # Lime Explanations

# In[48]:


import lime
import lime.lime_tabular
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Assuming you have already split and scaled your data, and applied SMOTE:
# X_train_scaled, X_test_scaled, y_train, y_test, and classifiers are defined
class_names = ['BRCA', 'KIRC', 'COAD', 'LUAD', 'PRAD']  # Class labels

# Initialize the LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled, 
    mode="classification", 
    feature_names=X.columns, 
    class_names=class_names,
    
#     class_names=[str(i) for i in range(5)],  # Assuming 5 classes in your problem
    discretize_continuous=True
)

# Choose an instance from the test set to explain
# Let's explain the first instance in the test set as an example
instance_to_explain = X_test_scaled[0]

# Select the classifier (Random Forest in this example)
clf = classifiers["Random Forest"]

# Explain the prediction
explanation = explainer.explain_instance(instance_to_explain, clf.predict_proba)

# Visualize the explanation
explanation.show_in_notebook()

# If you want to save the explanation, you can do so with:
# explanation.save_to_file('lime_explanation.html')


# In[52]:


# Set BRCA class index (assuming it's 0)
brca_index = 0

# Select an instance to explain
instance_to_explain = X_test_scaled[1]

# Generate LIME explanation for BRCA
explanation = explainer.explain_instance(
    instance_to_explain,
    clf.predict_proba,
    labels=[brca_index]
)

# Show explanation in Jupyter
explanation.show_in_notebook(labels=[brca_index])

# Save explanation to HTML
explanation.save_to_file("lime_explanation_instance0_BRCA.html")


# In[37]:


import lime
import lime.lime_tabular
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Class labels
class_names = ['BRCA', 'KIRC', 'COAD', 'LUAD', 'PRAD']  

# Initialize the LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled, 
    mode="classification", 
    feature_names=X.columns, 
    class_names=class_names,
    discretize_continuous=True
)

# Choose an instance from the test set to explain
instance_to_explain = X_test_scaled[6]  

# Select the classifier (Random Forest in this example)
clf = classifiers["Random Forest"]

# Loop through each class and generate explanations
for class_index, class_name in enumerate(class_names):
    explanation = explainer.explain_instance(
        instance_to_explain, 
        clf.predict_proba, 
        labels=[class_index]  # Focus on one class at a time
    )
    
    print(f"\nLIME Explanation for class: {class_name}")
    explanation.show_in_notebook()  # Display explanation in Jupyter Notebook
    
    # Save explanation as an HTML file for reference
    explanation.save_to_file(f'updated lime_explanation_{class_name}.html')


# In[38]:


print(type(y_test))


# In[35]:


# import matplotlib.pyplot as plt

# # Loop through instances in the test set
# for i, instance_to_explain in enumerate(X_test_scaled):
#     actual_label = y_test.iloc[i]
#     predicted_label = clf.predict([instance_to_explain])[0]

#     if actual_label == predicted_label:
#         print(f"\nInstance {i}: Actual label = {class_names[actual_label]}, Predicted label = {class_names[predicted_label]}")
        
#         explanation = explainer.explain_instance(
#             instance_to_explain,
#             clf.predict_proba,
#             labels=[predicted_label]
#         )

#         # Extract explanation for the predicted label
#         exp_list = explanation.as_list(label=predicted_label)

#         # Prepare data for plotting
#         features, weights = zip(*exp_list)
#         plt.figure(figsize=(8, 6))
#         plt.barh(features, weights, color='skyblue')
#         plt.xlabel("Feature Weight")
#         plt.title(f"LIME Explanation\nInstance {i} - Predicted: {class_names[predicted_label]}")
#         plt.tight_layout()

#         # Save plot to PNG
#         plt.savefig(f'lime_explanation_instance_{i}_class_{class_names[predicted_label]}.png')
#         plt.close()


# In[1]:


get_ipython().system('pip install selenium')
get_ipython().system('pip install webdriver-manager')


# In[5]:


get_ipython().system('pip install lime')


# In[43]:


chrome_options.binary_location = "/path/to/google-chrome"


# In[44]:


chrome_options.binary_location = "C:/Program Files/Google/Chrome/Application/chrome.exe"


# In[46]:


import lime
import lime.lime_tabular
import os
import matplotlib.pyplot as plt

# Create folder for saving LIME plots
os.makedirs("lime_pngs", exist_ok=True)

# Loop through test instances
for i, instance in enumerate(X_test_scaled):
    actual_label = y_test.iloc[i]
    predicted_label = clf.predict([instance])[0]

    if actual_label == predicted_label:
        print(f"\nInstance {i}: Actual = {class_names[actual_label]}, Predicted = {class_names[predicted_label]}")

        # Generate LIME explanation
        explanation = explainer.explain_instance(instance, clf.predict_proba, labels=[predicted_label])

        # Plot and save the explanation using matplotlib
        fig = explanation.as_pyplot_figure(label=predicted_label)
        fig.tight_layout()
        fig.savefig(f"lime_pngs/lime_explanation_instance_{i}_class_{class_names[predicted_label]}.png")
        plt.close(fig)  # Close to prevent memory buildup


# In[40]:


import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

# Make a folder for outputs
os.makedirs("lime_pngs", exist_ok=True)

# Loop through test instances
for i, instance in enumerate(X_test_scaled):
    actual_label = y_test.iloc[i]
    predicted_label = clf.predict([instance])[0]

    if actual_label == predicted_label:
        print(f"\nInstance {i}: Actual = {class_names[actual_label]}, Predicted = {class_names[predicted_label]}")

        # Create LIME explanation
        explanation = explainer.explain_instance(instance, clf.predict_proba, labels=[predicted_label])

        # Save as HTML
        html_path = f'lime_explanation_instance_{i}_class_{class_names[predicted_label]}.html'
        explanation.save_to_file(html_path)

        # Set up headless Chrome
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')

#         driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
        driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(), options=chrome_options)
        driver.set_window_size(1200, 800)

        # Open HTML and take screenshot
        driver.get("file://" + os.path.abspath(html_path))
        time.sleep(2)  # wait to fully render
        png_path = f"lime_pngs/lime_explanation_instance_{i}_class_{class_names[predicted_label]}.png"
        driver.save_screenshot(png_path)
        driver.quit()


# In[80]:


# Loop through instances in the test set
for i, instance_to_explain in enumerate(X_test_scaled):
    # Get the actual class label (numerical) for the instance
    actual_label = y_test.iloc[i]  # Correct indexing for Pandas Series

    # Predict the class for the instance
    predicted_label = clf.predict([instance_to_explain])[0]

    # Check if actual and predicted labels match
    if actual_label == predicted_label:
        print(f"\nInstance {i}: Actual label = {class_names[actual_label]}, Predicted label = {class_names[predicted_label]}")
        
        # Generate LIME explanation for the matched class
        explanation = explainer.explain_instance(
            instance_to_explain, 
            clf.predict_proba, 
            labels=[predicted_label]  # Focus on the predicted class
        )
        
        explanation.show_in_notebook()  # Display explanation in Jupyter Notebook
        explanation.save_to_file(f'updated_lime_explanation_instance_{i}_class_{class_names[predicted_label]}.html')


# In[79]:


import matplotlib.pyplot as plt

# Define number of instances to explain
num_instances_to_explain = 5  # You can change this
num_classes = len(class_names)

for i, instance_to_explain in enumerate(X_test_scaled[:num_instances_to_explain]):
    actual_label = y_test.iloc[i]
    predicted_label = clf.predict([instance_to_explain])[0]
    
    print(f"\nInstance {i}: Actual = {class_names[actual_label]}, Predicted = {class_names[predicted_label]}")

    # Create LIME explanations for all classes
    explanation = explainer.explain_instance(
        instance_to_explain,
        clf.predict_proba,
        top_labels=num_classes  # Get explanations for top N classes
    )

    # Plot subplots for each class
    fig, axs = plt.subplots(1, num_classes, figsize=(20, 4))
    fig.suptitle(f'LIME Explanations for Instance {i}', fontsize=16)

    for j, class_idx in enumerate(explanation.top_labels):
        ax = axs[j]
        exp = explanation.as_list(label=class_idx)
        
        # Unzip the explanation into features and weights
        features, weights = zip(*exp)
        
        ax.barh(features, weights, color='skyblue')
        ax.set_title(f'Class: {class_names[class_idx]}')
        ax.invert_yaxis()  # Highest contribution on top
        ax.set_xlabel('Weight')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# In[30]:


get_ipython().system('pip install shap')


# In[31]:


import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Initialize SHAP Explainer
explainer = shap.Explainer(clf, X_train_scaled)

# Choose an instance from the test set to explain
instance_to_explain = X_test_scaled[6].reshape(1, -1)  

# Compute SHAP values for the instance
shap_values = explainer(instance_to_explain)

# Loop through each class sfigeparately
for class_index, class_name in enumerate(class_names):
    print(f"SHAP Explanation for class: {class_name}")
    
    # Extract SHAP values for the specific class (fixing shape issue)
    shap_class_values = shap.Explanation(
        values=shap_values.values[0, :, class_index],  # Extract a single row correctly
        base_values=shap_values.base_values[0, class_index],  # Extract corresponding base value
        data=shap_values.data[0],  # Extract feature values for instance
        feature_names=X.columns
    )

    # Generate waterfall plot
    shap.waterfall_plot(shap_class_values)
    plt.show()


# In[81]:


import shap

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test_scaled)
print(type(X_test_scaled))


# In[82]:


import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define feature names (modify if actual names exist)
num_features = X_test_scaled.shape[1]
feature_names = [f"Feature_{i}" for i in range(num_features)]  # Replace with actual feature names if available

# Define class names (modify if needed)
# class_names = ["Class 0", "Class 1", "Class 2"]  # Adjust based on your dataset

# Initialize SHAP Explainer
explainer = shap.Explainer(clf, X_train_scaled)

# Choose an instance from the test set to explain
instance_index = 6  # Modify this index as needed
instance_to_explain = X_test_scaled[instance_index].reshape(1, -1)  

# Compute SHAP values for the instance
shap_values = explainer(instance_to_explain)

# Loop through each class separately
for class_index, class_name in enumerate(class_names):
    print(f"\nSHAP Explanation for class: {class_name}")

    # Extract SHAP values correctly
    shap_class_values = shap.Explanation(
        values=shap_values.values[0, :, class_index],  # Correct slicing
        base_values=shap_values.base_values[0, class_index],  # Correct base value
        data=shap_values.data[0],  # Feature values for instance
        feature_names=feature_names  # Use defined feature names
    )

    # Generate SHAP force plot for the class
    shap.force_plot(shap_class_values.base_values, shap_class_values.values, shap_class_values.data, feature_names=feature_names)
    
    # Save force plot as an image (optional)
    shap.save_html(f"shap_force_plot_class_{class_index}.html", shap.force_plot(shap_class_values.base_values, shap_class_values.values, shap_class_values.data, feature_names=feature_names))


# In[83]:


import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define feature names
num_features = X_test_scaled.shape[1]
feature_names = [f"Feature_{i}" for i in range(num_features)]  # Or use real names if available

# Define class names
class_names = clf.classes_ if hasattr(clf, "classes_") else [f"Class {i}" for i in range(clf.n_classes_)]

# Initialize SHAP Explainer
explainer = shap.Explainer(clf, X_train_scaled)

# Choose a test instance to explain
instance_index = 6
instance_to_explain = X_test_scaled[instance_index].reshape(1, -1)

# Compute SHAP values
shap_values = explainer(instance_to_explain)

# Generate SHAP explanation per class
for class_index, class_name in enumerate(class_names):
    print(f"\nSHAP Explanation for class: {class_name}")
    
    shap_class_values = shap.Explanation(
        values=shap_values.values[0, :, class_index],
        base_values=shap_values.base_values[0, class_index],
        data=shap_values.data[0],
        feature_names=feature_names
    )

    # Display force plot
    shap.force_plot(
        shap_class_values.base_values,
        shap_class_values.values,
        shap_class_values.data,
        feature_names=feature_names,
        matplotlib=True
    )

    # Save to HTML (optional)
    shap.save_html(
        f"shap_force_plot_class_{class_index}.html",
        shap.force_plot(
            shap_class_values.base_values,
            shap_class_values.values,
            shap_class_values.data,
            feature_names=feature_names
        )
    )


# In[91]:


class_index = 3  # Change from 0 to 4 based on the class you're explaining
shap.summary_plot(shap_values_all.values[:, :, class_index],
                  X_test_scaled,
                  feature_names=feature_names)


# In[92]:


shap.summary_plot(shap_values_all.values[:, :, class_index],
                  X_test_scaled,
                  feature_names=feature_names,
                  plot_type="bar")


# In[93]:


import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define feature names
num_features = X_test_scaled.shape[1]
feature_names = [f"Feature_{i}" for i in range(num_features)]  # Or replace with actual feature names

# Define actual class names (replace with your real class labels)
class_names = ["BRCA", "KIRC", "LUAD", "COAD", "PRAD"]  # Example: Update as needed

# Initialize SHAP Explainer
explainer = shap.Explainer(clf, X_train_scaled)

# Choose a test instance to explain
instance_index = 6
instance_to_explain = X_test_scaled[instance_index].reshape(1, -1)

# Compute SHAP values
shap_values = explainer(instance_to_explain)

# Generate SHAP explanation per class
for class_index, class_name in enumerate(class_names):
    print(f"\nSHAP Explanation for class: {class_name}")
    
    shap_class_values = shap.Explanation(
        values=shap_values.values[0, :, class_index],
        base_values=shap_values.base_values[0, class_index],
        data=shap_values.data[0],
        feature_names=feature_names
    )

    # Display force plot
    shap.force_plot(
        shap_class_values.base_values,
        shap_class_values.values,
        shap_class_values.data,
        feature_names=feature_names,
        matplotlib=True
    )

    # Save interactive force plot to HTML with class name
    shap.save_html(
        f"shap_force_plot_{class_name}.html",
        shap.force_plot(
            shap_class_values.base_values,
            shap_class_values.values,
            shap_class_values.data,
            feature_names=feature_names
        )
    )


# In[96]:


import shap
import matplotlib.pyplot as plt
import numpy as np
import os

# Optional: Create directory to save plots
os.makedirs("shap_plots", exist_ok=True)

# Define feature names (update if you have real names)
feature_names = [f"Feature_{i}" for i in range(X_test_scaled.shape[1])]

# Define actual class names
class_names = ["BRCA", "KIRC", "LUAD", "COAD", "PRAD"]  # Update as per clf.classes_

# Loop over each class to generate summary and waterfall plots
for class_idx, class_name in enumerate(class_names):
    print(f"Generating plots for class: {class_name}")
    
    # --- Summary plot (dot) ---
    plt.figure()
    shap.summary_plot(shap_values_all.values[:, :, class_idx], X_test_scaled, feature_names=feature_names, show=False)
#     plt.title(f"SHAP Summary Plot - {class_name}")
    plt.tight_layout()
    plt.savefig(f"shap_plots/summary_dot_{class_name}.png", dpi=300)
    plt.close()

    # --- Summary plot (bar) ---
    plt.figure()
    shap.summary_plot(shap_values_all.values[:, :, class_idx], X_test_scaled, feature_names=feature_names, plot_type="bar", show=False)
#     plt.title(f"SHAP Bar Plot - {class_name}")
    plt.tight_layout()
    plt.savefig(f"shap_plots/summary_bar_{class_name}.png", dpi=300)
    plt.close()

    # --- Waterfall plot for a specific instance (e.g., index 6) ---
    plt.figure()
    shap.plots.waterfall(shap_values_all[6, :, class_idx], show=False)
#     plt.title(f"SHAP Waterfall - Instance 6 - {class_name}")
    plt.savefig(f"shap_plots/waterfall_instance6_{class_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

# --- Identify top contributing feature across all classes ---
mean_abs_shap = np.mean(np.abs(shap_values_all.values), axis=0)  # shape: (num_features, num_classes)
mean_feature_contrib_across_classes = np.mean(mean_abs_shap, axis=1)  # average per feature

top_feature_idx = np.argmax(mean_feature_contrib_across_classes)
top_feature = feature_names[top_feature_idx]
print(f"\n🔍 Top contributing feature across all classes: {top_feature}")


# In[52]:


import shap
import numpy as np
import matplotlib.pyplot as plt

# Define feature names (modify if actual names exist)
# num_features = X_test_scaled.shape[1]
# feature_names = [f"Feature_{i}" for i in range(num_features)]  # Replace with actual feature names if available
# Ensure feature names are correctly assigned
feature_names = list(X.columns)
# Initialize SHAP Explainer
explainer = shap.Explainer(clf, X_train_scaled)

# Compute SHAP values for the entire test set
shap_values = explainer(X_test_scaled)

# Convert SHAP values to NumPy array
shap_values_array = shap_values.values  # Shape: (num_samples, num_features, num_classes)

# Loop through each class separately
for class_index, class_name in enumerate(class_names):
    print(f"\nGenerating SHAP summary plot for class: {class_name}")

    # Extract SHAP values for the specific class
    shap_class_values = shap_values_array[:, :, class_index]  # Shape: (num_samples, num_features)

    # Generate summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_class_values, X_test_scaled, feature_names=feature_names)
    plt.title(f"SHAP Summary Plot for Class: {class_name}")
    plt.show()


# In[44]:


import shap
import numpy as np
import matplotlib.pyplot as plt

# Define feature names (modify if actual names exist)
num_features = X_test_scaled.shape[1]
feature_names = [f"Feature_{i}" for i in range(num_features)]  # Replace with actual feature names if available

# Initialize SHAP Explainer
explainer = shap.Explainer(clf, X_train_scaled)

# Compute SHAP values for the entire test set
shap_values = explainer(X_test_scaled)

# Extract SHAP values as a NumPy array
shap_values_array = shap_values.values  # Shape: (num_samples, num_features, num_classes)

# Compute mean absolute SHAP values across samples and sum across classes
shap_values_mean = np.mean(np.abs(shap_values_array), axis=0)  # Shape: (num_features, num_classes)
shap_values_mean_sum = np.sum(shap_values_mean, axis=1)  # Summing across classes → Shape: (num_features,)

# Get indices of top 15 features
top_features_idx = np.argsort(shap_values_mean_sum)[-15:]  # Get top 15 important features
top_feature_names = [feature_names[i] for i in top_features_idx]

# ✅ Extract SHAP values correctly for the summary plot
shap_values_selected = shap_values_array[:, top_features_idx, :].sum(axis=2)  # Summing across classes

# SHAP Summary Plot for top 15 features
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_selected, X_test_scaled[:, top_features_idx], feature_names=top_feature_names)
plt.title("SHAP Summary Plot (Top 15 Features)")
plt.show()


# In[47]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
shap_values_normalized = scaler.fit_transform(shap_values_selected)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_normalized, X_test_scaled[:, top_features_idx], feature_names=top_feature_names)
plt.title("Normalized SHAP Summary Plot")
plt.show()


# In[48]:


scaling_factor = 1e10  # Adjust this factor if needed
shap_values_rescaled = shap_values_selected * scaling_factor

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_rescaled, X_test_scaled[:, top_features_idx], feature_names=top_feature_names)
plt.title(f"SHAP Summary Plot (Scaled by {scaling_factor})")
plt.show()


# In[49]:


import shap
import numpy as np
import matplotlib.pyplot as plt

# Define feature names (modify if actual names exist)
num_features = X_test_scaled.shape[1]
feature_names = [f"Feature_{i}" for i in range(num_features)]  # Replace with actual feature names

# Initialize SHAP Explainer
explainer = shap.Explainer(clf, X_train_scaled)

# Compute SHAP values for the entire test set
shap_values = explainer(X_test_scaled)

# Extract SHAP values as a NumPy array
shap_values_array = shap_values.values  # Shape: (num_samples, num_features, num_classes)

# Compute mean absolute SHAP values across samples and sum across classes
shap_values_mean = np.mean(np.abs(shap_values_array), axis=0)  # Shape: (num_features, num_classes)
shap_values_mean_sum = np.sum(shap_values_mean, axis=1)  # Summing across classes → Shape: (num_features,)

# Get indices of top 15 features
top_features_idx = np.argsort(shap_values_mean_sum)[-15:]  # Get top 15 most important features
top_feature_names = [feature_names[i] for i in top_features_idx]
top_feature_importance = shap_values_mean_sum[top_features_idx]

# 🔵 **Bar Plot**
plt.figure(figsize=(10, 6))
plt.barh(top_feature_names, top_feature_importance, color='steelblue')
plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature Name")
plt.title("Feature Importance (SHAP Values)")
plt.gca().invert_yaxis()  # Invert y-axis for better visualization
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


# In[ ]:




