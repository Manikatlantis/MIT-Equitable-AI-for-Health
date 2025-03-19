# BTTAI_AJL_Team_2
# Equitable AI for Dermatology
## Repo for Equitable AI for Dermatology Kaggle 2025 Competition

# Project Overview 
Skin condition classification models often exhibit bias due to the lack of diverse training data, leading to disparities in healthcare, particularly for individuals with darker skin tones. This project, part of the Spring 2025 AI Studio, aims to address this challenge by developing a machine learning model that classifies 16 different skin conditions across diverse skin tones while ensuring fairness and explainability.

The competition is hosted by Break Through Tech AI and the Algorithmic Justice League (AJL), with an emphasis on equity and transparency in AI-driven healthcare solutions.

# Dataset ->
This project utilizes a subset of the Fitzpatrick17k dataset, a collection of 17,000 labeled dermatological images. The dataset includes:

* 4,500 images across 16 skin conditions

* Images labeled with Fitzpatrick skin tone scale (1-6), indicating skin type diversity

* Annotations for medical conditions sourced from DermaAmin and Atlas Dermatologico
### Fitzpatrick Skin Types (FST) ->

![](Skin_types.png)

| Skin type | Typical features                                      | Tanning ability                         |
|-----------|------------------------------------------------------|-----------------------------------------|
| **I**     | Pale white skin, blue/green eyes, blond/red hair     | Always burns, does not tan             |
| **II**    | Fair skin, blue eyes                                 | Burns easily, tans poorly              |
| **III**   | Darker white skin                                    | Tans after initial burn                |
| **IV**    | Light brown skin                                     | Burns minimally, tans easily           |
| **V**     | Brown skin                                          | Rarely burns, tans darkly easily       |
| **VI**    | Dark brown or black skin                            | Never burns, always tans darkly        |
### Skin Conditions ->
![](skin_condition_distribution.png)

### Distribution os Skin Types->
![](fst_distribution.png)

### Feature Columns ->

| Column                | Data Type | Kaggle Description                                      | Our Understanding                                       |
|-----------------------|----------|---------------------------------------------------------|---------------------------------------------------------|
| `md5hash`            | Object   | An alphanumeric hash serving as a unique identifier; file name of an image without .jpg | Unique hash value using MD5 hashing algorithm. |
| `fitzpatrick_scale`  | int64    | Integer in the range [-1, 0) and [1, 6] indicating self-described FST *Fitzpatrick Skin Type (FST)* | -1 = Missing/Unlabeled data  <br> 1 - 6 = Fitzpatrick skin types <br> Type 1 = Very fair, burns easily <br> Type 2 = Deeply pigmented, never burns. |
| `fitzpatrick_centaur`| int64    | Integer in the range [-1, 0) and [1, 6] indicating FST assigned by Centaur Labs, a medical data annotation firm | -1 = Missing/Unlabeled data <br> 1 - 6 = Fitzpatrick skin types <br> **Difference from `fitzpatrick_scale`**: <br> 1) `fitzpatrick_scale` is self-reported. <br> 2) `fitzpatrick_centaur` is annotated by medical experts. |
| `label`              | Object   | String indicating medical diagnosis; the target for this competition | Medical diagnosis (our target label) <br> **Example** → melanoma, psoriasis, eczema etc. |
| `nine_partition_label` | Object   | String indicating one of nine diagnostic categories | **Categories** <br> 1. benign-dermal <br> 2. benign-epidermal <br> 3. inflammatory <br> 4. malignant-cutaneous-lymphoma <br> 5. malignant-dermal <br> 6. malignant-epidermal <br> 7. malignant-melanoma <br> 8. ?? <br> 9. ?? |
| `three_partition_label` | Object | String indicating one of three diagnostic categories | **Categories** <br> 1. benign <br> 2. malignant <br> 3. non-neoplastic |
| `qc`                 | Object   | Quality control check by a Board-certified dermatologist. <br> The `qc` column has responses for 500 observations of the FULL FitzPatrick dataset. Only about 90 observations in the train set have responses, and only about 30 observations in the test set have responses. | **Possible Values** <br> [nan <br> '1 Diagnostic' <br> '2 Characteristic' <br> '3 Potentially' <br> '3 Wrongly labelled' <br> '4 Other'] <br> |
| `ddi_scale`          | int64    | A column used to reconcile this dataset with another dataset (may not be relevant). <br> | Used to merge another dataset with our Fitzpatrick dataset. May not be useful and can overlook for now. |

The dataset is provided as train.csv (with labels) and test.csv (unlabeled for submission). Image files are stored in an images.zip archive, with train/test splits structured into folders.

### QC Labels ->

| Code | Label                 | Meaning                                                                 | Count |
|------|-----------------------|-------------------------------------------------------------------------|-------|
| 1    | **Diagnostic**         | The image provides a **good example** of the skin condition and is **useful for diagnosis**. | 348   |
| 2    | **Characteristic**     | The image **may show** the skin condition, but **isn't conclusive** for diagnosis. | 32    |
| 3    | **Wrongly labeled**    | The image **does not correspond** to the labeled condition; it was **misclassified**. | 17    |
| 4    | **Other**              | The image does **not fit any specific category**, possibly due to **image quality issues**. | 10    |
| 5    | **Potentially Diagnostic** | The image is **unclear**, meaning **further testing** is needed to confirm its diagnostic value. | 97    |
