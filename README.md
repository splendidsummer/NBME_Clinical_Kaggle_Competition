# NBME_Clinical_Kaggle_Competition

## Overview 

### Description 

When you visit a doctor, how they interpret your symptoms can determine whether your diagnosis is accurate. By the time they’re licensed, physicians have had a lot of practice writing patient notes that document the history of the patient’s complaint, physical exam findings, possible diagnoses, and follow-up care. Learning and assessing the skill of writing patient notes requires feedback from other doctors, a time-intensive process that could be improved with the addition of machine learning.

Until recently, the Step 2 Clinical Skills examination was one component of the United States Medical Licensing Examination® (USMLE®). The exam required test-takers to interact with Standardized Patients (people trained to portray specific clinical cases) and write a patient note. Trained physician raters later scored patient notes with rubrics that outlined each case’s important concepts (referred to as features). The more such features found in a patient note, the higher the score (among other factors that contribute to the final score for the exam).

However, having physicians score patient note exams requires significant time, along with human and financial resources. Approaches using natural language processing have been created to address this problem, but patient notes can still be challenging to score computationally because features may be expressed in many ways. For example, the feature "loss of interest in activities" can be expressed as "no longer plays tennis." Other challenges include the need to map concepts by combining multiple text segments, or cases of ambiguous negation such as “no cold intolerance, hair loss, palpitations, or tremor” corresponding to the key essential “lack of other thyroid symptoms.”

In this competition, you’ll identify specific clinical concepts in patient notes. Specifically, you'll develop an automated method to map clinical concepts from an exam rubric (e.g., “diminished appetite”) to various ways in which these concepts are expressed in clinical patient notes written by medical students (e.g., “eating less,” “clothes fit looser”). Great solutions will be both accurate and reliable.

If successful, you'll help tackle the biggest practical barriers in patient note scoring, making the approach more transparent, interpretable, and easing the development and administration of such assessments. As a result, medical practitioners will be able to explore the full potential of patient notes to reveal information relevant to clinical skills assessment.

This competition is sponsored by the National Board of Medical Examiners® (NBME®). Through research and innovation, NBME supports medical school and residency program educators in addressing issues around the evolution of teaching, learning, technology, and the need for meaningful feedback. NBME offers high-quality assessments and educational services for students, professionals, educators, regulators, and institutions dedicated to the evolving needs of medical education and health care. To serve these communities, NBME collaborates with a diverse and comprehensive array of practicing health professionals, medical educators, state medical board members, test developers, academic researchers, scoring experts and public representatives.

### Evaluation 

This competition is evaluated by a **micro-averaged F1 score**.

For each instance, we predict a set of character spans. A character span is a pair of indexes representing a range of characters within a text. A span i j represents the characters with indices i through j, inclusive of i and exclusive of j. In Python notation, a span i j is equivalent to a slice **i:j**.

For each instance there is a collection of ground-truth spans and a collection of predicted spans. The spans we delimit with a semicolon, like: **0 3**; **5 9**.

We score each character index as:

**TP** if it is within both a ground-truth and a prediction,
**FN** if it is within a ground-truth but not a prediction, and,
**FP** if it is within a prediction but not a ground truth.
Finally, we compute an **overall F1 score from the TPs, FNs, and FPs aggregated across all instances**.

$$F1 Score = \frac{2(P*R)}{P+R}$$ where P = precision, and R = the recall of the classification model 
$$ Precision = \frac{TP}{TP + FP} \quad \quad Recall = \frac{TP}{TP + FN}$$

#### Example
Suppose we have an instance: 
| ground-truth | Prediction | 
|--------------|------------|
| 0 3;     3 5 | 2 5; 7 9; 2 3|
These spans give the sets of indices:
| ground-truth | Prediction |    
|--------------|------------|
| 0 1 2 3 4    | 2 3 4 7 8  |

Therefore we compute:
* TP = size of $\{ 2, 3, 4 \} = 3$
* FN = size of $\{ 0, 1\}$
* FP = size of $\{7, 8\}$ 

#### Sample Submission
```
id,location
00016_000,0 100
00016_001,
00016_002,200 250;300 500
...
```
For 00016_002 you should give predictions for feature 000 in patient note 00016. 

### Code Competition 

**This is a Code Competition**  
Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

CPU Notebook <= 9 hours run-time
GPU Notebook <= 9 hours run-time
Internet access disabled
Freely & publicly available external data is allowed, including pre-trained models
Submission file must be named submission.csv
Please see the [Code Competition FAQ](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ) for more information on how to submit. And review the [code debugging doc](https://www.kaggle.com/code-competition-debugging) if you are encountering submission errors.

## Data  

### Dataset Description
The text data presented here is from the USMLE® Step 2 Clinical Skills examination, a medical licensure exam. This exam measures a trainee's ability to recognize pertinent clinical facts during encounters with standardized patients.

During this exam, each test taker sees a Standardized Patient, a person trained to portray a clinical case. After interacting with the patient, the test taker documents the relevant facts of the encounter in a patient note. Each patient note is scored by a trained physician who looks for the presence of certain key concepts or features relevant to the case as described in a rubric. The goal of this competition is to develop an automated way of identifying the relevant features within each patient note, with a special focus on the patient history portions of the notes where the information from the interview with the standardized patient is documented.

### Important Terms
Clinical Case: The scenario (e.g., symptoms, complaints, concerns) the Standardized Patient presents to the test taker (medical student, resident or physician). Ten clinical cases are represented in this dataset.
Patient Note: Text detailing important information related by the patient during the encounter (physical exam and interview).
Feature: A clinically relevant concept. A rubric describes the key concepts relevant to each case.
### Training Data
patient_notes.csv - A collection of about 40,000 Patient Note history portions. Only a subset of these have features annotated. You may wish to apply unsupervised learning techniques on the notes without annotations. The patient notes in the test set are not included in the public version of this file.
pn_num - A unique identifier for each patient note.
case_num - A unique identifier for the clinical case a patient note represents.
pn_history - The text of the encounter as recorded by the test taker.
features.csv - The rubric of features (or key concepts) for each clinical case.
feature_num - A unique identifier for each feature.
case_num - A unique identifier for each case.
feature_text - A description of the feature.
train.csv - Feature annotations for 1000 of the patient notes, 100 for each of ten cases.
id - Unique identifier for each patient note / feature pair.
pn_num - The patient note annotated in this row.
feature_num - The feature annotated in this row.
case_num - The case to which this patient note belongs.
annotation - The text(s) within a patient note indicating a feature. A feature may be indicated multiple times within a single note.
location - Character spans indicating the location of each annotation within the note. Multiple spans may be needed to represent an annotation, in which case the spans are delimited by a semicolon ;.
### Example Test Data
To help you author submission code, we include a few example instances selected from the training set. When your submitted notebook is scored, this example data will be replaced by the actual test data. The patient notes in the test set will be added to the patient_notes.csv file. These patient notes are from the same clinical cases as the patient notes in the training set. There are approximately 2000 patient notes in the test set.

test.csv - Example instances selected from the training set.
sample_submission.csv - A sample submission file in the correct format.

## Data EDA 

## 