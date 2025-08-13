# ASL Dataset Browser - Oregon Institute of Technoogy - Applied Computing Program
This report analyzes the ASL dataset, providing insights into its structure and content.

## 1. CSV File Analysis

### Analyzing how2sign_realigned_train.csv
#### Head:
      VIDEO_ID  ...                                           SENTENCE
0  --7E2sU6zP4  ...  And I call them decorative elements because ba...
1  --7E2sU6zP4  ...  So they don't really have much of a symbolic m...
2  --7E2sU6zP4  ...  Now this is very, this is actually an insert o...
3  --7E2sU6zP4  ...  This is all the you know, take off on the idea...
4  --7E2sU6zP4  ...     It's almost has a feathery like posture to it.

[5 rows x 7 columns]
#### Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 31165 entries, 0 to 31164
Data columns (total 7 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   VIDEO_ID         31165 non-null  object 
 1   VIDEO_NAME       31165 non-null  object 
 2   SENTENCE_ID      31165 non-null  object 
 3   SENTENCE_NAME    31165 non-null  object 
 4   START_REALIGNED  31165 non-null  float64
 5   END_REALIGNED    31165 non-null  float64
 6   SENTENCE         31165 non-null  object 
dtypes: float64(2), object(5)
memory usage: 1.7+ MB

#### Description:
       START_REALIGNED  END_REALIGNED
count     31165.000000   31165.000000
mean         58.170693      65.024236
std          46.600799      46.997779
min           0.000000       0.190000
25%          22.600000      29.370000
50%          49.820000      56.770000
75%          81.600000      88.720000
max         376.300000     382.300000

### Analyzing how2sign_realigned_val.csv
#### Head:
      VIDEO_ID  ...                                           SENTENCE
0  -d5dN54tH2E  ...  We're going to work on a arm drill that will h...
1  -d5dN54tH2E  ...                       I call it painting the wall.
2  -d5dN54tH2E  ...  So we're going to go up and down; let's switch...
3  -d5dN54tH2E  ...                  And just let those fingers relax.
4  -d5dN54tH2E  ...          Now together you're going to go opposite.

[5 rows x 7 columns]
#### Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1741 entries, 0 to 1740
Data columns (total 7 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   VIDEO_ID         1741 non-null   object 
 1   VIDEO_NAME       1741 non-null   object 
 2   SENTENCE_ID      1741 non-null   object 
 3   SENTENCE_NAME    1741 non-null   object 
 4   START_REALIGNED  1741 non-null   float64
 5   END_REALIGNED    1741 non-null   float64
 6   SENTENCE         1741 non-null   object 
dtypes: float64(2), object(5)
memory usage: 95.3+ KB

#### Description:
       START_REALIGNED  END_REALIGNED
count      1741.000000    1741.000000
mean         54.025503      60.940948
std          41.523009      42.093858
min           0.000000       0.370000
25%          21.790000      29.130000
50%          47.450000      54.170000
75%          75.580000      82.880000
max         259.990000     267.930000

### Analyzing how2sign_realigned_test.csv
#### Head:
      VIDEO_ID  ...                                           SENTENCE
0  -fZc293MpJk  ...                                                Hi!
1  -fZc293MpJk  ...  The aileron is the control surface in the wing...
2  -fZc293MpJk  ...  By moving the stick, you cause pressure to inc...
3  -fZc293MpJk  ...  The elevator is the part that moves with the s...
4  -fZc293MpJk  ...  Therefore, it's either going uphill, downhill,...

[5 rows x 7 columns]
#### Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2357 entries, 0 to 2356
Data columns (total 7 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   VIDEO_ID         2357 non-null   object 
 1   VIDEO_NAME       2357 non-null   object 
 2   SENTENCE_ID      2357 non-null   object 
 3   SENTENCE_NAME    2357 non-null   object 
 4   START_REALIGNED  2357 non-null   float64
 5   END_REALIGNED    2357 non-null   float64
 6   SENTENCE         2357 non-null   object 
dtypes: float64(2), object(5)
memory usage: 129.0+ KB

#### Description:
       START_REALIGNED  END_REALIGNED
count      2357.000000    2357.000000
mean         54.912372      61.555248
std          42.159327      42.636204
min           0.000000       0.370000
25%          22.110000      29.050000
50%          48.090000      55.100000
75%          77.520000      83.920000
max         267.870000     296.940000

## 2. Directory Structure Analysis
- **Train Videos**: 31048 files
- **Train Keypoints**: 4896 files
- **Validation Videos**: 1739 files
- **Validation Keypoints**: 1739 files
- **Test Videos**: 2343 files
- **Test Keypoints**: 2343 files

## 3. Sample Keypoint File

### Analyzing sample keypoint file: test_2D_keypoints/openpose_output/json/FZLxEwsoc1c_3-8-rgb_front/FZLxEwsoc1c_3-8-rgb_front_000000000259_keypoints.json
#### File Content (first 200 chars):
{'version': 1.3, 'people': [{'person_id': [-1], 'pose_keypoints_2d': [658.764, 257.063, 0.862666, 680.297, 411.902, 0.822018, 568.599, 413.807, 0.675458, 531.443, 600.006, 0.779548, 633.239, 693.96, 0...
#### Data Structure:
- The file contains a dictionary with a 'people' key.
- There are 1 people detected in this frame.
- Each person has 'pose_keypoints_2d' with 75 values.

## 4. Correlating CSV Data with Files

### Checking correlation for a sample from `how2sign_realigned_train.csv`:
- **CSV Row:**
VIDEO_ID                                                 --7E2sU6zP4
VIDEO_NAME                                   --7E2sU6zP4-5-rgb_front
SENTENCE_ID                                           --7E2sU6zP4_10
SENTENCE_NAME                             --7E2sU6zP4_10-5-rgb_front
START_REALIGNED                                               129.06
END_REALIGNED                                                 142.48
SENTENCE           And I call them decorative elements because ba...
Name: 0, dtype: object
- **Expected Video File:** train_rgb_front_clips/raw_videos/--7E2sU6zP4_10-5-rgb_front.mp4
- **Video File Exists:** True
- **Expected Keypoint Directory:** train_2D_keypoints/openpose_output/json/--7E2sU6zP4_10-5-rgb_front
- **Keypoint Directory Exists:** False

# End of Analysis
@LordAmdal - Ahmed Ali 
