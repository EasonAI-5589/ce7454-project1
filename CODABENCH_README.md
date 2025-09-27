# CE7454 CelebAMask Face Parsing Project - Complete Guide

## 📋 Project Overview

**Face parsing** assigns pixel-wise labels for each semantic component (eyes, nose, mouth). The goal of this mini challenge is to design and train a face parsing network using the CelebAMask-HQ Dataset.

### Dataset Specifications
- **Mini-dataset**: 1000 training and 100 validation pairs of images
- **Resolution**: 512 x 512 for both images and annotations
- **Classes**: 19 semantic classes (background + 18 facial components)
- **Format**: Following CelebAMask-HQ setting (see [color definition](color) and [Document](Document))

## 🗓️ Important Dates

### Development Phase
- **Start**: 05 Sep 2025 00:00 (GMT+8)
- **End**: 04 Oct 2025 23:59 (GMT+8)

### Test Phase
- **Start**: 04 Oct 2025 23:59 (GMT+8)
- **End**: 14 Oct 2025 23:59 (GMT+8)

### Critical Deadlines
- **🚨 DO NOT SUBMIT AFTER 00:00 AM OCT 15 🚨**
- **Test phase ends**: 00:00 Oct 15 (UTC+8, beginning of the day)
- **⚠️ ALL SUBMISSIONS AFTER 00:00 OCT 15 WILL BE CONSIDERED LATE**

## 📁 Dataset Access

### Where to Find the Data
**The dataset is available in the "Files" section of this CodaBench page.**

### Dataset Structure
The mini-dataset consists of:
1. **1000 training image pairs** (images + masks)
2. **100 validation image pairs** (images + masks)
3. **Resolution**: 512 x 512 pixels

### Data Usage Instructions
From the Development Phase description:
> "Divide the 1,000 images in the 'train' folder into a training set and a validation set, then train your neural network. Submit the inference results of the 100 images in the 'test' folder for online verification of your model's accuracy."

## 🎯 Model Requirements

### Parameter Constraints
- **Maximum trainable parameters**: < 1,821,085
- **Check with**: `sum(p.numel() for p in model.parameters())`
- **Must report** parameter count in your final report

### Training Restrictions
- ❌ **No external data** and pretrained models allowed
- ❌ **No ensemble** of models
- ❌ **No knowledge distillation**
- ✅ **Only allowed**: Train from scratch using the 1000 image pairs in given training dataset

## 📤 CodaBench Submission Requirements

### Test Phase Submission
- **Test images available**: From the "Files" page (beginning Oct 04, 2025, 12:00 AM)
- **Maximum submissions**: 10 submissions total
- **Submission time**: All times refer to SUBMISSION TIME in UTC+8

### Output Format Requirements
**CRITICAL**: The final output is **NOT** an RGB image but a **SINGLE-CHANNEL image**

#### Correct Submission Structure
```
submission.zip (correct)
├── solution/
└── masks/
    ├── 0001.png
    ├── 0002.png
    └── ...
```

#### File Requirements
- **ALL MASKS MUST BE PLACED IN A FOLDER NAMED "masks"**
- **Use same filename** as input image (e.g., input: 0001.png → output: 0001.png)
- **Single-channel PNG** format
- **Check mask format** against [THIS SAMPLE](THIS_SAMPLE) if score is very low

### Code Integration Requirements
- **DON'T FORGET TO INCLUDE YOUR CODE AND MODELS IN THE SUBMISSIONS!**
- A sample submission file can be found at [this link](this_link)
- **ONLY the FINAL submission** will be used as the official result

## 📊 Evaluation Process

### Online Evaluation
- **Began**: Oct 04, 2025, at 12:00 AM
- **Platform**: CodaBench online evaluation system
- **Metric**: F-Score between predicted masks and ground truth

### Submission Limits and Policies
- **Maximum 10 submissions**: Failed submissions WILL NOT COUNT toward this limit
- **No resubmission** close to deadline to avoid CodaBench bugs
- **Late submissions**: All submissions after Oct 15 00:00 considered late
- **Bug handling**: Due to CodaBench bugs, some submissions may hang - DO NOT RESUBMIT

## 🔧 Technical Tips

### Performance Boost Techniques
1. **Data augmentation**
2. **Deeper model** (but be careful of parameter constraint)

### Debugging
- **File structure errors**: 99.9% chance due to incorrect folder structure
- **Low scores**: Check mask format against provided sample
- **Hanging submissions**: Known CodaBench bug - be patient, don't resubmit

## 📚 Resources

### Reference Materials
- **CelebA Dataset**: [GitHub - switchablenorms/CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)
- **All online resources** allowed for this project
- **Specify codebase** used in your report

### Computational Resources
- **Amazon EC2** or **Google CoLab** recommended
- **AWS Educate**: Students can get free $100 credit
- **Recommended**: g2.2xlarge instances running Ubuntu
- **Runtime**: $100 allows ~6 days continuous g2.2xlarge GPU usage

## 📋 NTULearn Submission Requirements

### Submission Deadline
- **Date**: Before 14 October 2025 11:59 PM
- **Format**: Single ZIP file named with matric number (e.g., A12345678B.zip)
- **Policy**: Only latest submission counted

### Required Files in ZIP
1. **📄 Technical Report** (PDF, ≤5 A4 pages, Arial 10 font)
   - Model description
   - Loss functions and operations used
   - **F-measure on validation dataset**
   - **Number of parameters** of your model

2. **🖼️ Test Results**
   - Predicted masks from 100 test images
   - Place in subfolder with same filenames as input

3. **💻 Source Code**
   - All necessary codes used in project

4. **🎯 Model Checkpoint**
   - Model weights of submitted model

5. **📝 README.txt**
   - Description of submitted files
   - References to third-party libraries used
   - Testing instructions (which script to run)

## ⚠️ Important Warnings

### Submission Warnings
- **CodaBench bugs**: Some submissions may hang due to confirmed bugs
- **Don't submit** close to deadline to avoid issues
- **Cancelled submissions** still count toward the 10-submission limit
- **Time reference**: All times are SUBMISSION TIME, not evaluation completion time

### Model Compliance
- **Parameter verification**: Essential to check parameter count
- **Output format**: Must be single-channel, not RGB
- **File structure**: Critical for successful evaluation

## 🎯 Success Strategy

1. **Early Development**: Use development phase effectively
2. **Parameter Management**: Stay well under 1,821,085 limit
3. **Format Compliance**: Test output format early
4. **Strategic Submissions**: Plan your 10 submissions carefully
5. **Documentation**: Keep detailed records for report writing

## 📞 Support

- **Issues**: Check FAQ section first
- **Bugs**: Confirmed CodaBench issues - be patient
- **Questions**: Use forum for technical discussions

---
*Last updated based on CodaBench competition page as of September 2025*