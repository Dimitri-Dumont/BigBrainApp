# AAI-501

## Brain Tumor Detection Project Overview

Our Brain Tumor Detection project focuses on leveraging advanced deep learning techniques to accurately identify and classify brain tumors from MRI images. We trained our model using two powerful model architectures: YOLOv8 for object detection and ResNet50 for image classification. Through extensive training and optimization, we achieved a mean Average Precision (mAP) at IoU=0.50 (mAP50) of approximately 0.79 and a mAP across IoU thresholds from 0.50 to 0.95 (mAP50-95) of about 0.52.

To make our model accessible and easy to use, we have deployed the model API to Google Cloud Run, ensuring scalability and reliability. Additionally, we have set up a simple web interface for users to interact with our model, which can be accessed [here](https://ui-service-dot-eloquent-env-430802-s8.uw.r.appspot.com/).

This project hopes to represents the possibilities in the application of AI for medical diagnostics, providing a robust tool for healthcare professionals to assist in the early detection and treatment of brain tumors.

### Contributors:
Dimitri Dumont

Gurleen Virk

Mythreyi Thirumalai