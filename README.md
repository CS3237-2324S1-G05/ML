# Smart Carpark ML Codebase 🚗💡

Welcome to the GitHub repository for the Smart Carpark system's ML codes! This code repository contains the detection models for the .

This is done as part of the Project for CS3237 - Introduction to Internet of Things, AY23/24 Semester 1.

Done by Group 5!

## Repository Structure 🗂️

Here's a guide to what you'll find in this repository:

- `extra/`: Contains script to format images and training of a object detection model using YOLOv8

- `final/`: Includes Python script for Car Plate Recognition and Human Detection and folders to store images received using Flask

- `licensePlateDetector/`: Dataset of Car Plates in YOLOv8 format

- `peopleDetection/`: Custom Dataset of human in YOLOv8 format

## Setup & Installation ⚙️

To set up your environment for contributing to this project, please follow these steps:

1. Ensure you have the Python 3.10 installed on your computer.
2. Ensure you have MQTT server set up.
3. Install all necessary libraries as listed in the Dependencies section.

## Usage 🛠️

To use the code:

1. Navigate to the "final" project folder.
2. Open the Python sketch (.py file) with your preferred text editor/IDE.
3. Using the terminal/command prompt, navigate into the same folder and type `python <script name>` in the command line

## Dependencies 📚

This project requires the following libraries using pip or conda:

- For `final/`:
  - ultralytics@8.0.195
  - flask@2.3.3
  - paddleocr@2.7.0.3
  - paddlepaddle@2.5.2
  - pillow@10.0.0

Please ensure that you have these libraries installed before running the project.

---
We're excited to see how you'll help us drive the future of smart parking! 🌟

Happy coding! 🚀👩‍💻👨‍💻\
CS3237 Group G05