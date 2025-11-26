# VINS-Mo: A Robust and Versatile Monocular Visual-Inertial State Estimator

**VINS-Mo** is an enhanced visual-inertial state estimation framework based on the classic [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono). This project aims to fuse the perceptual advantages of deep learning with the geometric precision of traditional VINS, tackling inherent challenges in **initialization speed**, **scale drift**, and **robustness in dynamic environments**.

---

## 🚀 Core Features & Innovations

Building upon VINS-Mono, this project focuses on three key enhancements:

### 1. Deep-Fast-Init: Fast Initialization with Depth Aid

Traditional monocular VINS, including VINS-Mono, relies heavily on sufficient motion parallax to recover scale during initialization. This often leads to slow or failed initialization in scenarios with slow or limited motion.

* **Our Solution:** We leverage a single-frame depth estimation network (e.g., MiDaS or other lightweight models).
* **Advantage:** During the initialization phase, the predicted depth provides scale priors for 2D feature points. This enables **rapid or even zero-motion initialization**, significantly improving the system's responsiveness and usability.
* **Status:** ✅ **Done** (Branch: `feat/deep-init`)

### 2. Deep-Sensor: Fusing Depth as a Virtual Sensor

Monocular VINS systems inherently suffer from scale drift, where the absolute scale of the trajectory becomes unreliable over long runs.

* **Our Solution:** We treat the depth estimator not just as an initialization tool, but as a continuous "Pseudo-Depth Sensor."
* **Advantage:** The predicted depth measurements are integrated as new factors **tightly-coupled into the back-end nonlinear optimization**. This allows the system to continuously observe and correct the absolute scale, effectively mitigating drift and transforming the system into a "Monocular-Depth-Inertial" VIO.
* **Status:** 🚧 **In Progress** (Branch: `feat/deep-sensor`)

### 3. Dynamic-Remove: Robustness in Dynamic Environments

VINS-Mono, like most SLAM/VIO systems, operates on a static world assumption—that all objects in the scene are stationary. In real-world environments with dynamic objects like pedestrians or vehicles, this assumption is violated, leading to tracking failures and severe pose estimation errors.

* **Our Solution:** (Planned) We will integrate a dynamic object removal module.
* **Advantage:** By using semantic segmentation (e.g., YOLOv8, Mask R-CNN) or motion consistency checks, the system will **actively identify and reject** feature points that lie on dynamic objects. This ensures that only features from the static background are used for state estimation.
* **Status:** 💡 **Planned**

---

## 🎯 Project Goals

The ultimate goal of VINS-Mo is to deliver a **high-precision**, **highly robust**, and **fast-to-deploy** monocular VINS solution that retains the computational efficiency of VINS-Mono while being capable of handling more challenging real-world applications.
