# Dental X-ray Analysis

AI-powered automated dental diagnosis support system for X-ray image analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub issues](https://img.shields.io/github/issues/dental-xray/dental-xray)](https://github.com/dental-xray/dental-xray/issues)
[![GitHub stars](https://img.shields.io/github/stars/dental-xray/dental-xray)](https://github.com/dental-xray/dental-xray/stargazers)

## Overview

This project provides an AI-powered system for automated analysis of dental X-ray images (panoramic and periapical radiographs). The system can detect and analyze:

- **Tooth Detection & Classification**: Precise tooth localization and dental numbering
- **Caries Detection**: Identification of cavities from initial to advanced stages
- **Periodontal Disease Assessment**: Bone loss evaluation and gum disease indicators
- **Dental Implant Detection**: Implant position and condition assessment
- **Anomaly Detection**: Detection of unusual dental conditions and pathologies

## Installation

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/dental-xray/dental-xray.git
cd dental-xray
```

2. **Create virtual environment**
```bash
pyenv virtualenv 3.12.9 dental-xray
pyenv local dental-xray
```

3. **Install dependencies**
```bash
make install
```

### Web Interface

Launch the web application:
```bash
make build-api-amd64
make run-api
make run-frontend
```
Navigate to `http://localhost:8080` in your browser


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


**⚠️ Medical Disclaimer**: This system is designed as a diagnostic aid tool. All final diagnoses must be made by qualified dental professionals. The software is not intended to replace professional medical judgment and should be used only as a supplementary tool in clinical practice.
