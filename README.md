# Taxi Trip Duration Prediction

This project predicts taxi trip durations using machine learning.

## Setup
1. Clone the repository:
   ```sh
   git clone <repo_url>
   cd taxi-duration-prediction
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### Train the Model
Run:
```sh
python train.py
```

### Make Predictions
Run:
```sh
python main.py
```

## Files
- **config.py** - Configuration (file paths, features)
- **main.py** - Loads model & predicts
- **model.py** - ML model pipeline
- **preprocess.py** - Data preparation
- **train.py** - Trains & evaluates model
- **utils.py** - Helper functions

## License
Open-source under MIT License.
