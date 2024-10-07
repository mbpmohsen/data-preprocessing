
# NASA Logs Preprocessing Project

## Overview
This project processes log data from NASA's JSONL (JSON Lines) format to prepare it for machine learning tasks, specifically anomaly detection. The project uses Python for loading, cleaning, and transforming the log data. The steps include parsing dates, handling missing values, vectorizing textual data, and scaling numeric fields.

The processed data is saved as a CSV file and can be used in further analysis or modeling.

## Features
- **JSONL Parsing**: Reads and processes JSONL (JSON Lines) formatted data.
- **Date Handling**: Parses `date.issued` and `date.available` fields and extracts useful features like year and month.
- **Text Processing**: Cleans and vectorizes the `description` field using TF-IDF (Term Frequency-Inverse Document Frequency).
- **Feature Scaling**: Scales numerical features such as the year and month of issuance, and the length of the description.
- **CSV Output**: Saves the preprocessed data into a CSV file for further analysis.

## Project Structure
```
.
├── README.md               # Project documentation
├── preprocessing.py        # Main preprocessing script
├── test_records.jsonl       # Sample NASA log file (JSON Lines format)
└── preprocessed_nasa_logs.csv  # Output file (created after running the script)
```

## Requirements
The following Python libraries are required to run this project:
- `pandas`
- `json`
- `scikit-learn`

You can install the required packages using `pip`:

```bash
pip install pandas scikit-learn
```

## Data
The input data is in JSONL (JSON Lines) format, where each line is a JSON object representing a NASA log entry. The file contains nested data such as creators, subjects, descriptions, and publication dates.

### Sample Input Data (JSONL format):
```json
{"creator": ["Cutright, Steve", "Moore, James B."], "date.available": ["2017-02-07"], "date.issued": ["20170109", "January 9, 2017"], "description": ["Advancements in aircraft electric propulsion ..."], "format": ["application/pdf"], "identifier": ["http://hdl.handle.net/2060/20170001321"], "title": ["Structural Design Exploration of an Electric Powered Multi-Propulsor Wing Configuration"], "type": ["Conference Paper"]}
{"creator": ["Zubair, Mohammad", "Warner, James E."], "date.available": ["2017-02-07"], "date.issued": ["20170109", "January 9, 2017"], "description": ["This work investigates novel approaches to probabilistic damage diagnosis ..."], "format": ["application/pdf"], "identifier": ["http://hdl.handle.net/2060/20170001319"], "title": ["Near Real-Time Probabilistic Damage Diagnosis Using Surrogate Modeling and High Performance Computing"], "type": ["Conference Paper"]}
```

## Preprocessing Steps
The preprocessing pipeline includes the following steps:

### 1. **Load JSONL File**
The script reads the NASA log file in JSON Lines format and loads each JSON object as a record in a pandas DataFrame.

### 2. **Handle Missing Values**
Missing values in the data are filled with the placeholder `'Unknown'`.

### 3. **Date Parsing**
- `date.issued` and `date.available` fields are parsed into datetime format.
- Year and month are extracted as new features (`year_issued`, `month_issued`).

### 4. **Description Processing**
The `description` field is processed to calculate the length of each description. If the description contains multiple entries, they are joined into a single string.

### 5. **Text Vectorization**
The `description` field is vectorized using the TF-IDF method, which converts text into numerical features that represent the importance of words across the log entries.

### 6. **Feature Scaling**
The script scales the numerical features (such as `year_issued`, `month_issued`, and `description_length`) to ensure that they lie within the same range, which is important for machine learning models.

### 7. **Save to CSV**
The processed data is saved as `preprocessed_nasa_logs.csv` for further use in analysis or modeling.

## How to Run the Project
1. **Clone the repository** (or copy the files to your local machine):

   ```bash
   git clone https://github.com/yourusername/nasa-logs-preprocessing.git
   cd nasa-logs-preprocessing
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt  # If you have a requirements.txt file, or manually use pip install
   ```

3. **Run the preprocessing script**:
   ```bash
   python preprocessing.py
   ```

   This will read the `test_records.jsonl` file, preprocess the data, and save the result in `preprocessed_nasa_logs.csv`.

## Example Output
The preprocessed output will look like this:
```
   creator          date.issued   description_length  tfidf_features ...
0  [Cutright, ...]  2017-01-09    567                 [0.12, 0.34, ...]
1  [Zubair, ...]    2017-01-09    345                 [0.11, 0.31, ...]
...
```

## Future Improvements
- **Anomaly Detection**: Build a machine learning model to detect anomalies in the log data, such as missing fields, unusual descriptions, or unusual publication dates.
- **Real-time Log Processing**: Adapt the script to process logs in real time, for example, by monitoring new logs generated by NASA systems.
- **Dashboard Integration**: Create a visualization dashboard using tools like Streamlit or Flask to visualize log entries and detected anomalies.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
This project was inspired by NASA's need for efficient log management and processing. Special thanks to the open-source community for providing the libraries used in this project.
