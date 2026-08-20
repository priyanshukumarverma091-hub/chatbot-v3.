| `chatbot_model.pkl` | Exported trained classifier. |
| `vectorizer_v3.pkl` | Exported TF-IDF vectorizer. |
| `label_encoder_v3.pkl` | Exported intent-label encoder. |

> The generated `.pkl` files are produced by the notebook and do not need to be committed if you prefer users to train the model locally.

## Getting started

### Option 1: Run in Google Colab

1. Upload `chatbot_v3 (1).ipynb` to [Google Colab](https://colab.research.google.com/).
2. Run the import cells.
3. When prompted, upload `attendance_v3_dataset_1800.json`.
4. Run the training and evaluation cells.
5. Download the exported model artifacts if needed.

### Option 2: Run locally

Install the required packages:

```bash
pip install scikit-learn numpy matplotlib joblib
```

Place `attendance_v3_dataset_1800.json` in the same directory as the notebook, then open and run it with Jupyter:

```bash
jupyter notebook
```

For local use, you can remove or replace the Google Colab-specific upload and download cells.

## Dataset format

The training dataset is expected to be a JSON array structured like this:

```json
[
  {
    "text": "What attendance percentage is needed to sit for exams?",
    "intent": "attendance_requirement"
  },
  {
    "text": "My attendance is below the requirement. What can I do?",
    "intent": "attendance_shortage"
  }
]
```

## Example inference

```python
import joblib

model = joblib.load("chatbot_model.pkl")
vectorizer = joblib.load("vectorizer_v3.pkl")
encoder = joblib.load("label_encoder_v3.pkl")

query = "Can I write my exam with 68% attendance?"
features = vectorizer.transform([query])
intent = encoder.inverse_transform(model.predict(features))[0]

print(intent)
```

## Important notes

- The chatbot predicts an intent; it does not itself generate official policy answers.
- Attendance rules vary by institution, programme, and academic year. Connect each intent to approved, up-to-date policy content before using the chatbot for student-facing decisions.
- Evaluate using a held-out test set before relying on the model, especially after adding new intents or training examples.
- Do not expose personal student records or use the model as the sole authority for exam eligibility.

## Possible improvements

- Add response templates or a knowledge base for each intent.
- Add confidence scores and a fallback response for uncertain predictions.
- Track precision, recall, and a confusion matrix on a held-out evaluation split.
- Provide a Streamlit or Flask interface.
- Expand the dataset with multilingual and institution-specific phrasing.

## License

Choose and add a license before publishing. The MIT License is a common choice for open-source educational projects.
# chatbot-v3.
A Jupyter Notebook implementation of an interactive chatbot.
