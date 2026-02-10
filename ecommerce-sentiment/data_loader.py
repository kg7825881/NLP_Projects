# data_loader.py
import pandas as pd

def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Ensure we have a 'Review' and 'Rating' column
        # Rename columns if necessary to match standard names
        return df
    else:
        # Fallback to a sample dataset if no file is uploaded
        data = {
            'Review': [
                "The battery life is amazing but the camera sucks.",
                "Delivery was terrible! Arrived 3 days late.",
                "I absolutely love the design, it's so sleek.",
                "Waste of money. Stopped working after a week.",
                "Customer service was helpful when I called."
            ],
            'Rating': [3, 1, 5, 1, 4]
        }
        return pd.DataFrame(data)